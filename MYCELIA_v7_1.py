# ============================================
# MYCELIA LM Architecture (v7.1) - Pure Model Definition
# T4-Optimized: d_model=512, n_layers=6, n_heads=8, max_seq_len=4096
# FIXED: PE uses actual_T (post-compression), not original T
# FIXED: Embedding vocab_size matches tokenizer exactly
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MyceliaConfig:
    d_model: int = 512 # <-- managable for T4
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = 151936  # Qwen-7B vocab size (will be overridden by tokenizer)
    max_seq_len: int = 4096
    fib_weights: Tuple = (5, 8, 13, 21, 34, 55)
    dissenter_threshold: float = 2.5
    dubito_threshold: float = 7.0
    consensus_rounds: int = 2
    # Compressor
    use_compression: bool = True
    compress_ratio: int = 8
    compress_window: int = 256
    compress_freq: int = 4

def get_sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional embeddings."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class GoldenDropout(nn.Module):
    def __init__(self):
        super().__init__()
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2
        self.keep_prob = float(1.0 / phi)
        self.scale = float(phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.rand_like(x) < self.keep_prob
            return x * mask.to(x.dtype) * self.scale
        return x

class MycelialAttention(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, return_heads: bool = True):
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = attn.softmax(-1)
        attn = self.dropout(attn)
        head_outputs = attn @ v
        out = head_outputs.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        if return_heads:
            return out, head_outputs
        return out, None

class MycelialConsensus(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        fib = self._generate_fibonacci(self.n_heads)
        self.register_buffer(
            'fib_weights',
            torch.tensor(fib, dtype=torch.float32) / sum(fib)
        )

    @staticmethod
    def _generate_fibonacci(n: int) -> list:
        seq = [1, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]
        
    def forward(self, head_outputs: torch.Tensor, step: int = 0):
        B, n_heads, T, d_head = head_outputs.shape
        weights = self.fib_weights.view(1, -1, 1, 1)
        weighted = head_outputs * weights
        consensus = weighted.sum(dim=1)
        mean_heads = head_outputs.mean(dim=1, keepdim=True)
        variance = (head_outputs - mean_heads).pow(2).mean(dim=1)
        max_variance = variance.mean(dim=-1).max().item()
        veto = max_variance > self.config.dissenter_threshold
        coherence = 1.0 - min(1.0, max_variance / self.config.dissenter_threshold)
        if veto:
            consensus = consensus * 0.85
        return consensus, veto, {'coherence': coherence, 'variance': max_variance}

class MycelialBlock(nn.Module):
    def __init__(self, config: MyceliaConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.attn = MycelialAttention(config)
        self.mycelia = MycelialConsensus(config)
        self.golden_dropout = GoldenDropout()
        d_ff = int(config.d_model * 4 * 2 / 3)
        self.gate = nn.Linear(config.d_model, d_ff * 2, bias=False)
        self.proj = nn.Linear(d_ff, config.d_model, bias=False)
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ffn = nn.Parameter(torch.ones(1))
        self._hidden_state = None
        self.layer_idx = layer_idx
        self.consensus_rounds = config.consensus_rounds
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

    def forward(self, x: torch.Tensor, step: int = 0):
        assert x.shape[1] <= 8192, f"Sequence length {x.shape[1]} exceeds max"
        assert x.shape[2] == self.norm1.normalized_shape[0], f"Feature dim mismatch"

        residual = x
        for _ in range(self.consensus_rounds):
            attn_out, head_outputs = self.attn(self.norm1(x), return_heads=True)
            consensus, veto, info = self.mycelia(head_outputs, step)
            if veto:
                attn_out = attn_out * 0.85
            B, T = consensus.shape[0], consensus.shape[1]
            consensus_expanded = consensus.unsqueeze(2).expand(B, T, self.n_heads, self.d_head).reshape(B, T, -1)
            attn_out = 0.9 * attn_out + 0.1 * consensus_expanded
            x = residual + self.alpha_attn * attn_out
            x = self.golden_dropout(x)
            residual = x
        g, h = self.gate(self.norm2(x)).chunk(2, dim=-1)
        ffn_out = self.proj(F.silu(g) * h)
        x = x + self.alpha_ffn * ffn_out
        x = self.golden_dropout(x)
        self._hidden_state = x.detach()
        return x, info

class MycelialCompressor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.window = config.compress_window
        self.ratio = config.compress_ratio
        self.latent_dim = config.d_model
        self.encoder_blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(2)])
        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.latent_pos = nn.Parameter(torch.randn(1, 512, config.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, D = x.shape
        assert W == self.window, f"Expected window {self.window}, got {W}"
        h = x
        for block in self.encoder_blocks:
            h, _ = block(h)
        h = h.view(B, W // self.ratio, self.ratio, D)
        latent = h.mean(dim=2)
        latent = self.latent_proj(latent)
        seq_len = latent.shape[1]
        latent = latent + self.latent_pos[:, :seq_len, :]
        return latent

class DubitoMonitor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config

    def forward(self, hidden_states: torch.Tensor, depth: int) -> float:
        if hidden_states is None or hidden_states.shape[0] < 5:
            return 0.0
        eps = 1e-8
        h_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + eps)
        v = h_norm[1:] - h_norm[:-1]
        v_unit = v / (v.norm(dim=-1, keepdim=True) + eps)
        persistence = (v_unit[1:] * v_unit[:-1]).sum(dim=-1)
        paradox_ratio = 1 - abs(persistence.mean().item())
        dubito = paradox_ratio * (1 + math.log(depth + 1))
        return max(0.0, min(15.0, dubito))

class FibonacciGuardrails(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config

    def should_continue(self, depth: int, dubito: float):
        if depth <= 5:
            ring = 0
        elif depth <= 8:
            ring = 1
        elif depth <= 13:
            ring = 2
        else:
            ring = 3
        if dubito > self.config.dubito_threshold and ring >= 2:
            return False, f"Stop: Dubito={dubito:.2f}"
        if depth > [5, 8, 13, 21][ring]:
            return False, f"Stop: Depth {depth} exceeds ring {ring} limit"
        return True, "Continue"

class MyceliaLM(nn.Module):
    """Core Mycelia Language Model Architecture."""
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        # CRITICAL: vocab_size MUST match tokenizer's actual vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.compressor = MycelialCompressor(config)
        self.blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.guardrails = FibonacciGuardrails(config)
        self.dubito_monitor = DubitoMonitor(config)
        self.depth = 0
        self.consensus_stats = []
        self.dubito_history = []
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, use_compression: bool = False, log_during_train: bool = False):
        B, T = input_ids.shape
        
        # CRITICAL FIX: Clamp token IDs to valid range to prevent embedding OOB
        # This handles any tokenizer that produces IDs outside [0, vocab_size)
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        x = self.embedding(input_ids)
        
        # CRITICAL FIX: PE must use actual_T (post-compression length), not original T
        # If compression is active, x will be shorter after compressor
        if use_compression and T > self.config.compress_window:
            prefix_len = self.config.compress_window
            prefix = x[:, :prefix_len, :]
            suffix = x[:, prefix_len:, :]
            latent = self.compressor(prefix)
            x = torch.cat([latent, suffix], dim=1)
            actual_T = x.shape[1]  # <-- THIS is the real sequence length now
        else:
            actual_T = T
        
        # FIXED: Use actual_T for positional embeddings, not original T
        x = x + get_sinusoidal_pe(actual_T, self.config.d_model, x.device)

        for i, block in enumerate(self.blocks):
            x, info = block(x, step=self.depth)
            if log_during_train and 'coherence' in info:
                self.consensus_stats.append(info['coherence'])

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if log_during_train:
            hidden = self.get_hidden_states()
            if hidden is not None:
                dubito = self.dubito_monitor(hidden, self.depth)
                self.dubito_history.append(dubito)

        return logits

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        if self.blocks and hasattr(self.blocks[-1], '_hidden_state'):
            return self.blocks[-1]._hidden_state
        return None

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 30, temperature: float = 0.7):
        """Simple generation method. Requires tokenizer."""
        self.eval()
        self.depth = 0
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        for step in range(max_new_tokens):
            self.depth = step
            logits = self(generated, use_compression=False, log_during_train=False)
            hidden = self.get_hidden_states()
            dubito = self.dubito_monitor(hidden, self.depth) if hidden is not None else 0
            should_continue, _ = self.guardrails.should_continue(self.depth, dubito)
            if not should_continue:
                break
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage:
if __name__ == "__main__":
    config = MyceliaConfig()
    model = MyceliaLM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Architecture ready. Note: For training with compression, adapt dataset chunking to handle windowed compression.")
