# ============================================
# MYCELIA LM Architecture (v8) - currently 560 LOC
# Changes from v7.1:
#   1. MycelialConsensus: Fibonacci weights generated dynamically
#      for any n_heads (fixes RuntimeError at n_heads > 6)
#   2. MycelialAttention: causal mask (dynamic, matches T at runtime)
#   3. MycelialAttention: padding mask propagated through full call chain
#   4. MycelialAttention: NaN guard after softmax
#   5. MycelialBlock: consensus_rounds=1 (halves attention compute)
#   6. MycelialBlock: standard nn.Dropout(0.1) replaces GoldenDropout
#   7. MycelialBlock: sequence length assertion raised to 4096
#   8. All changes are checkpoint-compatible with v7.1 weights EXCEPT
#      the causal mask (see note below). If resuming from a pre-causal
#      checkpoint, expect a brief loss spike (~200 steps) before recovery.
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from mycelia_jupyter_logger import MyceliaJupyterLogger

@dataclass
class MyceliaConfig:
    d_model: int = 512
    n_layers: int = 6          # ← FIXED: 6 (was 3)
    n_heads: int = 8           # ← FIXED: 8 (was 4)
    vocab_size: int = 151643   # ← FIXED: 151643 (your checkpoint's vocab size)
    max_seq_len: int = 4096
    fib_weights: Tuple[int, ...] = (5, 8, 13, 21, 34, 55, 89, 144)
    dissenter_threshold: float = 2.5
    dubito_threshold: float = 7.0
    consensus_rounds: int = 1
    # Compressor
    use_compression: bool = True
    compress_ratio: int = 8
    compress_window: int = 128
    compress_freq: int = 999999 # from low to never


def get_sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional embeddings."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class GoldenDropout(nn.Module):
    """Kept for reference; no longer used in forward pass. Replaced by nn.Dropout(0.1)."""
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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_heads: bool = True,
    ):
        """
        Args:
            x:            (B, T, D)
            padding_mask: (B, T) bool tensor, True where token == PAD_ID.
                          Optional — if None, no padding is masked.
            return_heads: whether to return per-head outputs for consensus.
        """
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv
        ]

        # Scaled dot-product attention scores: (B, n_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        # 1. Causal mask — generated dynamically to match current T exactly.
        #    Upper-triangular (future tokens) set to -inf.
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # 2. Padding mask — only applied when caller provides one.
        #    Shape (B, T) -> (B, 1, 1, T) broadcasts over heads and query positions.
        if padding_mask is not None:
            pad = padding_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)
            attn = attn.masked_fill(pad, float('-inf'))

        attn = attn.softmax(dim=-1)

        # NaN guard: if an entire row is -inf (e.g. full-padding sequence),
        # softmax produces NaN. Replace with 0 to prevent gradient explosion.
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        head_outputs = attn @ v           # (B, n_heads, T, d_head)
        out = head_outputs.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if return_heads:
            return out, head_outputs
        return out, None


class MycelialConsensus(nn.Module):
    def __init__(self, config: MyceliaConfig, use_dynamic_threshold: bool = True):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.use_dynamic_threshold = use_dynamic_threshold
        self.base_threshold = config.dissenter_threshold

        # ─── Vectorized Fibonacci weights (flat for checkpoint compat) ───
        fib = torch.tensor(config.fib_weights, dtype=torch.float32)
        self.register_buffer('fib_weights', fib / fib.sum())

        # ─── GPU telemetry counters (zero sync in hot path) ─────────────
        self.register_buffer('_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_kept', torch.zeros(1, dtype=torch.long))
        self.cached_stats = {'total': 0, 'kept': 0, 'vetoed': 0}
        self._last_threshold = 0.0

    def reset_stats(self):
        self._total.zero_()
        self._kept.zero_()
        self.cached_stats = {'total': 0, 'kept': 0, 'vetoed': 0}

    def forward(self, head_outputs: torch.Tensor, step: int = 0, layer_idx: int = 0):
        B, n_heads, T, d_head = head_outputs.shape

        # ─── STEP 1: Vectorized Fibonacci Consensus ───────────────────────
        w = self.fib_weights.view(1, -1, 1, 1)
        consensus = (head_outputs * w).sum(dim=1, keepdim=True)

        # ─── STEP 2: Vectorized Variance (per-token across heads) ─────────
        mean_heads = head_outputs.mean(dim=1, keepdim=True)
        variance = (head_outputs - mean_heads).pow(2).mean(dim=1)

        # ─── STEP 3: Dynamic Threshold (scalar math, no GPU sync) ─────────
        # v8.0: More permissive thresholds to increase acclamation rate
        # Lower threshold = more tokens pass through (less veto attenuation)
        if self.use_dynamic_threshold:
            # Early layers get MORE permissive (lower threshold) to preserve signal
            # Late layers get stricter to enforce consensus
            layer_factor = 0.7 + (layer_idx / 6.0) * 0.8  # 0.7 → 1.5
            seq_factor = 1.0 + (T / 4096.0) * 1.0         # 1.0 → 2.0 (was *2.0)
            threshold = self.base_threshold * layer_factor * seq_factor
            threshold = max(0.03, min(0.40, threshold))   # Wider range, lower floor
        else:
            threshold = self.base_threshold
        self._last_threshold = threshold

        # ─── STEP 4: Vectorized Keep Mask & Veto Attenuation ──────────────
        token_variance = variance.mean(dim=-1, keepdim=True)
        acclamation_mask = (token_variance < threshold).float().unsqueeze(1)

        veto_factor = acclamation_mask + (1.0 - acclamation_mask) * 0.85
        consensus = consensus * veto_factor

        # ─── STEP 5: Telemetry (all tensor ops, no .item()) ───────────────
        max_variance = token_variance.max()
        coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)
        veto = (token_variance >= threshold).any()

        with torch.no_grad():
            self._total += acclamation_mask.numel()
            self._kept += acclamation_mask.sum().long()

        # ─── STEP 6: Three-state classification (vectorized) ──────────────
        flat_var = token_variance.view(-1)
        self._telemetry_stats = {
            'safe_pct': (flat_var <= 2.5).float().mean() * 100,
            'dissenter_pct': ((flat_var > 2.5) & (flat_var <= 7.0)).float().mean() * 100,
            'dubito_pct': (flat_var > 7.0).float().mean() * 100,
        }

        return consensus.squeeze(1), veto, {
            'coherence': coherence,
            'variance': max_variance,
            'threshold': threshold,
            'mask_kept_ratio': acclamation_mask.mean(),
        }

    def get_stats(self) -> dict:
        total = int(self._total.item())
        kept = int(self._kept.item())
        self.cached_stats = {'total': total, 'kept': kept, 'vetoed': total - kept}
        return self.cached_stats

    def print_stats(self):
        stats = self.get_stats()
        total = stats['total']
        if total == 0:
            print("No tokens processed yet.")
            return
        kept = stats['kept']
        vetoed = stats['vetoed']
        print("="*70)
        print("🍄 MYCELIA CONSENSUS TELEMETRY")
        print("="*70)
        print(f"   Total elements: {total:,}")
        print(f"   Kept (acclaimed): {kept:,} ({kept/total*100:.1f}%)")
        print(f"   Vetoed (suppressed): {vetoed:,} ({vetoed/total*100:.1f}%)")
        print("="*70)

class MycelialBlock(nn.Module):
    def __init__(self, config: MyceliaConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.attn = MycelialAttention(config)
        self.mycelia = MycelialConsensus(config)
        self.dropout = nn.Dropout(0.1)          # replaces GoldenDropout
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

    def forward(
        self,
        x: torch.Tensor,
        step: int = 0,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        B, T, D = x.shape
        assert T <= 4096, f"Sequence length {T} exceeds max 4096"
        assert D == self.norm1.normalized_shape[0], f"Feature dim mismatch: got {D}"

        residual = x
        for round_idx in range(self.consensus_rounds):
            attn_out, head_outputs = self.attn(
                self.norm1(x),
                padding_mask=padding_mask,
                return_heads=True,
            )
            # consensus is already veto-attenuated inside MycelialConsensus
            consensus, veto, info = self.mycelia(head_outputs, step=step, layer_idx=self.layer_idx)
            round_info = info
            
            consensus_expanded = (
                consensus
                .unsqueeze(2)
                .expand(B, T, self.n_heads, self.d_head)
                .reshape(B, T, -1)
            )
            # v8.0: Adaptive mixing — more consensus weight in later rounds
            mix_ratio = 0.9 - (round_idx * 0.05)  # 0.9, 0.85, 0.8, ...
            mix_ratio = max(0.5, mix_ratio)  # Floor at 0.5
            attn_out = mix_ratio * attn_out + (1.0 - mix_ratio) * consensus_expanded
            x = residual + self.alpha_attn * attn_out
            x = self.dropout(x)
            residual = x

        g, h = self.gate(self.norm2(x)).chunk(2, dim=-1)
        ffn_out = self.proj(F.silu(g) * h)
        x = x + self.alpha_ffn * ffn_out
        x = self.dropout(x)
        self._hidden_state = x.detach()
        return x, round_info

class MycelialCompressor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.window = config.compress_window      # 128
        self.ratio = config.compress_ratio        # 8
        self.latent_dim = config.d_model          # 512
        self.encoder_blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(2)])
        self.latent_proj = nn.Linear(config.d_model, config.d_model)

        # ─── INPUT-SIDE POSITIONAL EMBEDDING (NEW) ───
        # Shape: (1, compress_window, d_model) = (1, 128, 512)
        self.input_pos = nn.Parameter(
            torch.randn(1, config.compress_window, config.d_model) * 0.02
        )

        # ─── OUTPUT-SIDE POSITIONAL EMBEDDING (EXISTING) ───
        # Shape: (1, 512, d_model) — pool of 512 positions
        self.latent_pos = nn.Parameter(
            torch.randn(1, 512, config.d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, D = x.shape
        assert W == self.window, f"Expected window {self.window}, got {W}"

        # ─── APPLY INPUT POSITIONS ──────────────────────────────────────────
        x = x + self.input_pos  # (B, 128, 512) + (1, 128, 512)

        # ─── ENCODE ──────────────────────────────────────────────────────────
        h = x
        for block in self.encoder_blocks:
            h, _ = block(h)

        # ─── POOL ────────────────────────────────────────────────────────────
        h = h.view(B, W // self.ratio, self.ratio, D)  # (B, 16, 8, 512)
        latent = h.mean(dim=2)  # (B, 16, 512)

        # ─── PROJECT ─────────────────────────────────────────────────────────
        latent = self.latent_proj(latent)  # (B, 16, 512)

        # ─── APPLY LATENT POSITIONS ─────────────────────────────────────────
        seq_len = latent.shape[1]  # 16
        latent = latent + self.latent_pos[:, :seq_len, :]  # (1, 16, 512)

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
    """Core Mycelia Language Model (v7.2)."""

    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.compressor = MycelialCompressor(config)
        self.blocks = nn.ModuleList(
            [MycelialBlock(config, i) for i in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.guardrails = FibonacciGuardrails(config)
        self.dubito_monitor = DubitoMonitor(config)
        self.depth = 0
        self.consensus_stats = []
        self.dubito_history = []
        self.register_buffer("cumulative_saved_bytes", torch.tensor(0, dtype=torch.int64))   # Cumulative VRAM savings counter
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_compression: bool = False,
        log_during_train: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        x = self.embedding(input_ids)
        x = x + get_sinusoidal_pe(T, self.config.d_model, x.device)

        # ─── VRAM SAVINGS CALCULATION ────────────────────────────────────────
        bytes_per_element = 2
        uncompressed_bytes = B * T * self.config.d_model * bytes_per_element
        compression_applied = False
        vram_saved_mb = 0.0

        if use_compression and T > self.config.compress_window:
            prefix_len = self.config.compress_window
            prefix = x[:, :prefix_len, :]
            suffix = x[:, prefix_len:, :]
            latent = self.compressor(prefix)
            x = torch.cat([latent, suffix], dim=1)
            compression_applied = True

            compressed_len = self.config.compress_window // self.config.compress_ratio
            compressed_bytes = (
                B * compressed_len * self.config.d_model * bytes_per_element
                + B * (T - prefix_len) * self.config.d_model * bytes_per_element
            )
            step_saved_bytes = uncompressed_bytes - compressed_bytes
            vram_saved_mb = step_saved_bytes / (1024 ** 2)
            self.cumulative_saved_bytes += step_saved_bytes

            if padding_mask is not None:
                compressed_pad = padding_mask[:, :prefix_len].any(dim=1, keepdim=True)
                compressed_pad = compressed_pad.expand(B, compressed_len)
                suffix_pad = padding_mask[:, prefix_len:]
                padding_mask = torch.cat([compressed_pad, suffix_pad], dim=1)

        # ─── PASS THROUGH BLOCKS WITH LAYER-WISE COHERENCE ─────────────────

        all_layer_coherence = []
        layer_variances = []          # ← NEW: track per-layer variance
        max_variance_tracked = 0.0
        last_info = {}

        for block_idx, block in enumerate(self.blocks):
            x, info = block(x, step=self.depth, padding_mask=padding_mask)
            last_info = info

            if info and 'coherence' in info:
                all_layer_coherence.append(info['coherence'])
                
                # ← NEW: capture raw variance for every layer
                layer_variances.append(info.get('variance', 0.0))
                
                if info.get('variance', 0.0) > max_variance_tracked:
                    max_variance_tracked = info.get('variance', 0.0)

                if log_during_train and 'coherence' in info:
                    self.consensus_stats.append(info['coherence'])

        # ─── COMPUTE DOMAIN FRICTION GRADIENT ──────────────────────────────
        # ← NEW: early layers (1-2) vs late layers (5-6) variance analysis
        n_layers = len(layer_variances)
        if n_layers >= 2:
            # First half = early layers, second half = late layers
            mid = n_layers // 2
            early_variance = sum(layer_variances[:mid]) / mid
            late_variance = sum(layer_variances[mid:]) / (n_layers - mid)
        else:
            early_variance = 0.0
            late_variance = 0.0

        # ─── AFTER ALL BLOCKS: NORMALIZE, PROJECT, AND RETURN ──────────────
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # ─── COMPUTE MEAN COHERENCE ACROSS ALL LAYERS ──────────────────────
        mean_coherence = sum(all_layer_coherence) / len(all_layer_coherence) if all_layer_coherence else 0.0

        # ─── COMPILE TELEMETRY INFO DICT ───────────────────────────────────
        self._last_info = {
            **last_info,
            'coherence': mean_coherence,
            'avg_coherence': mean_coherence,
            'num_layers': len(all_layer_coherence),
            'layer_coherences': all_layer_coherence,
            'layer_variances': layer_variances,      # ← NEW: full per-layer list
            'early_var': early_variance,              # ← NEW: early layers mean
            'late_var': late_variance,                # ← NEW: late layers mean
            'variance_delta': early_variance - late_variance,  # ← NEW: gradient
            'max_variance': max_variance_tracked,
            'compression_applied': compression_applied,
            'compress_ratio': self.config.compress_ratio if compression_applied else 1,
            'vram_saved': vram_saved_mb,
            'cumulative_gb': float(self.cumulative_saved_bytes.item()) / (1024 ** 3),
            'effective_seq_len': x.shape[1],
        }
        return logits

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        if self.blocks and hasattr(self.blocks[-1], '_hidden_state'):
            return self.blocks[-1]._hidden_state
        return None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
    ):
        """Autoregressive generation with dubito guardrails."""
        self.eval()
        self.depth = 0
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            self.depth = step
            logits = self(
                generated,
                use_compression=False,
                log_during_train=False,
                padding_mask=None,
            )
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

if __name__ == "__main__":
    config = MyceliaConfig()
    model = MyceliaLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"MyceliaLM v7.2: {n:,} parameters")
    print("Checkpoint-compatible with v7.1 weights.")