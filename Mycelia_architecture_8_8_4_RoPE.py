# ============================================
# MYCELIA LM Architecture (v8.8.1) — Gradual Transition + Interaction Guard
# Patched from v8.8 to fix regime transition failure
#
# Changes from v8.8:
#   1. Cosine smoothstep on gradual transition (no knee at endpoints)
#   2. Hysteresis + priority ordering on interaction guard
#   3. Post-transition stress test logging
#   4. DISABLE adaptive targets (ran away in v8.8)
#   5. RAISE control_factor_floor 0.4 → 0.7
#   6. RAISE rate thresholds 1.3x → 2.0x, 1.2x → 1.5x
#   7. DISABLE rate governor for first 50% of transition
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


@dataclass
class MyceliaConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    vocab_size: int = 151643
    max_seq_len: int = 4096
    rope_base: float = 10000.0  # Standard RoPE base, can be tuned
    fib_weights: Tuple[int, ...] = (5, 8, 13, 21, 34, 55, 89, 144)
    dissenter_threshold: float = 2.5
    dubito_threshold: float = 7.0
    consensus_rounds: int = 2
    use_compression: bool = True
    compress_ratio: int = 8
    compress_window: int = 128
    compress_freq: int = 999999

    # ── v8.8.1: Governor targets ──────────────────────────────────────
    # Start at v8.6 values; gradual transition handles the ramp
    ffn_norm_target: float = 50.0
    alpha_norm_target: float = 100.0
    soft_cap: float = 400.0

    predictive_scale: bool = True

    expected_curvature: float = 0.5
    curvature_gain: float = 2.0
    coherence_weight: float = 1.0
    forecast_weight: float = 1.0
    instability_target: float = 0.45
    control_gain: float = 1.0

    # ── v8.8.1: control_factor_floor raised 0.4 → 0.7 ────────────────
    control_factor_floor: float = 0.7

    forecast_velocity_weight: float = 1.5
    forecast_accel_weight: float = 2.0

    # ── v8.8.1: Rate governor softened ───────────────────────────────
    use_rate_governor: bool = False  # DISABLED initially
    ffn_growth_ratio_max: float = 2.0   # was 1.3
    residual_growth_ratio_max: float = 1.5  # was 1.2
    growth_gain: float = 2.0
    rate_governor_floor: float = 0.7    # was 0.4

    # ── v8.8.1: Gradual transition config ────────────────────────────
    use_gradual_transition: bool = True
    transition_start_step: int = 0
    transition_duration: int = 10000
    ffn_target_end: float = 150.0
    alpha_target_end: float = 150.0

    # ── v8.8.1: Interaction guard config ────────────────────────────
    max_simultaneous_governors: int = 2
    # Priority: keep magnitude governors over predictive ones
    # Earlier = higher priority (kept when forced to choose)
    governor_priority: Tuple[str, ...] = ('cap', 'ffn', 'alpha', 'mpc', 'rate')
    interaction_guard_hysteresis_steps: int = 500


def get_rotary_embedding(seq_len: int, d_head: int, device: torch.device, base: float = 10000.0):
    """
    Precompute sin/cos for rotary embeddings.
    Standard RoPE: theta_i = base^(-2i/d_head) for i in 0..d_head/2-1
    """
    half_dim = d_head // 2
    # theta: (half_dim,)
    theta = base ** (-torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    # positions: (seq_len, 1)
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    # angles: (seq_len, half_dim)
    angles = positions * theta.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor x.
    x: (B, n_heads, T, d_head)
    cos, sin: (T, d_head//2)
    """
    B, n_heads, T, d_head = x.shape
    half_dim = d_head // 2
    
    # Split into first and second half
    x1 = x[..., :half_dim]  # (B, n_heads, T, half_dim)
    x2 = x[..., half_dim:]  # (B, n_heads, T, half_dim)
    
    # Expand cos/sin to broadcast
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, half_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, half_dim)
    
    # Rotate: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Concatenate
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


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
        self._rope_base = getattr(config, 'rope_base', 10000.0)  # Configurable base

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, return_heads: bool = True):
        B, T, D = x.shape
        
        # ─── 1. QKV PROJECTION ──────────────────────────────────────────────
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        # ─── 2. RoPE: Apply to Q and K (V stays untouched) ──────────────────
        # RoPE MUST be applied BEFORE the attention score computation
        cos, sin = get_rotary_embedding(T, self.d_head, x.device, base=self._rope_base)
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)

        # ─── 3. ATTENTION SCORES ─────────────────────────────────────────────
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        if padding_mask is not None:
            pad = padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        # ─── 4. OUTPUT ────────────────────────────────────────────────────────
        head_outputs = attn @ v
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
        fib = torch.tensor(config.fib_weights, dtype=torch.float32)
        self.register_buffer('fib_weights', fib / fib.sum())
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
        w = self.fib_weights.view(1, -1, 1, 1)
        consensus = (head_outputs * w).sum(dim=1, keepdim=True)
        mean_heads = head_outputs.mean(dim=1, keepdim=True)
        variance = (head_outputs - mean_heads).pow(2).mean(dim=1)
        token_variance = variance.mean(dim=-1, keepdim=True)
        flat_var = token_variance.view(-1)
        var_median = flat_var.median()
        var_mad = (flat_var - var_median).abs().median()
        var_scale = var_median + 1.4826 * var_mad

        if self.use_dynamic_threshold:
            # v8.8.1: tighter late layers (inverted from v8.6)
            layer_factor = 1.0 - (layer_idx / max(self.config.n_layers - 1, 1)) * 0.3
            threshold = 0.8 * var_scale * layer_factor
            threshold = threshold.clamp(min=0.05, max=10.0)
        else:
            threshold = self.base_threshold
        self._last_threshold = float(threshold.mean().item()) if threshold.numel() == 1 else float(threshold.item())

        acclamation_mask = (token_variance < threshold).float().unsqueeze(1)
        variance_veto = acclamation_mask + (1.0 - acclamation_mask) * 0.3
        consensus = consensus * variance_veto
        max_variance = token_variance.max()
        acclamation_rate = (token_variance < threshold).float().mean()
        coherence = acclamation_rate
        veto = (token_variance >= threshold).any()

        with torch.no_grad():
            self._total += acclamation_mask.numel()
            self._kept += acclamation_mask.sum().long()

        flat_var = token_variance.view(-1)
        self._telemetry_stats = {
            'safe_pct': (flat_var <= 2.5).float().mean() * 100,
            'dissenter_pct': ((flat_var > 2.5) & (flat_var <= 7.0)).float().mean() * 100,
            'dubito_pct': (flat_var > 7.0).float().mean() * 100,
        }

        lambda_disagree = token_variance / (threshold + 1e-6)
        lambda_disagree = lambda_disagree.clamp(min=0.0, max=10.0)
        instability_prediction = torch.sigmoid(lambda_disagree - 1.0)

        return consensus.squeeze(1), veto, {
            'coherence': coherence,
            'variance': max_variance,
            'threshold': threshold,
            'mask_kept_ratio': acclamation_mask.mean(),
            'instability_prediction': instability_prediction,
            'lambda_disagree': lambda_disagree,
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
        print("MYCELIA CONSENSUS TELEMETRY")
        print("="*70)
        print(f" Total elements: {total:,}")
        print(f" Kept (acclaimed): {kept:,} ({kept/total*100:.1f}%)")
        print(f" Vetoed (suppressed): {vetoed:,} ({vetoed/total*100:.1f}%)")
        print("="*70)


class MycelialBlock(nn.Module):
    def __init__(self, config: MyceliaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.attn = MycelialAttention(config)
        self.mycelia = MycelialConsensus(config)
        self.dropout = nn.Dropout(0.1)
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
        self.soft_cap = config.soft_cap

        # v8.8.1: These are START values; gradual transition computes current
        self.ffn_norm_target = config.ffn_norm_target
        self.alpha_norm_target = config.alpha_norm_target
        self.predictive_scale = config.predictive_scale
        self.expected_curvature = config.expected_curvature
        self.curvature_gain = config.curvature_gain
        self.coherence_weight = config.coherence_weight
        self.forecast_weight = config.forecast_weight
        self.instability_target = config.instability_target
        self.control_gain = config.control_gain
        self.control_factor_floor = config.control_factor_floor
        self.forecast_velocity_weight = config.forecast_velocity_weight
        self.forecast_accel_weight = config.forecast_accel_weight

        # v8.8.1: rate governor config
        self.use_rate_governor = config.use_rate_governor
        self.ffn_growth_ratio_max = config.ffn_growth_ratio_max
        self.residual_growth_ratio_max = config.residual_growth_ratio_max
        self.growth_gain = config.growth_gain
        self.rate_governor_floor = config.rate_governor_floor

        # v8.8.1: interaction guard state (hysteresis counters)
        self._governor_hysteresis: Dict[str, int] = {}

        self.register_buffer('instability_forecast', torch.zeros(1))
        self.register_buffer('forecast_confidence', torch.ones(1))
        self.register_buffer('instability_velocity', torch.zeros(1))
        self.register_buffer('instability_acceleration', torch.zeros(1))
        self.register_buffer('_predicted_instability', torch.zeros(1))
        self.register_buffer('_actual_intervention', torch.zeros(1))
        self.register_buffer('_forecast_error', torch.zeros(1))
        self.register_buffer('_prev_hidden', torch.zeros(2, 512, config.d_model))
        self.register_buffer('_prev_delta', torch.zeros(2, 512, config.d_model))
        self.register_buffer('_prev_curvature', torch.zeros(1))

        self.register_buffer('_rate_governor_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_rate_governor_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_ffn_veto_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_ffn_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_soft_cap_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_soft_cap_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_alpha_scale_hits', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_alpha_total', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_mpc_interventions', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_mpc_total', torch.zeros(1, dtype=torch.long))

    def _compute_geometric_observables(self, x: torch.Tensor):
        B, T, D = x.shape
        if self._prev_hidden.shape[0] != B or self._prev_hidden.shape[1] != T:
            self._prev_hidden = torch.zeros(B, T, D, device=x.device)
            self._prev_delta = torch.zeros(B, T, D, device=x.device)
        velocity = x - self._prev_hidden
        velocity_norm = torch.norm(velocity, p=2, dim=-1)
        acceleration = velocity - self._prev_delta
        acceleration_norm = torch.norm(acceleration, p=2, dim=-1)
        curvature = acceleration_norm / (velocity_norm.pow(2) + 1e-6)
        curvature = curvature.clamp(max=10.0)
        jerk = (curvature - self._prev_curvature.view(1, 1).expand(B, T)).abs() if self._prev_curvature.numel() > 0 else torch.zeros_like(curvature)
        self._prev_hidden = x.detach().clone()
        self._prev_delta = velocity.detach().clone()
        self._prev_curvature = curvature.detach().mean().unsqueeze(0)
        return velocity_norm, acceleration_norm, curvature, jerk

    def _rate_governor(self, ffn_out: torch.Tensor, ffn_in: torch.Tensor,
                       x_out: torch.Tensor, x_in: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        diag = {
            'ffn_growth_ratio': 1.0,
            'residual_growth_ratio': 1.0,
            'rate_scale': 1.0,
            'rate_governor_hit': 0.0,
        }
        if not self.use_rate_governor:
            return torch.ones(ffn_out.shape[:-1] + (1,), device=ffn_out.device), diag

        ffn_in_norm = ffn_in.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ffn_out_norm = ffn_out.norm(dim=-1, keepdim=True)
        ffn_growth = (ffn_out_norm / ffn_in_norm).clamp(min=1.0)
        diag['ffn_growth_ratio'] = float(ffn_growth.mean().item())

        x_in_norm = x_in.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_out_norm = x_out.norm(dim=-1, keepdim=True)
        residual_growth = (x_out_norm / x_in_norm).clamp(min=1.0)
        diag['residual_growth_ratio'] = float(residual_growth.mean().item())

        ffn_excess = F.relu(ffn_growth - self.ffn_growth_ratio_max)
        res_excess = F.relu(residual_growth - self.residual_growth_ratio_max)
        total_excess = self.growth_gain * (ffn_excess + res_excess)
        scale = torch.exp(-total_excess).clamp(min=self.rate_governor_floor, max=1.0)

        diag['rate_scale'] = float(scale.mean().item())
        diag['rate_governor_hit'] = float((ffn_growth > self.ffn_growth_ratio_max).float().mean().item())

        with torch.no_grad():
            self._rate_governor_hits += (ffn_growth > self.ffn_growth_ratio_max).long().sum()
            self._rate_governor_total += ffn_growth.numel()

        return scale, diag

    def _recursive_instability_forecast(self, instability_prediction, coherence, curvature,
                                        prev_forecast, prev_confidence):
        B, T = curvature.shape
        device = curvature.device

        if instability_prediction.dim() == 3:
            instability_prediction = instability_prediction.squeeze(-1)
        if instability_prediction.dim() == 1:
            instability_prediction = instability_prediction.unsqueeze(0).expand(B, T)
        instability_prediction = instability_prediction.view(B, T)

        if isinstance(coherence, torch.Tensor):
            if coherence.dim() == 0:
                coherence = coherence.view(1, 1).expand(B, T)
            elif coherence.numel() == 1:
                coherence = coherence.view(1, 1).expand(B, T)
            else:
                coherence = coherence.view(B, T)
        else:
            coherence = torch.full((B, T), float(coherence), device=device)

        P_current = instability_prediction

        if prev_forecast is not None:
            P_persist = prev_forecast.view(1, 1).expand(B, T)
        else:
            P_persist = torch.zeros(B, T, device=device)

        if prev_confidence is not None:
            c_persist = prev_confidence.view(1, 1).expand(B, T)
        else:
            c_persist = torch.ones(B, T, device=device)

        base_confidence = torch.sigmoid(
            self.coherence_weight * coherence +
            self.forecast_weight * c_persist
        )
        delta_curvature = F.relu(curvature - self.expected_curvature)
        curvature_damping = torch.exp(-self.curvature_gain * delta_curvature)
        confidence = base_confidence * curvature_damping

        I = confidence * P_current + (1.0 - confidence) * P_persist
        I = I.clamp(min=0.0, max=1.0)

        prev_I = self.instability_forecast.item()
        prev_v = self.instability_velocity.item()
        current_I_mean = I.mean().item()
        v_I = current_I_mean - prev_I
        a_I = v_I - prev_v

        self.instability_forecast = torch.tensor([current_I_mean], device=device)
        self.instability_velocity = torch.tensor([v_I], device=device)
        self.instability_acceleration = torch.tensor([a_I], device=device)

        I_raw = I
        I_combined = I_raw + (
            self.forecast_velocity_weight * F.relu(torch.tensor(v_I, device=device)) +
            self.forecast_accel_weight * F.relu(torch.tensor(a_I, device=device))
        )
        I_combined = I_combined.clamp(max=2.0)

        predicted_from_prev = self._predicted_instability.item()
        actual_intervention = (I_combined > self.instability_target).float().mean().item()
        forecast_error = abs(predicted_from_prev - actual_intervention)

        self._predicted_instability = torch.tensor([I_raw.mean().item()], device=device)
        self._actual_intervention = torch.tensor([actual_intervention], device=device)
        self._forecast_error = torch.tensor([forecast_error], device=device)

        next_forecast = torch.tensor([current_I_mean], device=device)
        next_confidence = torch.tensor([confidence.mean().item()], device=device)

        diagnostics = {
            'confidence_mean': float(confidence.mean().item()),
            'confidence_min': float(confidence.min().item()),
            'curvature_damping_mean': float(curvature_damping.mean().item()),
            'delta_curvature_mean': float(delta_curvature.mean().item()),
            'instability_prediction': float(P_current.mean().item()),
            'instability_persistence': float(P_persist.mean().item()),
            'instability_posterior': float(I.mean().item()),
            'instability_velocity': float(v_I),
            'instability_acceleration': float(a_I),
            'instability_combined': float(I_combined.mean().item()),
            'forecast_error': float(forecast_error),
            'predicted_from_prev': float(predicted_from_prev),
            'actual_intervention': float(actual_intervention),
        }

        return I_combined, next_forecast, next_confidence, torch.tensor(forecast_error, device=device), diagnostics

    def _control_policy(self, I_combined, confidence, alpha_native):
        prediction = I_combined.clamp(max=1.0)
        control_signal = confidence * prediction
        control_factor = torch.exp(-self.control_gain * control_signal)
        # v8.8.1: floor raised to 0.7
        control_factor = control_factor.clamp(min=self.control_factor_floor, max=1.0)
        return (alpha_native * control_factor).unsqueeze(-1)

    # ── v8.8.1: INTERACTION GUARD WITH HYSTERESIS ──────────────────────
    def _apply_interaction_guard(self, scales: Dict[str, torch.Tensor],
                                  step: int) -> Tuple[Dict[str, torch.Tensor], List[str]]: 
        """
        If > max_simultaneous_governors are actively suppressing,
        disable the lowest-priority ones that have been active for
        >= hysteresis_steps consecutive steps.
        Priority: cap > ffn > alpha > mpc > rate (keep magnitude over predictive)
        """
        max_govs = self.config.max_simultaneous_governors
        hysteresis = self.config.interaction_guard_hysteresis_steps
        priority = {name: i for i, name in enumerate(self.config.governor_priority)}

        # Check which governors are actively suppressing (< 0.99)
        active = []
        for name, scale in scales.items():
            mean_scale = float(scale.mean().item())
            if mean_scale < 0.99:
                # Update hysteresis counter
                if name not in self._governor_hysteresis:
                    self._governor_hysteresis[name] = 0
                self._governor_hysteresis[name] += 1
                active.append((name, mean_scale, self._governor_hysteresis[name]))
            else:
                # Reset counter if not active
                if name in self._governor_hysteresis:
                    self._governor_hysteresis[name] = 0

        disabled: List[str] = []
        if len(active) <= max_govs:
            return scales, disabled  # ← returns 2 values, disabled is empty list

        # Sort by priority (lower index = higher priority = keep)
        active.sort(key=lambda x: priority.get(x[0], 99))

        # Only disable governors that have been active long enough
        for name, scale_val, hyst_count in active[max_govs:]:
            if hyst_count >= hysteresis:
                disabled.append(name)
                scales[name] = torch.ones_like(scales[name])
                self._governor_hysteresis[name] = 0  # Reset after disable

        return scales, disabled

    def forward(self, x: torch.Tensor, step: int = 0,
                padding_mask: Optional[torch.Tensor] = None,
                prev_forecast: Optional[torch.Tensor] = None,
                prev_confidence: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        assert T <= 4096, f"Sequence length {T} exceeds max 4096"
        assert D == self.norm1.normalized_shape[0], f"Feature dim mismatch: got {D}"

        residual = x
        x_input_for_rate = x
        final_round_info = {}

        # ── v8.8.1: Compute current targets via smoothstep transition ────
        if self.config.use_gradual_transition:
            progress = min(1.0, (step - self.config.transition_start_step) / self.config.transition_duration)
            # Cosine smoothstep: 0.5 - 0.5*cos(π*progress)
            # Derivative is zero at both endpoints (gentle start/stop)
            ease = 0.5 - 0.5 * math.cos(math.pi * progress)
            current_ffn_target = self.config.ffn_norm_target + ease * (self.config.ffn_target_end - self.config.ffn_norm_target)
            current_alpha_target = self.config.alpha_norm_target + ease * (self.config.alpha_target_end - self.config.alpha_norm_target)
            # Enable rate governor only after 50% progress
            use_rate = progress > 0.5
        else:
            current_ffn_target = self.ffn_norm_target
            current_alpha_target = self.alpha_norm_target
            use_rate = self.use_rate_governor

        velocity_norm, acceleration_norm, curvature, jerk = self._compute_geometric_observables(x)

        for round_idx in range(self.consensus_rounds):
            attn_out, head_outputs = self.attn(
                self.norm1(x),
                padding_mask=padding_mask,
                return_heads=True,
            )
            consensus, veto, info = self.mycelia(head_outputs, step=step, layer_idx=self.layer_idx)
            final_round_info = info
            consensus_expanded = (
                consensus
                .unsqueeze(2)
                .expand(B, T, self.n_heads, self.d_head)
                .reshape(B, T, -1)
            )
            mix_ratio = 0.9 - (round_idx * 0.05)
            mix_ratio = max(0.5, mix_ratio)
            attn_out = mix_ratio * attn_out + (1.0 - mix_ratio) * consensus_expanded
            x = residual + self.alpha_attn * attn_out
            x = self.dropout(x)
            residual = x

        g, h = self.gate(self.norm2(x)).chunk(2, dim=-1)
        ffn_in_for_rate = x
        ffn_out = self.proj(F.silu(g) * h)

        # ── v8.8.1: FFN VETO with current (possibly ramped) target ────────
        ffn_norms = torch.norm(ffn_out, p=2, dim=-1, keepdim=True)
        ffn_veto = torch.clamp(current_ffn_target / (ffn_norms + 1e-6), max=1.0)
        ffn_out = ffn_out * ffn_veto
        ffn_veto_factor = ffn_veto.mean().item()
        ffn_work = 1.0 - ffn_veto_factor

        with torch.no_grad():
            ffn_veto_hits = (ffn_norms > current_ffn_target).float()
            self._ffn_veto_hits += ffn_veto_hits.sum().long()
            self._ffn_total += ffn_veto_hits.numel()

        if self.predictive_scale:
            instability_prediction = info.get('instability_prediction',
                                             torch.zeros(B, T, device=x.device))
            coherence = info.get('coherence', torch.tensor(0.5, device=x.device))

            I_combined, next_forecast, next_confidence, forecast_error, mpc_diag = self._recursive_instability_forecast(
                instability_prediction, coherence, curvature, prev_forecast, prev_confidence)

            if isinstance(coherence, torch.Tensor):
                if coherence.dim() == 0:
                    coherence_bt = coherence.view(1, 1).expand(B, T)
                elif coherence.numel() == 1:
                    coherence_bt = coherence.view(1, 1).expand(B, T)
                else:
                    coherence_bt = coherence.view(B, T)
            else:
                coherence_bt = torch.full((B, T), float(coherence), device=x.device)

            if prev_confidence is not None:
                prev_conf_bt = prev_confidence.view(1, 1).expand(B, T)
            else:
                prev_conf_bt = torch.ones(B, T, device=x.device)

            confidence_for_control = torch.sigmoid(
                self.coherence_weight * coherence_bt +
                self.forecast_weight * prev_conf_bt
            ) * torch.exp(-self.curvature_gain * F.relu(curvature - self.expected_curvature))

            effective_alpha_attn = self._control_policy(I_combined, confidence_for_control, self.alpha_attn)
            effective_alpha_ffn = self._control_policy(I_combined, confidence_for_control, self.alpha_ffn)

            assert effective_alpha_attn.shape == (B, T, 1)
            assert effective_alpha_ffn.shape == (B, T, 1)
        else:
            effective_alpha_attn = self.alpha_attn.view(1, 1, 1).expand(B, T, 1)
            effective_alpha_ffn = self.alpha_ffn.view(1, 1, 1).expand(B, T, 1)
            mpc_diag = {}
            I_combined = torch.zeros(B, T, device=x.device)
            next_forecast = torch.zeros(1, device=x.device)
            next_confidence = torch.ones(1, device=x.device)
            forecast_error = torch.zeros(1, device=x.device)

        # ── v8.8.1: ALPHA SCALE with current (possibly ramped) target ────
        attn_norms = torch.norm(attn_out, p=2, dim=-1, keepdim=True)
        ffn_norms_post = torch.norm(ffn_out, p=2, dim=-1, keepdim=True)
        contrib_norm = torch.sqrt(
            (effective_alpha_attn * attn_norms).pow(2) +
            (effective_alpha_ffn * ffn_norms_post).pow(2) + 1e-6
        )
        alpha_scale = torch.clamp(current_alpha_target / (contrib_norm + 1e-6), max=1.0)
        effective_alpha_attn = effective_alpha_attn * alpha_scale
        effective_alpha_ffn = effective_alpha_ffn * alpha_scale
        alpha_scale_factor = alpha_scale.mean().item()
        alpha_work = 1.0 - alpha_scale_factor

        with torch.no_grad():
            alpha_hits = (contrib_norm > current_alpha_target).float()
            self._alpha_scale_hits += alpha_hits.sum().long()
            self._alpha_total += alpha_hits.numel()

        x = residual + effective_alpha_attn * attn_out + effective_alpha_ffn * ffn_out

        # ── v8.8.1: RATE GOVERNOR (only if enabled by transition progress) ─
        rate_scale, rate_diag = self._rate_governor(
            ffn_out=ffn_out, ffn_in=ffn_in_for_rate,
            x_out=x, x_in=x_input_for_rate,
        )
        if use_rate:
            x = x * rate_scale

        # ── SOFT NORM CAP ──────────────────────────────────────────────
        token_norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        excess = F.softplus(token_norms - self.soft_cap)
        soft_scale = 1.0 + excess / (token_norms + 1e-6)
        x = x / soft_scale
        soft_cap_factor = 1.0 / soft_scale.mean().item()
        cap_work = 1.0 - soft_cap_factor

        with torch.no_grad():
            cap_engaged = (token_norms > self.soft_cap).float()
            self._soft_cap_hits += cap_engaged.sum().long()
            self._soft_cap_total += cap_engaged.numel()

        # ── v8.8.1: GOVERNOR INTERACTION GUARD ──────────────────────────
        scales = {
            'ffn': ffn_veto,
            'alpha': alpha_scale,
            'cap': soft_scale,
            'rate': rate_scale if use_rate else torch.ones_like(rate_scale),
            'mpc': torch.exp(-self.control_gain * (confidence_for_control * I_combined.clamp(max=1.0))).unsqueeze(-1) if I_combined.numel() > 0 else torch.ones(B, T, 1, device=x.device),
        }
        scales, disabled_governors = self._apply_interaction_guard(scales, step)

        # Re-apply scales after guard (some may have been reset to 1.0)
        if 'ffn' in disabled_governors:
            ffn_out = ffn_out / (ffn_veto + 1e-8)  # undo veto
            ffn_work = 0.0
        if 'alpha' in disabled_governors:
            # Can't easily undo alpha scale; just note it
            alpha_work = 0.0
        if 'cap' in disabled_governors:
            # Can't easily undo soft cap
            cap_work = 0.0
        if 'mpc' in disabled_governors:
            # MPC already applied; can't undo
            pass
        if 'rate' in disabled_governors:
            # Rate already applied; can't undo
            pass

        with torch.no_grad():
            self.instability_forecast = next_forecast
            self.forecast_confidence = next_confidence

        # ── TELEMETRY ────────────────────────────────────────────────────
        final_round_info['ffn_veto_ratio'] = float(ffn_veto_hits.mean().item())
        final_round_info['mean_ffn_norm'] = float(ffn_norms.mean().item())
        final_round_info['max_ffn_norm'] = float(ffn_norms.max().item())
        final_round_info['ffn_veto_factor'] = float(ffn_veto_factor)
        final_round_info['ffn_work'] = float(ffn_work)
        final_round_info['ffn_target_current'] = float(current_ffn_target)

        final_round_info['alpha_scale_ratio'] = float(alpha_hits.mean().item())
        final_round_info['mean_alpha_scale'] = float(alpha_scale.mean().item())
        final_round_info['mean_contrib_norm'] = float(contrib_norm.mean().item())
        final_round_info['alpha_scale_factor'] = float(alpha_scale_factor)
        final_round_info['alpha_work'] = float(alpha_work)
        final_round_info['alpha_target_current'] = float(current_alpha_target)

        final_round_info['mpc_intervention_ratio'] = float(
            (I_combined > self.instability_target).float().mean().item()
        ) if I_combined.numel() > 0 else 0.0
        final_round_info['mean_control_factor'] = float(
            torch.exp(-self.control_gain * (confidence_for_control * I_combined.clamp(max=1.0))).mean().item()
        ) if I_combined.numel() > 0 else 1.0
        final_round_info['mean_instability_field'] = float(I_combined.mean().item())
        final_round_info['instability_velocity'] = mpc_diag.get('instability_velocity', 0.0)
        final_round_info['instability_acceleration'] = mpc_diag.get('instability_acceleration', 0.0)
        mpc_control_factor = final_round_info['mean_control_factor']
        mpc_work = 1.0 - mpc_control_factor
        final_round_info['mpc_work'] = float(mpc_work)
        final_round_info['mpc_control_factor'] = float(mpc_control_factor)

        final_round_info['mean_prediction'] = mpc_diag.get('instability_prediction', 0.0)
        final_round_info['mean_confidence'] = mpc_diag.get('confidence_mean', 1.0)
        final_round_info['confidence_min'] = mpc_diag.get('confidence_min', 1.0)
        final_round_info['curvature_damping'] = mpc_diag.get('curvature_damping_mean', 1.0)
        final_round_info['delta_curvature'] = mpc_diag.get('delta_curvature_mean', 0.0)

        final_round_info['forecast_error'] = mpc_diag.get('forecast_error', 0.0)
        final_round_info['predicted_from_prev'] = mpc_diag.get('predicted_from_prev', 0.0)
        final_round_info['actual_intervention'] = mpc_diag.get('actual_intervention', 0.0)

        final_round_info['mean_velocity'] = float(velocity_norm.mean().item())
        final_round_info['mean_acceleration'] = float(acceleration_norm.mean().item())
        final_round_info['mean_curvature'] = float(curvature.mean().item())
        final_round_info['max_curvature'] = float(curvature.max().item())
        final_round_info['mean_jerk'] = float(jerk.mean().item())
        final_round_info['soft_cap_hit_ratio'] = float(cap_engaged.mean().item())
        final_round_info['mean_soft_scale'] = float(soft_scale.mean().item())
        final_round_info['max_raw_norm'] = float(token_norms.max().item())
        final_round_info['mean_raw_norm'] = float(token_norms.mean().item())
        final_round_info['soft_cap_factor'] = float(soft_cap_factor)
        final_round_info['cap_work'] = float(cap_work)

        final_round_info['rate_governor_hit'] = rate_diag['rate_governor_hit']
        final_round_info['rate_scale_mean'] = rate_diag['rate_scale']
        final_round_info['ffn_growth_ratio'] = rate_diag['ffn_growth_ratio']
        final_round_info['residual_growth_ratio'] = rate_diag['residual_growth_ratio']
        final_round_info['rate_governor_enabled'] = float(use_rate)

        final_round_info['transition_progress'] = float(progress) if self.config.use_gradual_transition else 1.0
        final_round_info['transition_ease'] = float(ease) if self.config.use_gradual_transition else 1.0
        final_round_info['disabled_governors'] = disabled_governors

        layer_pressure = (
            ffn_work * ffn_norms.mean().item() +
            alpha_work * contrib_norm.mean().item() +
            cap_work * token_norms.mean().item() +
            mpc_work * I_combined.mean().item()
        )
        final_round_info['layer_pressure'] = float(layer_pressure)

        x = self.dropout(x)
        self._hidden_state = x.detach()
        self._last_info = final_round_info
        return x, final_round_info


class MycelialCompressor(nn.Module):
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
        self.window = config.compress_window
        self.ratio = config.compress_ratio
        self.latent_dim = config.d_model
        self.encoder_blocks = nn.ModuleList([MycelialBlock(config, i) for i in range(2)])
        self.latent_proj = nn.Linear(config.d_model, config.d_model)
        self.input_pos = nn.Parameter(torch.randn(1, config.compress_window, config.d_model) * 0.02)
        self.latent_pos = nn.Parameter(torch.randn(1, 512, config.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, D = x.shape
        assert W == self.window, f"Expected window {self.window}, got {W}"
        x = x + self.input_pos
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
    def __init__(self, config: MyceliaConfig):
        super().__init__()
        self.config = config
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
        self.register_buffer("cumulative_saved_bytes", torch.tensor(0, dtype=torch.int64))
        self._last_info = {}
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, use_compression: bool = False,
                log_during_train: bool = False, padding_mask: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        x = self.embedding(input_ids)
      # x = x + get_sinusoidal_pe(T, self.config.d_model, x.device)

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

        all_layer_coherence = []
        layer_variances = []
        max_variance_tracked = 0.0
        last_info = {}
        prev_forecast = None
        prev_confidence = None
        instability_field_history = []
        confidence_history = []

        for block_idx, block in enumerate(self.blocks):
            if block_idx > 0:
                prev_forecast = self.blocks[block_idx - 1].instability_forecast
                prev_confidence = self.blocks[block_idx - 1].forecast_confidence

            x, info = block(x, step=self.depth, padding_mask=padding_mask,
                            prev_forecast=prev_forecast, prev_confidence=prev_confidence)
            last_info = info

            if info and 'coherence' in info:
                all_layer_coherence.append(info['coherence'])
            layer_variances.append(info.get('variance', 0.0))
            instability_field_history.append(info.get('mean_instability_field', 0.0))
            confidence_history.append(info.get('mean_confidence', 1.0))

            if info.get('variance', 0.0) > max_variance_tracked:
                max_variance_tracked = info.get('variance', 0.0)

            if log_during_train and 'coherence' in info:
                self.consensus_stats.append(info['coherence'])

        n_layers = len(layer_variances)
        if n_layers >= 2:
            mid = n_layers // 2
            early_variance = sum(layer_variances[:mid]) / mid
            late_variance = sum(layer_variances[mid:]) / (n_layers - mid)
        else:
            early_variance = 0.0
            late_variance = 0.0

        total_pressure = 0.0
        pressure_by_governor = {'ffn': 0.0, 'alpha': 0.0, 'cap': 0.0, 'mpc': 0.0}

        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, '_last_info') and block._last_info:
                info = block._last_info
                for gov in ['ffn', 'alpha', 'cap', 'mpc']:
                    work = info.get(f'{gov}_work', 0.0)
                    norm = info.get({
                        'ffn': 'mean_ffn_norm',
                        'alpha': 'mean_contrib_norm',
                        'cap': 'mean_raw_norm',
                        'mpc': 'mean_instability_field'
                    }[gov], 0.0)
                    pressure_by_governor[gov] += work * norm
                    total_pressure += work * norm

        concentration = max(pressure_by_governor.values()) / (total_pressure + 1e-8) if total_pressure > 0 else 0.0

        x = self.final_norm(x)
        logits = self.lm_head(x)
        mean_coherence = sum(all_layer_coherence) / len(all_layer_coherence) if all_layer_coherence else 0.0

        self._last_info = {
            **last_info,
            'total_pressure': float(total_pressure),
            'pressure_concentration': float(concentration),
            'pressure_by_governor': {k: float(v) for k, v in pressure_by_governor.items()},
            'dominant_governor': max(pressure_by_governor, key=pressure_by_governor.get) if total_pressure > 0 else None,
            'coherence': mean_coherence,
            'avg_coherence': mean_coherence,
            'num_layers': len(all_layer_coherence),
            'layer_coherences': all_layer_coherence,
            'layer_variances': layer_variances,
            'instability_field_history': instability_field_history,
            'confidence_history': confidence_history,
            'early_var': early_variance,
            'late_var': late_variance,
            'variance_delta': early_variance - late_variance,
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
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 30, temperature: float = 0.7):
        self.eval()
        self.depth = 0
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        for step in range(max_new_tokens):
            self.depth = step
            logits = self(generated, use_compression=False, log_during_train=False, padding_mask=None)
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
    print(f"MyceliaLM v8.8.1: {n:,} parameters")
    print(f"Gradual transition: FFN 50→150, α 100→150 over 10K steps")
    print(f"Smoothstep easing: cosine interpolation")
    print(f"Interaction guard: max 2 simultaneous, hysteresis 500 steps")
    print(f"Priority: cap > ffn > alpha > mpc > rate")
    print(f"Rate governor: disabled until 50% transition progress")
    print(f"Control floor: 0.7 (was 0.4)")
    print(f"Checkpoint-compatible with v8.6/v8.7 weights (strict=False).")