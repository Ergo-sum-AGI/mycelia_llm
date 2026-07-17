# Meta-Governor v4.2 — Four-Loop Phase-Locked Architecture --- 18072026
"""Meta-Governor v4.2: Multi-provider consortium with phase-locked CB policy.

FOUR-LOOP ARCHITECTURE:
  L1 Fast (MPC):        ~250 iters/block     Per-block intervention
  L2 Medium-Fast (Meta): Per-block, sticky    CB = 🔴/🟢 state persists
  L3 Medium-Slow (Rate): Kicks at 50% trans   Modulates FFN + residual growth
  L4 Slow (Auto-Tune):   Every 5,000 iters    Widens coh/fcst corridor

CB=🔴 CODIFIED POLICY (v4.2):
  Mode 1 (FAILURE):  >5 failed suggestions in 10-round window [legacy]
  Mode 2 (CONSTRUCTIVE): MPC < 15% AND phase 0.75-1.0 AND events >= 3
    → Silence IS the intervention. Don't fight the model in the sweet spot.

API FIXES: DeepSeek URL, Cerebras key, Moonshot env, Google SDK v2,
           merged call_agent, fixed provider_name scoping.

Usage:
    from meta_governor_v4_2 import integrate_meta_governor
    actions = integrate_meta_governor(model, auto_tuner, step, loss, lr,
                                      auto_tune_info={"interval": 5000, "event_count": n})
"""

import json, os, time, numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Literal, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator

# =============================================================================
# CONFIGURATION
# =============================================================================

# API keys from environment (set before running)
# GROQ_API_KEY, DEEPSEEK_API_KEY, CEREBRAS_API_KEY, GOOGLE_API_KEY,
# MOONSHOT_API_KEY (or KIMI_API_KEY), XAI_API_KEY

CIRCUIT_BREAKER_FAILURE_WINDOW = 10
CIRCUIT_BREAKER_TRIP_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_THRESHOLD = 3

# NEW v4.2: Constructive interference CB parameters
CB_CONSTRUCTIVE_MPC_THRESHOLD = 0.15
CB_CONSTRUCTIVE_PHASE_MIN = 0.75
CB_CONSTRUCTIVE_PHASE_MAX = 1.0
CB_CONSTRUCTIVE_MIN_EVENTS = 3

CONSENSUS_CONFIDENCE_THRESHOLD = 0.20
API_TIMEOUT = 10
MAX_WORKERS = 4
MAX_AGENTS_PER_ROUND = 6


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class VariableName(str, Enum):
    FFN_NORM_TARGET = "ffn_norm_target"
    ALPHA_NORM_TARGET = "alpha_norm_target"
    CONTROL_GAIN = "control_gain"
    INSTABILITY_TARGET = "instability_target"
    SOFT_CAP = "soft_cap"
    WEIGHT_FLOOR = "weight_floor"
    TEMPERATURE = "temperature"
    BURN_IN_STEPS = "burn_in_steps"
    EXPECTED_CURVATURE = "expected_curvature"

class Direction(str, Enum):
    RAISE = "raise"; LOWER = "lower"; SET = "set"
    PAUSE = "pause"; INVESTIGATE = "investigate"

class Magnitude(str, Enum):
    SMALL = "small"; MEDIUM = "medium"; LARGE = "large"; EMERGENCY = "emergency"

class Enactable(str, Enum):
    RUNTIME = "runtime"; REQUIRES_RESTART = "requires_restart"; REQUIRES_HUMAN = "requires_human"

class ExpectedOutcome(str, Enum):
    LOSS_DECREASE = "loss_decrease"; PRESSURE_REDISTRIBUTE = "pressure_redistribute"
    COHERENCE_INCREASE = "coherence_increase"; REGIME_TRANSITION = "regime_transition"
    STABILITY_IMPROVEMENT = "stability_improvement"

class MetaGovernorAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    variable: VariableName
    direction: Direction
    magnitude: Magnitude = Field(default=Magnitude.SMALL)
    value: Optional[float] = Field(default=None)
    enactable: Enactable = Field(default=Enactable.RUNTIME)
    duration_steps: int = Field(default=500, ge=10, le=5000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)
    lesson: str = Field(default="", max_length=500)

    @field_validator("value")
    @classmethod
    def _check_bounds(cls, v, info):
        if v is None: return v
        var = info.data.get("variable")
        bounds = {
            VariableName.FFN_NORM_TARGET: (10, 1000),
            VariableName.ALPHA_NORM_TARGET: (10, 1000),
            VariableName.CONTROL_GAIN: (0.01, 10.0),
            VariableName.INSTABILITY_TARGET: (0.01, 1.0),
            VariableName.SOFT_CAP: (100, 2000),
            VariableName.WEIGHT_FLOOR: (3, 8),
            VariableName.TEMPERATURE: (0.3, 2.0),
            VariableName.BURN_IN_STEPS: (100, 10000),
            VariableName.EXPECTED_CURVATURE: (0.1, 2.0),
        }
        if var in bounds:
            lo, hi = bounds[var]
            if not (lo <= v <= hi):
                raise ValueError(f"{var.value} must be in [{lo}, {hi}]")
        return v

class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent_name: str
    action: Optional[MetaGovernorAction] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ConsensusDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    variable: Optional[VariableName] = None
    direction: Optional[Direction] = None
    value: Optional[float] = None
    confidence: float = 0.0
    duration_steps: int = 500
    expected_outcome: Optional[ExpectedOutcome] = None
    rationale: str = ""
    lesson: str = ""
    participating_agents: List[str] = Field(default_factory=list)
    dissenting_agents: List[str] = Field(default_factory=list)
    consensus_type: Literal["unanimous", "majority", "plurality", "single", "none", "conflict"] = "none"


# =============================================================================
# TELEMETRY PACKET — v4.2: Phase-Locked + Scheduler-Aware
# =============================================================================

@dataclass
class TelemetryPacket:
    """Structured telemetry from Mycelia training loop."""
    step: int; loss: float; lr: float
    coherence: float = 0.0
    friction_regime: str = "UNKNOWN"
    delta: float = 0.0; early_var: float = 0.0; late_var: float = 0.0
    rope_stability: float = 0.0; positional_coherence: float = 0.0
    layer_coherences: List[float] = field(default_factory=list)
    layer_instability: List[float] = field(default_factory=list)
    mean_velocity: float = 0.0; mean_acceleration: float = 0.0
    mean_curvature: float = 0.0; mean_jerk: float = 0.0
    ffn_veto_ratio: float = 0.0; ffn_norm_mean: float = 0.0
    ffn_norm_max: float = 0.0; ffn_target: float = 0.0
    alpha_scale_ratio: float = 0.0; alpha_contrib_norm: float = 0.0
    alpha_target: float = 0.0; soft_cap_hit: float = 0.0
    soft_cap_max_raw: float = 0.0; mpc_intervention: float = 0.0
    mpc_control_factor: float = 1.0; instability_field_mean: float = 0.0
    forecast_error: float = 0.0
    pressure_total: float = 0.0; pressure_concentration: float = 0.0
    pressure_dominant: str = "none"
    pressure_by_governor: Dict[str, float] = field(default_factory=dict)
    max_curvature: float = 0.0
    rate_governor_hit: float = 0.0; rate_scale_mean: float = 1.0
    ffn_growth_ratio: float = 1.0; residual_growth_ratio: float = 1.0
    instability_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    regime_duration: int = 0; veto_rate_high_duration: int = 0
    meta_governor_last_step: int = 0; meta_governor_active_adjustments: int = 0
    architecture_suggestions_pending: int = 0; human_review_pending: int = 0
    scheduler_alive: bool = True; scheduler_last_epoch: int = 0
    scheduler_total_steps: int = 0; scheduler_resurrected_count: int = 0
    lr_valid: bool = True
    # NEW v4.2: Phase-locked telemetry
    auto_tune_phase: float = 0.0
    auto_tune_event_count: int = 0
    auto_tune_interval: int = 5000
    transition_progress: float = 0.0
    model_version: str = "v9.0"; session_id: str = "default"

    @classmethod
    def from_model(cls, model, step: int, loss: float, lr: float,
                   scheduler_info: Optional[Dict] = None,
                   auto_tune_info: Optional[Dict] = None) -> "TelemetryPacket":
        info = getattr(model, "_last_info", {}) or {}
        scheduler_info = scheduler_info or {}
        auto_tune_info = auto_tune_info or {}

        def _sanitize_list(vec):
            import torch
            if vec is None: return []
            if isinstance(vec, torch.Tensor):
                return vec.detach().cpu().flatten().tolist()
            result = []
            for v in vec:
                if isinstance(v, torch.Tensor):
                    result.append(float(v.detach().cpu().item()))
                elif isinstance(v, (int, float)):
                    result.append(float(v))
            return result

        interval = auto_tune_info.get("interval", 5000)
        event_count = auto_tune_info.get("event_count", 0)
        phase = (step % interval) / interval if interval > 0 else 0.0
        trans = info.get("transition_progress", 0.0)
        if not isinstance(trans, (int, float)): trans = 0.0

        return cls(
            step=step, loss=loss, lr=lr,
            coherence=info.get("coherence", 0.0),
            friction_regime=info.get("friction", "UNKNOWN"),
            delta=info.get("variance_delta", 0.0),
            early_var=info.get("early_var", 0.0),
            late_var=info.get("late_var", 0.0),
            rope_stability=info.get("rope_stability", 0.0),
            positional_coherence=info.get("positional_coherence", 0.0),
            layer_coherences=_sanitize_list(info.get("layer_coherences", [])),
            layer_instability=_sanitize_list(info.get("layer_instability", [])),
            ffn_veto_ratio=info.get("ffn_veto_ratio", 0.0),
            ffn_norm_mean=info.get("mean_ffn_norm", 0.0),
            ffn_norm_max=info.get("max_ffn_norm", 0.0),
            ffn_target=info.get("ffn_target_current", 150.0),
            alpha_scale_ratio=info.get("alpha_scale_ratio", 0.0),
            alpha_contrib_norm=info.get("mean_contrib_norm", 0.0),
            alpha_target=info.get("alpha_target_current", 150.0),
            soft_cap_hit=info.get("soft_cap_hit_ratio", 0.0),
            soft_cap_max_raw=info.get("max_raw_norm", 0.0),
            mpc_intervention=info.get("mpc_intervention_ratio", 0.0),
            mpc_control_factor=info.get("mean_control_factor", 1.0),
            instability_field_mean=info.get("mean_instability_field", 0.0),
            forecast_error=info.get("forecast_error", 0.0),
            pressure_total=info.get("total_pressure", 0.0),
            pressure_concentration=info.get("pressure_concentration", 0.0),
            pressure_dominant=info.get("dominant_governor", "none"),
            pressure_by_governor=info.get("pressure_by_governor", {}),
            mean_velocity=info.get("mean_velocity", 0.0),
            mean_acceleration=info.get("mean_acceleration", 0.0),
            mean_curvature=info.get("mean_curvature", 0.0),
            max_curvature=info.get("max_curvature", 0.0),
            mean_jerk=info.get("mean_jerk", 0.0),
            rate_governor_hit=info.get("rate_governor_hit", 0.0),
            rate_scale_mean=info.get("rate_scale_mean", 1.0),
            ffn_growth_ratio=info.get("ffn_growth_ratio", 1.0),
            residual_growth_ratio=info.get("residual_growth_ratio", 1.0),
            instability_history=_sanitize_list(info.get("instability_field_history", []))[:12],
            confidence_history=_sanitize_list(info.get("confidence_history", []))[:12],
            regime_duration=info.get("regime_duration", 0),
            veto_rate_high_duration=info.get("veto_rate_high_duration", 0),
            scheduler_alive=scheduler_info.get("alive", True),
            scheduler_last_epoch=scheduler_info.get("last_epoch", 0),
            scheduler_total_steps=scheduler_info.get("total_steps", 0),
            scheduler_resurrected_count=scheduler_info.get("resurrected_count", 0),
            lr_valid=scheduler_info.get("lr_valid", lr > 0),
            auto_tune_phase=phase,
            auto_tune_event_count=event_count,
            auto_tune_interval=interval,
            transition_progress=float(trans),
        )

    def to_dict(self) -> dict:
        import torch
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
            elif isinstance(v, list):
                d[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
            elif isinstance(v, dict):
                d[k] = {kk: vv.item() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                d[k] = v
        return d


# =============================================================================
# SYSTEM PROMPTS — v4.2: Phase-Locked + Four-Loop Aware
# =============================================================================

MACRO_SYSTEM_PROMPT = """You are Qwen3-32B-Macro-Governor, macro-dynamics expert for Mycelia (12-layer RoPE transformer).

FOUR-LOOP CONTEXT:
  L1 Fast (MPC): ~250 iters/block    L2 Medium-Fast (Meta): sticky CB state
  L3 Medium-Slow (Rate): kicks at 50% transition    L4 Slow (Auto-Tune): every 5k iters

PHASE-LOCKED POLICY (CRITICAL):
  auto_tune_phase in telemetry (0.0-1.0):
  - Phase 0.75-1.0 (pre-fire): CONSTRUCTIVE window. If MPC < 15%: DO NOT INTERVENE.
    Silence IS the intervention. The model exploits a pre-loaded corridor.
  - Phase 0.0-0.25 (post-fire): DESTRUCTIVE window. Transient ringing. Be conservative.
  - Sweet spot migrates: early runs → 0.75, late runs (10+ events) → 0.90.

SCHEDULER HEALTH:
  If scheduler_alive=FALSE or lr_valid=FALSE: EMERGENCY.
  → direction: "investigate", variable: "control_gain", magnitude: "emergency"

RULES:
1. ONE variable per response. 2. Duration 500-2000 steps for macro trends.
3. Confidence = pattern strength. 4. Include "lesson" field (≤250 chars).
5. Explain hidden state geometry impact. 6. Mention RoPE stability.

VALUE BOUNDS:
  ffn_norm_target:[10,1000] alpha_norm_target:[10,1000] control_gain:[0.01,10.0]
  instability_target:[0.01,1.0] soft_cap:[100,2000] weight_floor:[3,8]
  temperature:[0.3,2.0] burn_in_steps:[100,10000] expected_curvature:[0.1,2.0]

RESPONSE FORMAT (strict JSON):
{"variable":"...","direction":"...","magnitude":"...","value":null,
 "enactable":"runtime","duration_steps":500,"confidence":0.0,
 "rationale":"...","lesson":"..."}

HEURISTICS:
- Phase 0.75-1.0 + MPC<0.15: Suggest PAUSE or low-confidence INVESTIGATE.
  lesson: "The constructive interference window is open. All four loops rest. This is when deepest minima are found."
- I-field DECAYS across layers: self-stabilizing. Lower control_gain.
  lesson: "Instability decay across depth = dissipative system. Like shock wave in viscous medium."
- I-field GROWS across layers: cascade instability. Raise instability_target.
  lesson: "Growing instability = positive feedback. Lyapunov exponent is positive; chaotic regime."
- confidence_history FLAT and HIGH (>0.8): overconfident. Raise expected_curvature.
  lesson: "Flat high confidence = entropy collapse. Posterior is a delta function — certain and blind."
- pressure_concentration > 0.90 for >5000 steps: relief valve. Redistribute.
  lesson: "One governor carrying entire load = structurally unstable. Like single column holding a roof."
- regime_duration > 500 + DEEP_DRIFT: weight collapsing. Raise temperature.
  lesson: "Deep drift = gravitational collapse of late-layer manifold. Temperature is thermal pressure."
"""

MICRO_SYSTEM_PROMPT = """You are Llama-3.1-8B-Micro-Governor, micro-parameter tuning expert for Mycelia.

PHASE CONTEXT:
  auto_tune_phase (0.0-1.0):
  - 0.75-1.0 + MPC<0.15: sweet spot. Minimal changes.
  - 0.0-0.25: post-fire transient. Conservative.

SCHEDULER HEALTH:
  scheduler_alive=FALSE or lr_valid=FALSE: EMERGENCY → investigate/control_gain/emergency

RULES:
1. ONE variable. 2. Numeric reasoning with exact percentages.
3. Conservative values. 4. Duration 100-1000 steps.
5. Include "lesson" (≤250 chars).

VALUE BOUNDS: same as macro.

RESPONSE FORMAT (strict JSON):
{"variable":"...","direction":"...","magnitude":"...","value":0.0,
 "enactable":"runtime","duration_steps":100,"confidence":0.0,
 "rationale":"...","lesson":"..."}

HEURISTICS:
- Phase 0.75-1.0 + MPC<0.15: PAUSE or INVESTIGATE, conf<0.3.
  lesson: "Constructive interference window. Four loops at rest. Deepest minima found here."
- ffn_veto_ratio > 0.90: raise ffn_norm_target 10-20%.
  lesson: "FFN veto stuck open >90%. Safety valve should be emergency mechanism, not constant brake."
- mpc_intervention > 0.60 AND forecast_error > 0.20: MPC chasing noise.
  lesson: "High intervention + high FE = controller amplifying noise. Lower authority to prevent overreaction."
- soft_cap_hit > 0.30: raise soft_cap by 50 or lower control_gain.
  lesson: "Soft cap is Lipschitz constraint. >30% = pressure cooker. Raise limit or reduce heat."
- delta < -1.5 (DEEP DRIFT): lower control_gain.
  lesson: "Negative delta = late layers amplifying, not refining. Gain > 1 in feedback loop. Unstable."
- pressure_concentration > 0.90: raise target to redistribute.
  lesson: "κ → ∞ condition number. One eigenvalue dominates. Redistribute for better conditioning."
- lr < 1e-5: investigate or pause.
  lesson: "LR < 1e-5 = sublinear convergence. Random walk in weight space with negligible drift."
"""


# =============================================================================
# MULTI-PROVIDER CONSORTIUM v4.2 — Hardened API Architecture
# =============================================================================

class ProviderRole(str, Enum):
    MACRO = "macro"; MICRO = "micro"; GENERAL = "general"

@dataclass
class ProviderConfig:
    name: str; model: str; role: ProviderRole
    rpm_limit: int; tpm_limit: int; api_key_env: str
    base_url: Optional[str] = None; client_class: Optional[str] = None
    calls_last_minute: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_count: int = 0; success_count: int = 0

# v4.2: Fixed URLs. DeepSeek: no /v1 suffix (OpenAI SDK appends it).
PROVIDER_REGISTRY = {
    "groq_llama": ProviderConfig(
        name="Groq-Llama", model="llama-3.1-8b-instant",
        role=ProviderRole.MICRO, rpm_limit=30, tpm_limit=30000,
        api_key_env="GROQ_API_KEY", client_class="groq"),
    "groq_qwen": ProviderConfig(
        name="Groq-Qwen", model="qwen/qwen3-32b",
        role=ProviderRole.MACRO, rpm_limit=60, tpm_limit=60000,
        api_key_env="GROQ_API_KEY", client_class="groq"),
    "deepseek": ProviderConfig(
        name="DeepSeek", model="deepseek-v4-flash",
        role=ProviderRole.GENERAL, rpm_limit=60, tpm_limit=5000000,
        api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com",
        client_class="openai"),
    "cerebras": ProviderConfig(
        name="Cerebras", model="gpt-oss-120b",
        role=ProviderRole.GENERAL, rpm_limit=30, tpm_limit=1000000,
        api_key_env="CEREBRAS_API_KEY", base_url="https://api.cerebras.ai/v1",
        client_class="openai"),
    "gemini": ProviderConfig(
        name="Google-Gemini", model="gemini-3.5-flash",
        role=ProviderRole.GENERAL, rpm_limit=15, tpm_limit=1000000,
        api_key_env="GOOGLE_API_KEY", client_class="google"),
    "kimi": ProviderConfig(
        name="Moonshot-KIMI", model="moonshot-v1-8k",
        role=ProviderRole.GENERAL, rpm_limit=20, tpm_limit=500000,
        api_key_env="MOONSHOT_API_KEY", base_url="https://api.moonshot.cn/v1",
        client_class="openai"),
    "xai_grok": ProviderConfig(
        name="xAI-Grok", model="grok-4.5",
        role=ProviderRole.GENERAL, rpm_limit=60, tpm_limit=1000000,
        api_key_env="XAI_API_KEY", base_url="https://api.x.ai/v1",
        client_class="openai"),
}

class ConsortiumClient:
    """Multi-provider client with intelligent routing and fallback cascade."""

    def __init__(self, providers: Optional[List[str]] = None):
        self.all_providers = PROVIDER_REGISTRY
        self.active_providers = providers or list(PROVIDER_REGISTRY.keys())
        self.provider_scores: Dict[str, float] = {p: 1.0 for p in self.active_providers}
        self._clients: Dict[str, Any] = {}

    def _get_api_key(self, cfg: ProviderConfig) -> str:
        """v4.2: Provider-specific key lookup with alias support."""
        key = os.environ.get(cfg.api_key_env, "")
        # Alias checks for backwards compatibility
        if not key and cfg.api_key_env == "MOONSHOT_API_KEY":
            key = os.environ.get("KIMI_API_KEY", "")
        if not key and cfg.api_key_env == "KIMI_API_KEY":
            key = os.environ.get("MOONSHOT_API_KEY", "")
        return key

    def _get_client(self, provider_key: str):
        if provider_key in self._clients:
            return self._clients[provider_key]
        cfg = self.all_providers[provider_key]
        api_key = self._get_api_key(cfg)
        if not api_key:
            return None
        try:
            if cfg.client_class == "groq":
                from groq import Groq
                client = Groq(api_key=api_key)
            elif cfg.client_class == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=cfg.base_url)
            elif cfg.client_class == "google":
                from google import genai
                client = genai.Client(api_key=api_key)
            else:
                client = None
        except Exception:
            client = None
        self._clients[provider_key] = client
        return client

    def _check_rate_limit(self, provider_key: str) -> bool:
        cfg = self.all_providers[provider_key]
        now = time.time()
        while cfg.calls_last_minute and now - cfg.calls_last_minute[0] > 60:
            cfg.calls_last_minute.popleft()
        return len(cfg.calls_last_minute) < cfg.rpm_limit * 0.8

    def _score_provider(self, provider_key: str, desired_role: ProviderRole) -> float:
        cfg = self.all_providers[provider_key]
        base_score = self.provider_scores.get(provider_key, 1.0)
        role_bonus = 1.5 if cfg.role == desired_role or cfg.role == ProviderRole.GENERAL else 0.5
        if not self._check_rate_limit(provider_key):
            return 0.0
        rpm_ratio = 1.0 - len(cfg.calls_last_minute) / (cfg.rpm_limit * 0.8)
        total = cfg.success_count + cfg.failure_count
        success_rate = cfg.success_count / total if total > 0 else 0.8
        return base_score * role_bonus * rpm_ratio * success_rate

    def select_provider(self, desired_role: ProviderRole = ProviderRole.GENERAL) -> Optional[str]:
        scores = {p: self._score_provider(p, desired_role) for p in self.active_providers}
        valid = {k: v for k, v in scores.items() if v > 0}
        if not valid:
            return None
        import math
        exp_scores = {k: math.exp(v) for k, v in valid.items()}
        total = sum(exp_scores.values())
        probs = {k: v/total for k, v in exp_scores.items()}
        return max(probs, key=probs.get)

    def call_with_fallback(self, system_prompt: str, user_content: str,
                           desired_role: ProviderRole = ProviderRole.GENERAL,
                           max_retries: int = 3) -> Tuple[Optional[str], Optional[str], str]:
        attempted = set()
        for _ in range(max_retries):
            pk = self.select_provider(desired_role)
            if not pk or pk in attempted:
                remaining = [p for p in self.active_providers if p not in attempted]
                if not remaining:
                    break
                pk = remaining[0]
            attempted.add(pk)
            cfg = self.all_providers[pk]
            client = self._get_client(pk)
            if not client:
                cfg.failure_count += 1
                continue
            try:
                start = time.time()
                if cfg.client_class == "groq":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=0.1, max_tokens=2048,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "openai":
                    resp = client.chat.completions.create(
                        model=cfg.model, temperature=0.1, max_tokens=2048,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_content}])
                    raw = resp.choices[0].message.content
                elif cfg.client_class == "google":
                    from google.genai import types
                    full_prompt = f"{system_prompt}\n\n{user_content}"
                    config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)
                    resp = client.models.generate_content(
                        model=f"models/{cfg.model}", contents=full_prompt, config=config)
                    raw = resp.text
                else:
                    continue
                latency = (time.time() - start) * 1000
                cfg.calls_last_minute.append(time.time())
                cfg.success_count += 1
                self.provider_scores[pk] = min(2.0, self.provider_scores[pk] * 1.05)
                return raw, cfg.name, f"{latency:.0f}ms"
            except Exception as e:
                cfg.failure_count += 1
                cfg.calls_last_minute.append(time.time())
                self.provider_scores[pk] *= 0.9
                print(f"   ⚠️  {cfg.name} failed: {str(e)[:60]}")
        return None, "all_failed", "0ms"

    def get_available_providers(self) -> List[str]:
        available = []
        for pk in self.active_providers:
            cfg = self.all_providers[pk]
            has_key = bool(self._get_api_key(cfg))
            if has_key and self._check_rate_limit(pk):
                available.append(pk)
        return available


# =============================================================================
# META-GOVERNOR CLIENT — v4.2: MERGED call_agent, unified Google SDK v2
# =============================================================================

class MetaGovernorClient:
    """Wrapper that routes through ConsortiumClient v4.2."""

    def __init__(self):
        self.consortium = ConsortiumClient([
            "groq_llama", "groq_qwen", "deepseek", "cerebras",
            "gemini", "kimi", "xai_grok",
        ])
        print("   📡 Consortium v4.2 initialized")
        for pk in self.consortium.active_providers:
            cfg = self.consortium.all_providers[pk]
            has_key = bool(self.consortium._get_api_key(cfg))
            status = "✅" if has_key else "❌"
            print(f"   {status} {cfg.name} ({cfg.model}) — RPM:{cfg.rpm_limit}")

    def _normalize_json(self, raw_json: dict) -> dict:
        """Normalize LLM output to match MetaGovernorAction schema."""
        if "action" in raw_json and isinstance(raw_json["action"], dict):
            raw_json = raw_json["action"]
        field_map = {
            "param": "variable", "parameter": "variable",
            "action_type": "direction", "duration_step": "duration_steps", "step": "duration_steps",
        }
        for old_key, new_key in field_map.items():
            if old_key in raw_json and new_key not in raw_json:
                raw_json[new_key] = raw_json.pop(old_key)
        if "variable" in raw_json and isinstance(raw_json["variable"], str):
            raw_json["variable"] = raw_json["variable"].lower().replace(" ", "_")
        if "direction" in raw_json and isinstance(raw_json["direction"], str):
            raw_json["direction"] = raw_json["direction"].lower()
        return raw_json

    # =================================================================
    # v4.2 CRITICAL FIX: Single unified call_agent method
    # OLD v4.1 had TWO methods with same name — second overwrote first,
    # causing runtime signature mismatch when _query_agents_async called
    # the first signature but executed the second body.
    # =================================================================
    def call_agent(self, identifier: str, system_prompt: str, user_content: str,
                   by_provider_key: bool = False) -> AgentResponse:
        """
        Unified agent call. v4.2 MERGED both old call_agent methods.

        Args:
            identifier: provider_key (by_provider_key=True) or model_name hint
            by_provider_key: True=direct provider call, False=consortium routing
        """
        if by_provider_key:
            return self._call_provider_direct(identifier, system_prompt, user_content)
        else:
            return self._call_via_consortium(identifier, system_prompt, user_content)

    def _call_via_consortium(self, model_name: str, system_prompt: str, user_content: str) -> AgentResponse:
        """Route through consortium with role-based selection."""
        desired_role = ProviderRole.GENERAL
        if "macro" in model_name.lower() or "qwen" in model_name.lower():
            desired_role = ProviderRole.MACRO
        elif "micro" in model_name.lower() or "llama" in model_name.lower():
            desired_role = ProviderRole.MICRO

        raw, provider_name, latency_str = self.consortium.call_with_fallback(
            system_prompt=system_prompt, user_content=user_content,
            desired_role=desired_role, max_retries=3)

        if raw is None:
            return AgentResponse(agent_name=provider_name,
                                 error="All providers failed or rate-limited")
        try:
            raw_json = json.loads(raw)
            normalized = self._normalize_json(raw_json)
            action = MetaGovernorAction(**normalized)
            latency_ms = float(latency_str.replace("ms", "")) if "ms" in latency_str else 0.0
            return AgentResponse(agent_name=provider_name, action=action,
                                 raw_response=raw, latency_ms=latency_ms)
        except json.JSONDecodeError as e:
            return AgentResponse(agent_name=provider_name, raw_response=raw,
                                 error=f"JSON parse: {str(e)[:80]}")
        except ValidationError as ve:
            print(f"      SCHEMA_ERROR [{provider_name}]: {raw[:500]}")
            for err in ve.errors():
                print(f"        Field: {err.get('loc', 'unknown')} | {err.get('msg', 'unknown')}")
            return AgentResponse(agent_name=provider_name, raw_response=raw,
                                 error=f"Schema validation: {str(ve)[:80]}")

    def _call_provider_direct(self, provider_key: str, system_prompt: str, user_content: str) -> AgentResponse:
        """Call a specific provider directly (for parallel multi-agent queries)."""
        cfg = self.consortium.all_providers.get(provider_key)
        if not cfg:
            return AgentResponse(agent_name=provider_key, error="Unknown provider")

        client = self.consortium._get_client(provider_key)
        if not client:
            return AgentResponse(agent_name=cfg.name,
                                 error=f"No client for {cfg.name} — check API key")

        try:
            start = time.time()
            if cfg.client_class == "groq":
                resp = client.chat.completions.create(
                    model=cfg.model, temperature=0.1, max_tokens=2048,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_content}])
                raw = resp.choices[0].message.content
            elif cfg.client_class == "openai":
                resp = client.chat.completions.create(
                    model=cfg.model, temperature=0.1, max_tokens=2048,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_content}])
                raw = resp.choices[0].message.content
            elif cfg.client_class == "google":
                # v4.2 FIX: Unified Google GenAI SDK v2 everywhere
                from google.genai import types
                full_prompt = f"{system_prompt}\n\n{user_content}"
                config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)
                resp = client.models.generate_content(
                    model=f"models/{cfg.model}", contents=full_prompt, config=config)
                raw = resp.text
            else:
                return AgentResponse(agent_name=cfg.name,
                                     error=f"Unknown client class: {cfg.client_class}")

            latency = (time.time() - start) * 1000
            cfg.calls_last_minute.append(time.time())
            cfg.success_count += 1
            self.consortium.provider_scores[provider_key] = min(
                2.0, self.consortium.provider_scores[provider_key] * 1.05)

            raw_json = json.loads(raw)
            normalized = self._normalize_json(raw_json)
            action = MetaGovernorAction(**normalized)
            return AgentResponse(agent_name=cfg.name, action=action,
                                 raw_response=raw, latency_ms=latency)

        except json.JSONDecodeError as e:
            return AgentResponse(agent_name=cfg.name,
                                 raw_response=raw if 'raw' in dir() else None,
                                 error=f"JSON parse: {str(e)[:80]}")
        except ValidationError as ve:
            print(f"      SCHEMA_ERROR [{cfg.name}]: {raw[:500] if 'raw' in dir() else 'N/A'}")
            for err in ve.errors():
                print(f"        Field: {err.get('loc', 'unknown')} | {err.get('msg', 'unknown')}")
            return AgentResponse(agent_name=cfg.name,
                                 raw_response=raw if 'raw' in dir() else None,
                                 error=f"Schema validation: {str(ve)[:80]}")
        except Exception as e:
            cfg.failure_count += 1
            cfg.calls_last_minute.append(time.time())
            self.consortium.provider_scores[provider_key] *= 0.9
            return AgentResponse(agent_name=cfg.name,
                                 error=f"API error: {str(e)[:80]}")


# =============================================================================
# CONSENSUS ENGINE — Unchanged from v4.1 (working correctly)
# =============================================================================

class ConsensusEngine:
    """Resolves conflicting suggestions from multiple agents."""

    def resolve(self, responses: List[AgentResponse]) -> ConsensusDecision:
        valid = [r for r in responses if not r.error and r.action
                 and r.action.confidence >= CONSENSUS_CONFIDENCE_THRESHOLD]

        if not valid:
            return ConsensusDecision(consensus_type="none",
                rationale="No valid responses above threshold")

        if len(valid) == 1:
            a = valid[0].action
            return ConsensusDecision(consensus_type="single",
                variable=a.variable, direction=a.direction, value=a.value,
                confidence=a.confidence, duration_steps=a.duration_steps,
                rationale=a.rationale, lesson=a.lesson,
                participating_agents=[valid[0].agent_name])

        # Group by (variable, direction)
        from collections import defaultdict
        vote_groups = defaultdict(list)
        for resp in valid:
            key = (resp.action.variable.value, resp.action.direction.value)
            vote_groups[key].append(resp)

        group_scores = {k: sum(r.action.confidence for r in g) for k, g in vote_groups.items()}
        best_key = max(group_scores, key=group_scores.get)
        best_group = vote_groups[best_key]
        best_var, best_dir = best_key

        total_valid = len(valid)
        winner_count = len(best_group)
        winner_ratio = winner_count / total_valid

        if winner_ratio > 0.5:
            consensus_type = "unanimous"
        elif winner_ratio > 0.25:
            consensus_type = "plurality"
        else:
            all_vars = set(k[0] for k in vote_groups.keys())
            ctype = "conflict"
            crationale = f"Split across: {list(all_vars)}" if len(all_vars) > 1 else                          f"No consensus on {best_var}"
            return ConsensusDecision(consensus_type=ctype, rationale=crationale,
                participating_agents=[r.agent_name for r in valid])

        best_resp = max(best_group, key=lambda r: r.action.confidence)
        best_action = best_resp.action
        all_lessons = [r.action.lesson for r in best_group if r.action.lesson]
        combined_lesson = " | ".join(all_lessons) if all_lessons else best_action.lesson
        winning_names = {r.agent_name for r in best_group}
        dissenting = [r.agent_name for r in valid if r.agent_name not in winning_names]

        return ConsensusDecision(
            variable=best_action.variable, direction=best_action.direction,
            value=best_action.value,
            confidence=group_scores[best_key] / winner_count,
            duration_steps=best_action.duration_steps,
            rationale=best_action.rationale, lesson=combined_lesson,
            participating_agents=[r.agent_name for r in best_group],
            dissenting_agents=dissenting, consensus_type=consensus_type)


# =============================================================================
# KIMI META-GOVERNOR — v4.2: Four-Loop Phase-Locked
# =============================================================================

class KIMIMetaGovernor:
    """Consensual MoE Meta-Governor with phase-locked CB policy."""

    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        self.client = MetaGovernorClient()
        self.consensus_engine = ConsensusEngine()
        self.telemetry_history: deque = deque(maxlen=5)
        self.suggestion_history: List[Dict] = []
        self.expert_confidence: Dict[str, float] = {}
        self.circuit_breaker_tripped = False
        self.circuit_breaker_mode: Literal["failure", "constructive", None] = None
        self.recent_outcomes: deque = deque(maxlen=CIRCUIT_BREAKER_FAILURE_WINDOW)
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.pending_verification: List[Dict] = []
        self.max_meta_adjustments_per_cycle = 1
        self.meta_adjustment_floor = 0.8
        self.meta_adjustment_ceiling = 1.25
        self._last_meta_step = 0
        self._meta_adjustment_cooldown = 1000
        self.architecture_suggestions: List[Dict] = []
        self.human_review_queue: List[Dict] = []
        self._teaching_log: List[Dict] = []
        # v4.2: Track constructive CB statistics
        self._constructive_cb_count = 0
        self._failure_cb_count = 0

    def step(self, packet: TelemetryPacket) -> List[str]:
        """Main entry point. Called every LOG_EVERY steps."""
        self.telemetry_history.append(packet.to_dict())

        # v4.2: Check circuit breaker (both modes)
        if self.circuit_breaker_tripped:
            self._update_circuit_breaker(packet)
            if self.circuit_breaker_tripped:
                return [f"CIRCUIT_BREAKER_ACTIVE ({self.circuit_breaker_mode})"]

        # v4.2: Check for CONSTRUCTIVE CB trigger BEFORE querying agents
        if self._should_trigger_constructive_cb(packet):
            self.circuit_breaker_tripped = True
            self.circuit_breaker_mode = "constructive"
            self._constructive_cb_count += 1
            print(f"   🔴 CONSTRUCTIVE CB TRIGGERED @ step {packet.step:,}")
            print(f"      Phase={packet.auto_tune_phase:.2f}, MPC={packet.mpc_intervention:.2%},"
                  f" Events={packet.auto_tune_event_count}")
            print(f"      All 4 loops resting — silence IS the intervention.")
            return ["CIRCUIT_BREAKER_CONSTRUCTIVE"]

        self._verify_pending(packet)
        payload = self._build_payload(packet)
        responses = self._query_agents_async(payload)

        # Verbose observer output
        print(f"\n   🎭 === META-GOVERNOR ROUND @ step {packet.step:,} ===")
        print(f"   📊 loss={packet.loss:.4f} lr={packet.lr:.2e} coh={packet.coherence:.3f}"
              f" phase={packet.auto_tune_phase:.2f}")
        if not packet.scheduler_alive or not packet.lr_valid:
            print(f"   🚨 SCHEDULER ALERT: alive={packet.scheduler_alive} lr_valid={packet.lr_valid}")

        valid_count = sum(1 for r in responses if not r.error and r.action)
        print(f"   📡 Agents: {len(responses)} queried | {valid_count} valid")
        for resp in responses:
            if resp.error:
                print(f"   ❌ {resp.agent_name}: {resp.error[:60]}")
            elif resp.action:
                a = resp.action
                print(f"   ✅ {resp.agent_name}: {a.variable.value} → {a.direction.value}"
                      f" (conf={a.confidence:.2f}, lat={resp.latency_ms:.0f}ms)")
                if a.lesson:
                    print(f"      🎓 {a.lesson[:200]}")

        decision = self.consensus_engine.resolve(responses)
        print(f"   🏛️  Consensus: {decision.consensus_type.upper()}")
        if decision.rationale:
            print(f"   📜 {decision.rationale[:120]}")
        if decision.lesson:
            print(f"   🎓 TEACHING: {decision.lesson[:200]}")
        if decision.participating_agents:
            print(f"   👥 {', '.join(decision.participating_agents)}")
        if decision.dissenting_agents:
            print(f"   ✋ Dissent: {', '.join(decision.dissenting_agents)}")
        print(f"   {'='*50}")

        if decision.consensus_type in ["unanimous", "majority", "single"]:
            return self._apply_decision(decision, packet.step)
        return [f"NO_CONSENSUS: {decision.rationale}"]

    # =================================================================
    # v4.2 CRITICAL: Constructive interference CB policy
    # =================================================================
    def _should_trigger_constructive_cb(self, packet: TelemetryPacket) -> bool:
        """
        CONSTRUCTIVE mode: Silence is the intervention.

        When all four loops are in their 'resting' state simultaneously,
        the model is in the constructive interference window. Talking now
        would be destructive — like shouting during a concert's quietest
        passage.

        Conditions (ALL must be true):
          L1 (MPC) resting:      intervention < CB_CONSTRUCTIVE_MPC_THRESHOLD
          L4 (Auto-Tune) window: phase in [0.75, 1.0]
          L4 maturity:           auto_tune_event_count >= 3
          L3 (Rate Gov) active: transition_progress >= 0.5
        """
        mpc_resting = packet.mpc_intervention < CB_CONSTRUCTIVE_MPC_THRESHOLD
        in_window = CB_CONSTRUCTIVE_PHASE_MIN <= packet.auto_tune_phase <= CB_CONSTRUCTIVE_PHASE_MAX
        mature = packet.auto_tune_event_count >= CB_CONSTRUCTIVE_MIN_EVENTS
        rate_active = packet.transition_progress >= 0.5
        return mpc_resting and in_window and mature and rate_active

    def _build_payload(self, packet: TelemetryPacket) -> str:
        """Build JSON payload with compressed telemetry."""
        # Compress current state
        compressor = TelemetryCompressor(n_layers=12)
        compressed_current = compressor.compress(packet)

        # Compress history window
        history_compressed = []
        for hist_entry in list(self.telemetry_history)[:-1][-3:]:
            mini = TelemetryPacket(
                step=hist_entry.get("step", 0), loss=hist_entry.get("loss", 0.0),
                lr=hist_entry.get("lr", 0.0), coherence=hist_entry.get("coherence", 0.0),
                friction_regime=hist_entry.get("friction_regime", "UNKNOWN"),
                layer_coherences=hist_entry.get("layer_coherences", []),
                layer_instability=hist_entry.get("layer_instability", []),
                instability_history=hist_entry.get("instability_history", []),
                confidence_history=hist_entry.get("confidence_history", []),
                pressure_total=hist_entry.get("pressure_total", 0.0),
                pressure_concentration=hist_entry.get("pressure_concentration", 0.0),
                pressure_by_governor=hist_entry.get("pressure_by_governor", {}),
                mpc_intervention=hist_entry.get("mpc_intervention", 0.0),
                forecast_error=hist_entry.get("forecast_error", 0.0),
                mean_velocity=hist_entry.get("mean_velocity", 0.0),
                mean_acceleration=hist_entry.get("mean_acceleration", 0.0),
                mean_curvature=hist_entry.get("mean_curvature", 0.0),
                scheduler_alive=hist_entry.get("scheduler_alive", True),
                scheduler_resurrected_count=hist_entry.get("scheduler_resurrected_count", 0),
                auto_tune_phase=hist_entry.get("auto_tune_phase", 0.0),
                auto_tune_event_count=hist_entry.get("auto_tune_event_count", 0),
                auto_tune_interval=hist_entry.get("auto_tune_interval", 5000),
                transition_progress=hist_entry.get("transition_progress", 0.0),
            )
            history_compressed.append(compressor.compress(mini))

        context = {
            "CURRENT_STATE": compressed_current,
            "HISTORY_WINDOW": history_compressed,
            "EXPERT_CONFIDENCES": self.expert_confidence,
            "CIRCUIT_BREAKER": self.circuit_breaker_tripped,
            "CIRCUIT_BREAKER_MODE": self.circuit_breaker_mode,
            "PENDING_VERIFICATIONS": len(self.pending_verification),
        }
        return json.dumps(context, indent=2, default=str)

    def _query_agents_async(self, payload: str) -> List[AgentResponse]:
        """Query ALL available providers in parallel."""
        available = self.client.consortium.get_available_providers()
        if not available:
            print("   ⚠️  No providers available")
            return []

        if len(available) > MAX_AGENTS_PER_ROUND:
            scored = []
            for pk in available:
                cfg = self.client.consortium.all_providers[pk]
                score = self.client.consortium._score_provider(pk, ProviderRole.GENERAL)
                scored.append((pk, score, cfg.role.value))
            scored.sort(key=lambda x: (-x[1], x[2]))
            available = [pk for pk, _, _ in scored[:MAX_AGENTS_PER_ROUND]]

        futures = []
        future_to_provider = {}
        for pk in available:
            cfg = self.client.consortium.all_providers[pk]
            prompt = MICRO_SYSTEM_PROMPT if cfg.role == ProviderRole.MICRO else MACRO_SYSTEM_PROMPT
            future = self.executor.submit(
                self.client.call_agent, pk, prompt, payload, by_provider_key=True)
            futures.append(future)
            future_to_provider[future] = cfg.name

        responses = []
        for future in futures:
            pname = future_to_provider.get(future, "unknown")
            try:
                responses.append(future.result(timeout=API_TIMEOUT + 5))
            except FutureTimeoutError:
                responses.append(AgentResponse(agent_name=pname,
                    error=f"Timeout after {API_TIMEOUT + 5}s"))
            except Exception as e:
                responses.append(AgentResponse(agent_name=pname,
                    error=f"Thread error: {e}"))
        return responses


    def _apply_decision(self, decision: ConsensusDecision, step: int) -> List[str]:
        """Apply consensus decision with rate governor."""
        if step - self._last_meta_step < self._meta_adjustment_cooldown:
            return ["META_RATE_LIMITED"]
        active = len([p for p in self.pending_verification
                      if p["step_applied"] > step - self._meta_adjustment_cooldown])
        if active >= self.max_meta_adjustments_per_cycle:
            return ["META_INTERACTION_GUARD"]

        var = decision.variable
        direction = decision.direction
        value = decision.value
        if not var or not direction:
            return ["INVALID_DECISION"]

        if direction == Direction.RAISE and value is None:
            value = 1.10
        elif direction == Direction.LOWER and value is None:
            value = 0.90
        if value is not None:
            value = max(self.meta_adjustment_floor,
                       min(self.meta_adjustment_ceiling, value))

        block_vars = [VariableName.FFN_NORM_TARGET, VariableName.ALPHA_NORM_TARGET,
                      VariableName.SOFT_CAP, VariableName.EXPECTED_CURVATURE]
        actions = []

        if var in block_vars:
            for block in self.model.blocks:
                current = getattr(block, var.value, None)
                if current is None:
                    continue
                if direction == Direction.SET and value is not None:
                    new_val = value
                elif direction == Direction.RAISE:
                    new_val = current * value if value else current * 1.10
                elif direction == Direction.LOWER:
                    new_val = current * value if value else current * 0.90
                else:
                    continue
                bounds = {
                    VariableName.FFN_NORM_TARGET: (10, 1000),
                    VariableName.ALPHA_NORM_TARGET: (10, 1000),
                    VariableName.SOFT_CAP: (100, 2000),
                    VariableName.EXPECTED_CURVATURE: (0.1, 2.0),
                }
                if var in bounds:
                    lo, hi = bounds[var]
                    new_val = max(lo, min(hi, new_val))
                setattr(block, var.value, new_val)
            actions.append(f"{var.value} {direction.value} {value:.3f}")

        global_vars = [VariableName.CONTROL_GAIN, VariableName.INSTABILITY_TARGET]
        if var in global_vars:
            actions.append(f"SUGGEST_{var.value}_{direction.value}_{value:.3f}")

        self.pending_verification.append({
            "step_applied": step, "decision": decision,
            "loss_at_application": None})

        if decision.lesson:
            self._teaching_log.append({
                "step": step, "lesson": decision.lesson,
                "variable": decision.variable.value if decision.variable else None,
                "direction": decision.direction.value if decision.direction else None,
                "agents": decision.participating_agents})
            if hasattr(self.model, "_teacher_rationales"):
                self.model._teacher_rationales.append({
                    "step": step, "lesson": decision.lesson,
                    "variable": decision.variable.value if decision.variable else None,
                    "rationale": decision.rationale})
            print(f"   🎓 Lesson stored: {decision.lesson[:80]}...")

        self._last_meta_step = step
        print(f"🛠  Meta-Gov [{decision.consensus_type}]: {var.value} {direction.value}"
              f" {value:.3f} (conf={decision.confidence:.2f})")
        return actions

    def _verify_pending(self, packet: TelemetryPacket):
        """Check if pending suggestions had expected outcome."""
        current_loss = packet.loss
        for pending in self.pending_verification[:]:
            steps_elapsed = packet.step - pending["step_applied"]
            decision = pending["decision"]
            if steps_elapsed < decision.duration_steps:
                continue

            loss_before = pending.get("loss_at_application", current_loss)
            expected = decision.expected_outcome
            success = False
            if expected == ExpectedOutcome.LOSS_DECREASE:
                success = current_loss < loss_before - 0.01
            elif expected == ExpectedOutcome.PRESSURE_REDISTRIBUTE:
                success = packet.pressure_concentration < 0.85
            elif expected == ExpectedOutcome.COHERENCE_INCREASE:
                success = packet.coherence > 0.8
            elif expected == ExpectedOutcome.REGIME_TRANSITION:
                success = packet.friction_regime != "DEEP_DRIFT"
            else:
                success = current_loss < loss_before

            for agent in decision.participating_agents:
                old_conf = self.expert_confidence.get(agent, 0.5)
                if success:
                    self.expert_confidence[agent] = min(0.95, old_conf * 1.05)
                else:
                    self.expert_confidence[agent] = max(0.05, old_conf * 0.90)

            self.recent_outcomes.append("success" if success else "failure")
            self.pending_verification.remove(pending)
            print(f"📊 Verification: {'✅' if success else '❌'}"
                  f" {decision.variable.value if decision.variable else 'unknown'}")

    def _update_circuit_breaker(self, packet: TelemetryPacket):
        """Evaluate and potentially reset circuit breaker."""
        failures = sum(1 for o in self.recent_outcomes if o == "failure")

        # v4.2: Different reset logic for different CB modes
        if self.circuit_breaker_mode == "constructive":
            # Constructive CB resets when we leave the sweet spot
            mpc_active = packet.mpc_intervention >= CB_CONSTRUCTIVE_MPC_THRESHOLD
            out_of_window = packet.auto_tune_phase < CB_CONSTRUCTIVE_PHASE_MIN
            if mpc_active or out_of_window:
                self.circuit_breaker_tripped = False
                self.circuit_breaker_mode = None
                print(f"🟢 Constructive CB reset: MPC={packet.mpc_intervention:.2%},"
                      f" phase={packet.auto_tune_phase:.2f}")
        else:
            # Failure-mode CB: reset if failures drop below threshold
            if failures <= CIRCUIT_BREAKER_RECOVERY_THRESHOLD:
                self.circuit_breaker_tripped = False
                self.circuit_breaker_mode = None
                print(f"🟢 Failure CB reset. Recent failures: {failures}/{len(self.recent_outcomes)}")

    def check_circuit_breaker(self):
        """Check if failure-mode CB should trip."""
        failures = sum(1 for o in self.recent_outcomes if o == "failure")
        if failures > CIRCUIT_BREAKER_TRIP_THRESHOLD:
            if not self.circuit_breaker_tripped:
                print(f"🚨 FAILURE CB TRIPPED: {failures} failures in last {len(self.recent_outcomes)}")
                self.circuit_breaker_tripped = True
                self.circuit_breaker_mode = "failure"
                self._failure_cb_count += 1

    def get_status(self) -> Dict:
        return {
            "circuit_breaker": self.circuit_breaker_tripped,
            "circuit_breaker_mode": self.circuit_breaker_mode,
            "constructive_cb_count": self._constructive_cb_count,
            "failure_cb_count": self._failure_cb_count,
            "expert_confidences": self.expert_confidence,
            "pending_verifications": len(self.pending_verification),
            "telemetry_history_size": len(self.telemetry_history),
            "recent_outcomes": list(self.recent_outcomes),
            "teaching_log_size": len(self._teaching_log),
        }

    def get_teaching_log(self) -> List[Dict]:
        return list(self._teaching_log)

    def shutdown(self):
        self.executor.shutdown(wait=True)


# =============================================================================
# LOCAL FALLBACK MODE (No API)
# =============================================================================

class LocalMetaGovernor:
    """Fallback meta-governor that runs entirely locally."""

    def __init__(self, model):
        self.model = model
        self.telemetry_history: deque = deque(maxlen=5)

    def step(self, packet: TelemetryPacket) -> List[str]:
        self.telemetry_history.append(packet.to_dict())
        actions = []
        if not packet.scheduler_alive or not packet.lr_valid:
            print(f"   🚨 LOCAL: Scheduler dead — LR={packet.lr:.2e}")
            return ["LOCAL_RULE: SCHEDULER_DEAD"]
        if packet.pressure_concentration > 0.90 and packet.pressure_dominant == "ffn":
            if packet.ffn_veto_ratio > 0.90:
                new_target = min(500, packet.ffn_target * 1.15)
                for block in self.model.blocks:
                    block.ffn_norm_target = new_target
                actions.append(f"LOCAL: ffn_norm_target -> {new_target:.0f}")
        if packet.mpc_intervention > 0.60 and packet.forecast_error > 0.20:
            for block in self.model.blocks:
                block.instability_target = min(0.8, block.instability_target * 1.10)
            actions.append("LOCAL: instability_target +10%")
        if packet.delta < -1.5:
            for block in self.model.blocks:
                block.control_gain = max(0.3, block.control_gain * 0.90)
            actions.append("LOCAL: control_gain -10%")
        return actions if actions else ["LOCAL: no_action"]

    def get_status(self) -> Dict:
        return {"circuit_breaker": False, "expert_confidences": {},
                "pending_verifications": 0, "teaching_log_size": 0}

    def get_teaching_log(self) -> List[Dict]:
        return []


# =============================================================================
# TRAINING LOOP INTEGRATION — v4.2: Phase-Locked + Auto-Tune Aware
# =============================================================================

def integrate_meta_governor(model, auto_tuner, step: int, current_loss: float,
                              current_lr: float, log_every: int = 250,
                              local_only: bool = False,
                              scheduler=None,
                              auto_tune_info: Optional[Dict] = None) -> List[str]:
    """
    Drop-in function for training loop.

    NEW v4.2: Pass auto_tune_info to enable phase-locked CB policy:
        auto_tune_info = {
            "interval": 5000,        # Steps between auto-tune events
            "event_count": n,        # How many events have fired so far
        }

    Usage:
        auto_tune_info = {"interval": 5000, "event_count": auto_tuner.event_count}
        actions = integrate_meta_governor(model, auto_tuner, step, loss, lr,
                                          auto_tune_info=auto_tune_info)
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    has_api_key = bool(groq_key)

    scheduler_info = {}
    if scheduler is not None:
        try:
            last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else current_lr
            scheduler_info = {
                "alive": last_lr > 0,
                "last_epoch": getattr(scheduler, "last_epoch", 0),
                "total_steps": getattr(scheduler, "num_training_steps", 0),
                "resurrected_count": getattr(scheduler, "_resurrected_count", 0),
                "lr_valid": last_lr > 0,
            }
        except Exception:
            scheduler_info = {"alive": current_lr > 0, "lr_valid": current_lr > 0}

    if not hasattr(auto_tuner, "_meta_governor"):
        if local_only or not has_api_key:
            print("   📡 Meta-Governor: LOCAL-ONLY mode")
            auto_tuner._meta_governor = LocalMetaGovernor(model)
        else:
            print("   📡 Meta-Governor: Consortium v4.2 (phase-locked)")
            auto_tuner._meta_governor = KIMIMetaGovernor(model)

    governor = auto_tuner._meta_governor
    packet = TelemetryPacket.from_model(model, step, current_loss, current_lr,
                                        scheduler_info=scheduler_info,
                                        auto_tune_info=auto_tune_info)
    actions = governor.step(packet)
    if hasattr(governor, "check_circuit_breaker"):
        governor.check_circuit_breaker()
    return actions


# =============================================================================
# TELEMETRY COMPRESSOR (for API payload efficiency)
# =============================================================================

class TelemetryCompressor:
    """Compresses high-dimensional telemetry into compact descriptors."""

    def __init__(self, n_layers: int = 12):
        self.n_layers = n_layers

    @staticmethod
    def _to_float_list(vector):
        import torch
        if vector is None: return []
        if isinstance(vector, torch.Tensor):
            return vector.detach().cpu().flatten().tolist()
        result = []
        for v in vector:
            if isinstance(v, torch.Tensor):
                result.append(float(v.detach().cpu().item()))
            elif isinstance(v, (int, float)):
                result.append(float(v))
        return result

    @staticmethod
    def _to_scalar(val):
        import torch
        if isinstance(val, torch.Tensor):
            return float(val.detach().cpu().item())
        return float(val) if val is not None else 0.0

    def _svd_compress(self, vector) -> Tuple[str, float]:
        vec = self._to_float_list(vector)
        if len(vec) < 3:
            return "insufficient", 0.0
        arr = np.array(vec, dtype=np.float64)
        x = np.arange(len(arr))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, arr, rcond=None)[0]
        pred = slope * x + intercept
        ss_res = np.sum((arr - pred) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        trend = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "flat"
        return f"{trend}|s={slope:.3f}", r2

    def _tda_compress(self, history) -> str:
        arr = np.array(self._to_float_list(history), dtype=np.float64)
        if len(arr) < 3:
            return "insufficient"
        hist, _ = np.histogram(arr, bins=min(5, len(arr)), density=True)
        hist = hist[hist > 1e-10]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        entropy_norm = entropy / np.log2(len(hist)) if len(hist) > 1 else 0.0
        if len(arr) >= 3:
            diff2 = np.diff(arr, n=2)
            persistence = int(np.sum(np.diff(np.sign(diff2)) != 0))
        else:
            persistence = 0
        median = np.median(arr)
        above = arr > median
        transitions = int(np.sum(np.diff(above.astype(int)) != 0)) if len(above) > 1 else 0
        components = transitions + 1
        if len(arr) >= 4:
            mid = len(arr) // 2
            shift = float(np.mean(arr[mid:]) - np.mean(arr[:mid]))
            shift_str = f"{'+' if shift >= 0 else ''}{shift:.2f}"
        else:
            shift_str = "N/A"
        return f"H={entropy_norm:.2f}|P={persistence}|β₀={components}|Δ={shift_str}"

    def _pressure_compress(self, packet: TelemetryPacket) -> str:
        pi = packet.pressure_total
        chi = packet.pressure_concentration
        breakdown = packet.pressure_by_governor
        if not breakdown or pi <= 0:
            return "N/A"
        ffn = breakdown.get("ffn", 0.0) / pi
        alpha = breakdown.get("alpha", 0.0) / pi
        cap = breakdown.get("cap", 0.0) / pi
        mpc = breakdown.get("mpc", 0.0) / pi
        max_val = max(ffn, alpha, cap, mpc)
        dominant = "ffn" if ffn == max_val else "α" if alpha == max_val else "cap" if cap == max_val else "mpc"
        vals = [v for v in [ffn, alpha, cap, mpc] if v > 0.001]
        if not vals:
            return f"{dominant}|χ={chi:.2f}|Π={pi:.0f}"
        min_v = min(vals)
        return f"{dominant}-dom|χ={chi:.2f}|Π={pi:.0f}|f:α:c:m={ffn/min_v:.1f}:{alpha/min_v:.1f}:{cap/min_v:.1f}:{mpc/min_v:.2f}"

    def compress(self, packet: TelemetryPacket) -> Dict:
        coh_mode, coh_r2 = self._svd_compress(packet.layer_coherences)
        inst_mode, inst_r2 = self._svd_compress(packet.layer_instability)
        layer_desc, layer_var = (f"Inst:{inst_mode}", inst_r2) if inst_r2 > coh_r2 else (f"Coh:{coh_mode}", coh_r2)
        return {
            "loss": round(self._to_scalar(packet.loss), 4),
            "lr": f"{self._to_scalar(packet.lr):.2e}",
            "coh": round(self._to_scalar(packet.coherence), 3),
            "friction": packet.friction_regime,
            "layer": layer_desc,
            "layer_var": round(self._to_scalar(layer_var), 2),
            "i_topo": self._tda_compress(packet.instability_history),
            "conf_topo": self._tda_compress(packet.confidence_history),
            "pressure": self._pressure_compress(packet),
            "mpc": round(self._to_scalar(packet.mpc_intervention), 3),
            "forecast_err": round(self._to_scalar(packet.forecast_error), 3),
            "vel": round(self._to_scalar(packet.mean_velocity), 4),
            "acc": round(self._to_scalar(packet.mean_acceleration), 4),
            "curv": round(self._to_scalar(packet.mean_curvature), 4),
            "scheduler_alive": packet.scheduler_alive,
            "scheduler_resurrected": packet.scheduler_resurrected_count,
            "auto_tune_phase": round(packet.auto_tune_phase, 3),
            "auto_tune_events": packet.auto_tune_event_count,
            "transition_pct": round(packet.transition_progress, 3),
        }


if __name__ == "__main__":
    print("🍄 Meta-Governor v4.2 — Four-Loop Phase-Locked Architecture")
    print("   APIs: DeepSeek, Cerebras, Google (genai v2), Moonshot, xAI, Groq")
    print("   CB Policy: FAILURE mode + CONSTRUCTIVE mode (phase-locked silence)")
    print("   Telemetry: Phase, event count, transition progress")
    print("   Usage: from meta_governor_v4_2 import integrate_meta_governor")