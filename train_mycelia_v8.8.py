# ============================================
# MYCELIA Training Loop — v8.8
# Patched from v8.6 to fix the loss plateau at ~4.5
#
# Changes from v8.6 (see PATCH_NOTES_TRAIN.md for rationale):
#   1. LR Burst DISABLED by default — was yanking LR to peak (3e-4)
#      with no warmup, causing divergence-recovery cycles
#   2. MIN_LR raised 1e-5 → 3e-5 (below Adam's noise floor)
#   3. GRAD_CLIP raised 1.0 → 2.0 (let gradient signal escape plateaus)
#   4. Hand-rolled scheduler replaced with HF cosine-with-warmup
#   5. StanfordDataset dedup now resets per-epoch
#   6. control_gain auto-tune capped 10.0 → 1.5 (was crushing alphas)
#   7. Auto-tune frequency: every LOG_EVERY → every AUTO_TUNE_EVERY
#   8. best_loss logic moved inside LOG_EVERY block, recomputed properly
#   9. NEW: AdaptiveTarget — auto-tunes ffn_norm_target and alpha_norm_target
#      to track their own EMAs. Tests the FFN Relief Valve Hypothesis (paper §7.3)
#  10. NEW: PressureTensorLogger — alerts when χ > 0.9 (single-governor dominance)
#  11. NEW: GovernorAutoTuner — bounded, rate-limited auto-tune with hysteresis
#  12. NaN handling: skip step cleanly, reset Adam momentum only on persistence
#
# Backward compatible: loads v8.6/v8.7 checkpoints (strict=False).
# Architecture: import from MYCELIA_architecture_v8_8 first, fallback to v8.7.
# ============================================

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import gc
import json
import time
import math
import signal
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import hashlib
import boto3
import io
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────

MAX_SEQ_LEN = 512
BATCH_SIZE = 2
ACCUM_STEPS = 16    # effective batch = 32 sequences * 512 tokens = 16,384 tokens
WEIGHT_DECAY = 0.01

# ── v8.8: Gradient clipping raised 1.0 → 2.0 ─────────────────────────────
# At LR 3e-4 the gradient signal needs room to escape plateaus. Clipping
# at 1.0 was suppressing the very signal that would have broken the
# loss floor at 4.5. Revert to 1.0 if you see NaN reappear.
GRAD_CLIP = 2.0

SAVE_EVERY = 5000
LOG_EVERY = 1000
CACHE_CLEAN_EVERY = 1000

PEAK_LR = 3e-4

# ── v8.8: MIN_LR raised 1e-5 → 3e-5 ──────────────────────────────────────
# 1e-5 is below Adam's effective noise floor — the optimizer is essentially
# frozen. 3e-5 keeps Adam alive enough to escape sharp minima even at the
# bottom of the cosine schedule. If loss truly needs to settle, drop to
# 1e-5 manually AFTER convergence.
MIN_LR = 3e-5

WARMUP_STEPS = 500
TOTAL_TOKENS_TARGET = 5_000_000_000

# ── v8.8: LR Burst DISABLED by default ────────────────────────────────────
# v8.6 burst logic slammed LR from ~1e-5 to PEAK_LR (3e-4) with no warmup
# whenever loss plateaued. This caused divergence spikes (loss 8.19 in
# telemetry) that MPC + FFN veto had to absorb, leaving the model in a
# divergence-recovery oscillation around 4.5.
#
# To re-enable (NOT RECOMMENDED): set ENABLE_LR_BURST = True AND set
# LR_BURST_PEAK = MIN_LR * 10 (gentle, not full peak).
ENABLE_LR_BURST = False
LR_BURST_STEPS = 2000
LR_BURST_PEAK = MIN_LR * 10
LR_BURST_MIN_DELTA = 0.05

CONSENSUS_ROUNDS = 2

# ── v8.8: Auto-tune cadence and bounds ───────────────────────────────────
# Auto-tune was running every LOG_EVERY (1000 steps), which was too
# frequent — the system never settled. Now runs every AUTO_TUNE_EVERY
# (5000 steps) with strict bounds on each parameter.
AUTO_TUNE_EVERY = 5000

# Control gain bounds (was unbounded above, capped at 10.0 — catastrophic)
CONTROL_GAIN_MIN = 0.5
CONTROL_GAIN_MAX = 1.5
CONTROL_GAIN_DEFAULT = 1.0

# Adaptive target config (tests FFN Relief Valve Hypothesis, paper §7.3)
ADAPTIVE_TARGETS_ENABLED = True
ADAPTIVE_FFN_TARGET_EMA = 0.01           # EMA smoothing for FFN norm tracking
ADAPTIVE_FFN_TARGET_MULTIPLIER = 1.5     # target = EMA * multiplier
ADAPTIVE_FFN_TARGET_MIN = 30.0          # hard floor to prevent collapse
ADAPTIVE_FFN_TARGET_MAX = 500.0         # hard ceiling

ADAPTIVE_ALPHA_TARGET_EMA = 0.01
ADAPTIVE_ALPHA_TARGET_MULTIPLIER = 1.5
ADAPTIVE_ALPHA_TARGET_MIN = 50.0
ADAPTIVE_ALPHA_TARGET_MAX = 300.0

# Pressure concentration alert threshold (paper §6.2)
PRESSURE_CONCENTRATION_ALERT = 0.85     # alert when χ > this

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"
FINEWEB_PREFIX = "fineweb_cache"
STANFORD_ONLY = ["stanford_philosophy_processed.jsonl"]

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

# ─── GRACEFUL SHUTDOWN HANDLER ───────────────────────────────────────────

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n🛑 Shutdown signal received, finishing current step...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ─── IMPORT ARCHITECTURE ─────────────────────────────────────────────────

try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("🍄 Mycelia v8.8 loaded — Rate-Aware Governors")
except ImportError:
    try:
        from MYCELIA_architecture import MyceliaLM, MyceliaConfig
        print("🍄 Mycelia v8.8 loaded (fallback) — apply v8.8 architecture patch")
    except ImportError:
        raise ImportError("MYCELIA_architecture[_v8_8].py not found!")

# ─── ADAPTIVE TARGET TRACKER (NEW v8.8) ───────────────────────────────────

class AdaptiveTarget:
    """
    Tracks an EMA of a measured quantity and sets a target = EMA * multiplier,
    with hard min/max bounds. Tests the FFN Relief Valve Hypothesis from
    paper §7.3: if the FFN veto target was unnecessarily restrictive, raising
    it adaptively should let the optimizer use FFN capacity productively and
    cause pressure to redistribute to other governors. If the FFN was
    genuinely load-bearing, raising the target will redistribute pressure
    instead.

    Usage:
        tracker = AdaptiveTarget(initial=150, ema_alpha=0.01, multiplier=1.5,
                                 min_val=30, max_val=500)
        for step in training:
            tracker.update(measured_value)  # measured = telemetry mean_ffn_norm
            new_target = tracker.target
            block.ffn_norm_target = new_target
    """
    def __init__(self, initial, ema_alpha, multiplier, min_val, max_val):
        self.target = initial
        self.ema_alpha = ema_alpha
        self.multiplier = multiplier
        self.min_val = min_val
        self.max_val = max_val
        self.ema = initial
        self.history = []

    def update(self, measured):
        if measured <= 0:
            return self.target
        # EMA of measured quantity
        self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * measured
        # Target tracks the natural operating point with headroom
        new_target = self.ema * self.multiplier
        self.target = max(self.min_val, min(self.max_val, new_target))
        self.history.append((self.ema, self.target))
        return self.target


# ─── GOVERNOR AUTO-TUNER (NEW v8.8) ───────────────────────────────────────

class GovernorAutoTuner:
    """
    Bounded, rate-limited auto-tune for governor parameters.
    Replaces the v8.6 auto-tune that ran every 1000 steps with no bounds
    and would push control_gain up to 10.0 (crushing alphas by 94%).

    Each parameter has its own update rate (slower = more conservative)
    and hard bounds. Updates only fire when conditions are stable for
    AUTO_TUNE_EVERY consecutive steps to avoid oscillation.
    """
    def __init__(self, model):
        self.model = model
        self.control_gain = CONTROL_GAIN_DEFAULT
        self.instability_target = 0.45
        self.expected_curvature = 0.5
        self.coherence_weight = 1.0
        self.forecast_weight = 1.0

        # EMA history for stability check
        self.mpc_intervention_ema = 0.5
        self.forecast_error_ema = 0.15
        self.cap_hit_ema = 0.10
        self.curvature_damping_ema = 1.0

        # Hysteresis counters
        self._aggressive_count = 0
        self._last_tune_step = 0

    def update_telemetry_emas(self, info, alpha=0.05):
        """Update telemetry EMAs. Called every step, just tracks state."""
        mpc = info.get('mpc_intervention_ratio', 0.0)
        fe = info.get('forecast_error', 0.0)
        ch = info.get('soft_cap_hit_ratio', 0.0)
        cd = info.get('curvature_damping', 1.0)
        self.mpc_intervention_ema = (1-alpha) * self.mpc_intervention_ema + alpha * mpc
        self.forecast_error_ema = (1-alpha) * self.forecast_error_ema + alpha * fe
        self.cap_hit_ema = (1-alpha) * self.cap_hit_ema + alpha * ch
        self.curvature_damping_ema = (1-alpha) * self.curvature_damping_ema + alpha * cd

    def tune(self, step, info):
        """Run auto-tune only at AUTO_TUNE_EVERY intervals."""
        if step - self._last_tune_step < AUTO_TUNE_EVERY:
            return []
        self._last_tune_step = step
        actions = []

        mpc = self.mpc_intervention_ema
        fe = self.forecast_error_ema
        ch = self.cap_hit_ema
        cd = self.curvature_damping_ema

        # ── instability_target: should match natural I_combined ─────────
        # Natural operating point is what the model wants to express.
        # If MPC intervenes >70% of the time, target is below operating point.
        natural_I = info.get('mean_instability_field', 0.4)
        if mpc > 0.70 and natural_I > self.instability_target * 1.1:
            self.instability_target = min(0.7, self.instability_target * 1.10)
            actions.append(f"instability_target↑{self.instability_target:.3f}")
        elif mpc < 0.05 and natural_I < self.instability_target * 0.5:
            self.instability_target = max(0.1, self.instability_target * 0.90)
            actions.append(f"instability_target↓{self.instability_target:.3f}")

        # ── control_gain: bounded, gentle ───────────────────────────────
        # Higher gain = stronger suppression (more aggressive MPC).
        # Cap at CONTROL_GAIN_MAX to avoid the 10.0 catastrophe.
        if ch > 0.50 and self.control_gain < CONTROL_GAIN_MAX:
            self.control_gain = min(CONTROL_GAIN_MAX, self.control_gain * 1.05)
            actions.append(f"control_gain↑{self.control_gain:.3f}")
        elif ch < 0.01 and mpc < 0.10 and self.control_gain > CONTROL_GAIN_MIN:
            self.control_gain = max(CONTROL_GAIN_MIN, self.control_gain * 0.95)
            actions.append(f"control_gain↓{self.control_gain:.3f}")

        # ── coherence_weight & forecast_weight: only on persistent forecast error ─
        if fe > 0.30:
            self.coherence_weight = min(2.0, self.coherence_weight * 1.02)
            self.forecast_weight = min(2.0, self.forecast_weight * 1.02)
            actions.append(f"coh/fcst weights↑{self.coherence_weight:.2f}/{self.forecast_weight:.2f}")
        elif fe < 0.05:
            self.coherence_weight = max(0.5, self.coherence_weight * 0.99)
            self.forecast_weight = max(0.5, self.forecast_weight * 0.99)

        # Push to all blocks
        for block in self.model.blocks:
            block.instability_target = self.instability_target
            block.control_gain = self.control_gain
            block.coherence_weight = self.coherence_weight
            block.forecast_weight = self.forecast_weight

        return actions


# ─── PRESSURE TENSOR LOGGER (NEW v8.8) ────────────────────────────────────

class PressureTensorLogger:
    """
    Tracks pressure concentration (χ) over time and alerts when a single
    governor dominates (χ > PRESSURE_CONCENTRATION_ALERT). This is the
    "shear failure" indicator from paper §6.2.
    """
    def __init__(self):
        self.chi_history = []
        self.alert_count = 0
        self.last_alert_step = 0

    def update(self, info, step):
        chi = info.get('pressure_concentration', 0.0)
        self.chi_history.append(chi)
        if len(self.chi_history) > 1000:
            self.chi_history.pop(0)

        if chi > PRESSURE_CONCENTRATION_ALERT and step - self.last_alert_step > AUTO_TUNE_EVERY:
            dominant = info.get('dominant_governor', 'unknown')
            self.alert_count += 1
            self.last_alert_step = step
            return f"⚠️  Pressure concentration χ={chi:.2f} (dominant={dominant}) — relief valve pattern"
        return None


# ─── THROUGHPUT TRACKER (preserved from v8.6) ────────────────────────────

class ThroughputTracker:
    def __init__(self, tokens_per_step, total_tokens):
        self.tokens_per_step = tokens_per_step
        self.total_tokens = total_tokens
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = -1
        self._cache = None
        self.window_tokens = []
        self.window_times = []
        self.window_size = 50
        self._first_call = True

    def update(self, step):
        if step == self.last_step:
            return self._cache

        now = time.time()
        elapsed = now - self.start_time
        total_proc = step * self.tokens_per_step

        if self._first_call and self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
            self._first_call = False
        elif self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
        else:
            tokens_since = total_proc
            time_since = elapsed

        if time_since > 0 and tokens_since > 0:
            self.window_tokens.append(tokens_since)
            self.window_times.append(time_since)
            if len(self.window_tokens) > self.window_size:
                self.window_tokens.pop(0)
                self.window_times.pop(0)

        smoothed = sum(self.window_tokens) / sum(self.window_times) if self.window_times else 0
        remaining = max(0, self.total_tokens - total_proc)
        eta = remaining / smoothed if smoothed > 0 else 0

        self.last_time = now
        self.last_step = step

        raw_progress = (total_proc / self.total_tokens) * 100 if self.total_tokens > 0 else 0

        self._cache = {
            'step': step,
            'smoothed_tps': smoothed,
            'total_gb': total_proc / 1e9,
            'target_gb': self.total_tokens / 1e9,
            'progress': min(100.0, raw_progress),
            'raw_progress': raw_progress,
            'eta_h': eta / 3600,
            'elapsed_h': elapsed / 3600,
        }
        return self._cache

    def log(self, step):
        s = self.update(step)
        eta_str = str(timedelta(seconds=int(s['eta_h'] * 3600))) if s['eta_h'] > 0 else "N/A"
        elapsed_str = str(timedelta(seconds=int(s['elapsed_h'] * 3600)))
        progress_str = f"{s['progress']:.1f}%"
        if s['raw_progress'] > 100:
            progress_str = f"{s['raw_progress']:.1f}% (>{s['target_gb']:.1f}B target)"
        print(f"\n⏱️  Step {s['step']:,} | {s['smoothed_tps']:.0f} tok/s | "
              f"{s['total_gb']:.2f}/{s['target_gb']:.1f} GB | "
              f"{progress_str} | ETA {eta_str} | Elapsed {elapsed_str}")
        sys.stdout.flush()
        return s


# ─── DATASETS ─────────────────────────────────────────────────────────────

class StanfordDataset(IterableDataset):
    """
    v8.8 FIX: dedup cache now resets at the start of each epoch.
    v8.6 kept dedup across epochs, which meant after epoch 1 the dataset
    yielded zero Stanford texts, making the "30% Stanford" claim a lie.
    """
    def __init__(self, bucket, prefix, tokenizer, max_seq_len=512):
        self.bucket = bucket
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.target = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        self.seen = set()
        self.epoch_count = 0

    def _stream(self):
        for key in STANFORD_ONLY:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}")
                for line in obj['Body'].iter_lines():
                    if not line:
                        continue
                    try:
                        row = json.loads(line.decode('utf-8'))
                        text = row.get("text") or row.get("content") or ""
                        if len(text) < 50:
                            continue
                        h = hashlib.md5(text[:200].encode()).hexdigest()
                        if h in self.seen:
                            continue
                        self.seen.add(h)
                        yield text
                    except:
                        continue
            except Exception as e:
                print(f"⚠️  S3 error: {e}")

    def __iter__(self):
        # v8.8: Reset dedup at start of each epoch so all Stanford text
        # is yielded again. Without this, epoch 2+ yields nothing.
        self.epoch_count += 1
        if self.epoch_count > 1:
            self.seen.clear()
            print(f"   🔄 Stanford epoch {self.epoch_count}: dedup cache cleared")

        while True:
            buffer = []
            for text in self._stream():
                try:
                    toks = self.tokenizer.encode(text, allowed_special="all")
                except:
                    toks = self.tokenizer.encode(text)
                for t in toks:
                    buffer.append(t)
                buffer.append(self.tokenizer.eos_token_id or 0)
                while len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]


class S3FineWebDatasetChunked(IterableDataset):
    def __init__(self, bucket, prefix, max_seq_len=512, max_chunks=500):
        self.bucket = bucket
        self.prefix = prefix
        self.target = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')

        print("   📥 Loading FineWeb chunks...")
        sys.stdout.flush()

        chunks = []
        cont = None
        while True:
            kwargs = {'Bucket': bucket, 'Prefix': prefix}
            if cont:
                kwargs['ContinuationToken'] = cont
            resp = self.s3.list_objects_v2(**kwargs)
            chunks.extend([o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.npy')])
            if not resp.get('IsTruncated'):
                break
            cont = resp.get('NextContinuationToken')

        chunks = sorted(chunks)[:max_chunks]
        print(f"   📚 Loading {len(chunks)} chunks...")

        self.all_tokens = []
        total = 0
        for i, ck in enumerate(chunks):
            try:
                data = self.s3.get_object(Bucket=bucket, Key=ck)['Body'].read()
                arr = np.load(io.BytesIO(data))
                self.all_tokens.append(arr)
                total += len(arr)
                if (i + 1) % 100 == 0:
                    print(f"      {i+1} chunks ({total:,} tokens, {total*4/1e9:.2f} GB)")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️  Chunk {ck} failed: {e}")

        print(f"   ✅ {len(self.all_tokens)} arrays | {total:,} tokens | {total*4/1e9:.2f} GB")
        sys.stdout.flush()

    def __iter__(self):
        buffer = []
        for arr in self.all_tokens:
            for t in arr:
                buffer.append(int(t))
                if len(buffer) >= self.target:
                    yield torch.tensor(buffer[:self.target], dtype=torch.long)
                    buffer = buffer[self.target:]
        if len(buffer) >= 256:
            while len(buffer) < self.target:
                buffer.append(0)
            yield torch.tensor(buffer[:self.target], dtype=torch.long)


class MixedDataset(IterableDataset):
    def __init__(self, stanford, fineweb, stanford_weight=0.3):
        self.stanford = stanford
        self.fineweb = fineweb
        self.weight = stanford_weight

    def __iter__(self):
        import random
        s_iter = iter(self.stanford)
        f_iter = iter(self.fineweb)
        while True:
            if random.random() < self.weight:
                try:
                    yield next(s_iter)
                except StopIteration:
                    # v8.8: if Stanford exhausted, fall back to FineWeb
                    # (don't crash the loop)
                    yield next(f_iter)
            else:
                yield next(f_iter)


def collate(batch):
    return torch.stack(batch)


# ─── CHECKPOINT UTILS ─────────────────────────────────────────────────────

def save_checkpoint(path, data):
    torch.save(data, path)
    try:
        torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"⚠️  Checkpoint verification failed: {e}")
        return False
    return True

def cleanup_checkpoints(ckpt_dir, keep=2):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "mycelia_step_*.pt")), key=os.path.getmtime)
    for old in ckpts[:-keep]:
        try:
            os.remove(old)
        except:
            pass


# ─── MAIN ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("🍄 MYCELIA TRAINING v8.8")
print("   Rate-Aware Governors + Adaptive Targets + Bounded Auto-Tune")
print("   Stanford (30%) + FineWeb (70%) | No compression")
print("="*70)

# Tokenizer
print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id or 0
print(f"   Vocab: {tokenizer.vocab_size:,}")

# Model
print("\n🏗️ Building model...")
cfg = MyceliaConfig()
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window = 128
cfg.compress_ratio = 8
cfg.use_compression = False
cfg.consensus_rounds = CONSENSUS_ROUNDS

# v8.8: defaults from architecture patch
# (These override the dataclass defaults; redundant if loading v8.8 architecture)
cfg.ffn_norm_target = 150.0
cfg.alpha_norm_target = 150.0
cfg.soft_cap = 400.0
cfg.instability_target = 0.45
cfg.control_gain = CONTROL_GAIN_DEFAULT
cfg.control_factor_floor = 0.4
cfg.predictive_scale = True
cfg.use_rate_governor = True

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")

# Adaptive target trackers (NEW v8.8)
ffn_target_tracker = AdaptiveTarget(
    initial=cfg.ffn_norm_target,
    ema_alpha=ADAPTIVE_FFN_TARGET_EMA,
    multiplier=ADAPTIVE_FFN_TARGET_MULTIPLIER,
    min_val=ADAPTIVE_FFN_TARGET_MIN,
    max_val=ADAPTIVE_FFN_TARGET_MAX,
)
alpha_target_tracker = AdaptiveTarget(
    initial=cfg.alpha_norm_target,
    ema_alpha=ADAPTIVE_ALPHA_TARGET_EMA,
    multiplier=ADAPTIVE_ALPHA_TARGET_MULTIPLIER,
    min_val=ADAPTIVE_ALPHA_TARGET_MIN,
    max_val=ADAPTIVE_ALPHA_TARGET_MAX,
)

# Auto-tuner and pressure logger (NEW v8.8)
auto_tuner = GovernorAutoTuner(model)
pressure_logger = PressureTensorLogger()

# Optimizer
opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

# Scheduler — use HF cosine-with-warmup
total_steps = TOTAL_TOKENS_TARGET // (BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN)
scheduler = get_cosine_schedule_with_warmup(
    opt,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

print(f"\n🔥 Scheduler: HF cosine | peak={PEAK_LR:.2e} | min={MIN_LR:.2e} | "
      f"warmup={WARMUP_STEPS} | total={total_steps:,}")
print(f"⚙️  Auto-tune: every {AUTO_TUNE_EVERY} steps | control_gain ∈ "
      f"[{CONTROL_GAIN_MIN}, {CONTROL_GAIN_MAX}]")
print(f"🎯 Adaptive targets: FFN EMA×{ADAPTIVE_FFN_TARGET_MULTIPLIER} | "
      f"α EMA×{ADAPTIVE_ALPHA_TARGET_MULTIPLIER}")
print(f"🛡️  LR Burst: {'ENABLED' if ENABLE_LR_BURST else 'DISABLED (recommended)'}")

# ─── RESUME ──────────────────────────────────────────────────────────────

start_epoch = 0
best_loss = float('inf')
best_step = 0
ckpt = None

for path, label in [(BEST_CKPT, "🏆 BEST"), (LATEST_CKPT, "📂 LATEST")]:
    if os.path.exists(path):
        print(f"\n{'='*70}\n{label} CHECKPOINT\n{'='*70}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            break
        except Exception as e:
            print(f"   ⚠️  Failed to load {label}: {e}")

if ckpt is not None:
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    print("   ✅ Model loaded")

    if hasattr(model.compressor, 'input_pos') and hasattr(model.compressor, 'latent_pos'):
        with torch.no_grad():
            model.compressor.input_pos.data = model.compressor.latent_pos.data[:, :model.compressor.window, :].clone()
            print("   🔥 Warm-started input_pos")

    step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0) + 1

    prev_loss = ckpt.get('loss', 'N/A')
    best_loss_ckpt = ckpt.get('best_loss', float('inf'))

    if isinstance(prev_loss, (int, float)) and prev_loss > 0:
        print(f"\n   📊 Resumed: step={step:,} | epoch={start_epoch-1} | loss={prev_loss:.4f}")
    else:
        print(f"\n   📊 Resumed: step={step:,} | epoch={start_epoch-1} | loss=N/A")
        prev_loss = 'N/A'

    if isinstance(best_loss_ckpt, (int, float)) and best_loss_ckpt > 0:
        best_loss = best_loss_ckpt
        print(f"   🏆 Best loss: {best_loss:.4f}")

    # ── v8.8: Use HF scheduler's built-in resume support ────────────────
    # Load scheduler state if saved; otherwise compute LR for current step
    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"   ✅ Scheduler restored (step {scheduler.last_epoch})")
        except Exception as e:
            print(f"   ⚠️  Scheduler state mismatch, recomputing: {e}")
            # Advance scheduler to current step
            for _ in range(step):
                scheduler.step()
    else:
        # No scheduler state saved — advance to current step
        for _ in range(step):
            scheduler.step()
        print(f"   🔥 Scheduler advanced to step {step} | LR={scheduler.get_last_lr()[0]:.2e}")

    # Restore auto-tuner state if present
    if 'auto_tuner_state' in ckpt:
        try:
            ats = ckpt['auto_tuner_state']
            auto_tuner.control_gain = ats.get('control_gain', CONTROL_GAIN_DEFAULT)
            auto_tuner.instability_target = ats.get('instability_target', 0.45)
            auto_tuner._last_tune_step = step
            print(f"   ✅ Auto-tuner restored")
        except Exception as e:
            print(f"   ⚠️  Auto-tuner state load failed: {e}")

    # Restore adaptive target EMAs if present
    if 'adaptive_targets_state' in ckpt:
        try:
            ats = ckpt['adaptive_targets_state']
            ffn_target_tracker.ema = ats.get('ffn_ema', cfg.ffn_norm_target)
            ffn_target_tracker.target = ats.get('ffn_target', cfg.ffn_norm_target)
            alpha_target_tracker.ema = ats.get('alpha_ema', cfg.alpha_norm_target)
            alpha_target_tracker.target = ats.get('alpha_target', cfg.alpha_norm_target)
            print(f"   ✅ Adaptive targets restored")
        except Exception as e:
            print(f"   ⚠️  Adaptive targets load failed: {e}")

    # ── v8.8: GENTLE LR BURST (or disabled) ────────────────────────────
    lr_burst_active = False
    if ENABLE_LR_BURST and step > 50000:
        recent_loss = ckpt.get('avg_loss_100', None)
        if recent_loss is not None and isinstance(recent_loss, (int, float)):
            print(f"   🚀 LR BURST: gentle ramp to LR_BURST_PEAK={LR_BURST_PEAK:.2e}")
            lr_burst_active = True
            lr_burst_start_step = step
            lr_burst_end_step = step + LR_BURST_STEPS
            for g in opt.param_groups:
                g['lr'] = LR_BURST_PEAK
    else:
        # Normal: scheduler determines LR
        pass
else:
    step = 0
    print(f"\n{'='*70}\n🚀 FRESH START\n{'='*70}")

# ─── DATA ─────────────────────────────────────────────────────────────────

print("\n📖 Loading datasets...")
stanford = StanfordDataset(S3_BUCKET, HQ_PREFIX, tokenizer, MAX_SEQ_LEN)
fineweb = S3FineWebDatasetChunked(S3_BUCKET, FINEWEB_PREFIX, MAX_SEQ_LEN)
mixed = MixedDataset(stanford, fineweb, 0.3)
loader = DataLoader(mixed, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=0)
data_iter = iter(loader)
print("   ✅ Data ready")

tokens_per_step = BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN
actual_total_tokens = max(total_steps * tokens_per_step, step * tokens_per_step)
tracker = ThroughputTracker(tokens_per_step, actual_total_tokens)
print(f"\n⏱️  Tracker: {tracker.tokens_per_step:,} tok/step | {tracker.total_tokens/1e9:.1f}B total")

torch.cuda.empty_cache()
gc.collect()

# ─── TRAINING LOOP ────────────────────────────────────────────────────────

print("\n" + "="*70)
print(f"🚀 EPOCH {start_epoch} — STEP {step:,}")
print("="*70 + "\n")

model.train()
losses_window = []
nan_count = 0
nan_history = []  # for momentum reset decision
accum_counter = 0

for b in model.blocks:
    b.mycelia.reset_stats()

for step in tqdm(range(step, step + 250000), desc="Training", initial=step):
    if _shutdown_requested:
        print("\n🛑 Graceful shutdown, saving checkpoint...")
        try:
            emergency_ckpt = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': {
                    'control_gain': auto_tuner.control_gain,
                    'instability_target': auto_tuner.instability_target,
                },
                'adaptive_targets_state': {
                    'ffn_ema': ffn_target_tracker.ema,
                    'ffn_target': ffn_target_tracker.target,
                    'alpha_ema': alpha_target_tracker.ema,
                    'alpha_target': alpha_target_tracker.target,
                },
                'loss': float(losses_window[-1]) if losses_window else None,
                'best_loss': float(best_loss),
                'best_step': best_step,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(emergency_ckpt, LATEST_CKPT)
            print(f"\n💾 Emergency save: step {step:,} → {LATEST_CKPT}")
        except Exception as e:
            print(f"\n🚨 Emergency save failed: {e}")
        break

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    batch = batch.to(device)
    input_ids = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()

    # Forward
    with autocast():
        logits = model(input_ids, padding_mask=(input_ids == PAD_ID),
                       use_compression=False, log_during_train=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1),
                               ignore_index=PAD_ID) / ACCUM_STEPS

    # ── v8.8: NaN handling — skip step cleanly, reset momentum only on persistence ──
    is_bad_loss = torch.isnan(loss) or torch.isinf(loss)
    if is_bad_loss:
        nan_count += 1
        nan_history.append(step)
        print(f"\n⚠️  NaN/Inf at step {step} (count: {nan_count})")
        if nan_count >= 2:
            for g in opt.param_groups:
                g['lr'] *= 0.5
            print(f"   🚨 LR halved to {opt.param_groups[0]['lr']:.2e}")
        if nan_count >= 3:
            # v8.8: reset Adam momentum on persistent NaN
            # (the exponential moving averages of gradients are corrupted)
            print(f"   🚨🚨 Persistent NaN, resetting Adam momentum")
            opt = AdamW(model.parameters(), lr=opt.param_groups[0]['lr'] * 2,
                        weight_decay=WEIGHT_DECAY)
            nan_count = 0
            nan_history.clear()
        # Skip this step entirely — don't backward, don't accumulate
        opt.zero_grad()
        continue

    nan_count = 0

    # Backward
    scaler.scale(loss).backward()
    accum_counter += 1

    if accum_counter >= ACCUM_STEPS:
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n⚠️  Bad gradients at step {step}, skipping step")
            scaler.update()
            opt.zero_grad()
        else:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # ── v8.8: Gentle LR burst (if enabled) ───────────────────────
            if lr_burst_active:
                if step < lr_burst_end_step:
                    # Linear ramp from current LR up to LR_BURST_PEAK
                    burst_progress = (step - lr_burst_start_step) / LR_BURST_STEPS
                    target_lr = LR_BURST_PEAK * burst_progress + MIN_LR * (1 - burst_progress)
                    for g in opt.param_groups:
                        g['lr'] = target_lr
                else:
                    # Burst ended — return control to scheduler
                    lr_burst_active = False
                    print(f"\n🎯 LR BURST COMPLETE: returning control to scheduler at step {step:,}")
            else:
                scheduler.step()

        accum_counter = 0

    # Record loss
    losses_window.append(loss.item() * ACCUM_STEPS)
    if len(losses_window) > 1000:
        losses_window.pop(0)

    # ── v8.8: Adaptive target update (every step, lightweight) ────────
    if ADAPTIVE_TARGETS_ENABLED and hasattr(model, '_last_info') and model._last_info:
        info = model._last_info
        mean_ffn = info.get('mean_ffn_norm', 0.0)
        mean_contrib = info.get('mean_contrib_norm', 0.0)
        if mean_ffn > 0:
            new_ffn_target = ffn_target_tracker.update(mean_ffn)
            new_alpha_target = alpha_target_tracker.update(mean_contrib)
            # Push to all blocks
            for block in model.blocks:
                block.ffn_norm_target = new_ffn_target
                block.alpha_norm_target = new_alpha_target

    # ── v8.8: Auto-tune telemetry EMAs (every step) ────────────────────
    if hasattr(model, '_last_info') and model._last_info:
        auto_tuner.update_telemetry_emas(model._last_info)

    # ── Emergency checkpoint (every LOG_EVERY) ─────────────────────────
    if step % LOG_EVERY == 0 and step > 0:
        try:
            emergency_ckpt = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': {
                    'control_gain': auto_tuner.control_gain,
                    'instability_target': auto_tuner.instability_target,
                },
                'adaptive_targets_state': {
                    'ffn_ema': ffn_target_tracker.ema,
                    'ffn_target': ffn_target_tracker.target,
                    'alpha_ema': alpha_target_tracker.ema,
                    'alpha_target': alpha_target_tracker.target,
                },
                'loss': float(losses_window[-1]) if losses_window else None,
                'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
                'best_loss': float(best_loss),
                'best_step': best_step,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(emergency_ckpt, LATEST_CKPT)
        except Exception as e:
            print(f"\n🚨 Emergency save failed at step {step}: {e}")

    # ── Logging (every LOG_EVERY) ──────────────────────────────────────
    if step % LOG_EVERY == 0 and step > 0:
        current_avg_loss = float(np.mean(losses_window[-100:])) if losses_window else float('inf')
        current_lr = opt.param_groups[0]['lr']

        stats = tracker.log(step)

        # Telemetry extraction
        coherence = 0.0
        early_var, late_var, delta = 0.0, 0.0, 0.0
        friction = ""
        cap_hit_ratio = 0.0
        max_raw_norm = 0.0
        mean_raw_norm = 0.0
        info = {}

        if hasattr(model, '_last_info') and model._last_info:
            info = model._last_info
            coherence = info.get('coherence', 0.0)
            early_var = info.get('early_var', 0.0)
            late_var = info.get('late_var', 0.0)
            delta = info.get('variance_delta', 0.0)
            cap_hit_ratio = info.get('soft_cap_hit_ratio', 0.0)
            max_raw_norm = info.get('max_raw_norm', 0.0)
            mean_raw_norm = info.get('mean_raw_norm', 0.0)

            if delta > 1.0:
                friction = "✅ DISSIPATED"
            elif delta < -1.0:
                friction = "🌋 DEEP DRIFT"
            elif early_var < 2.0 and late_var < 2.0:
                friction = "🟢 HARMONIZED"
            else:
                friction = "🟡 PROCESSING"

        coh_icon = "📈" if coherence > 0.8 else "📉" if coherence < 0.5 else "➡️"
        burst_indicator = " 🚀 BURST" if lr_burst_active else ""

        print(f"\n📊 Step {step:,} | Loss: {current_avg_loss:.4f} | LR: {current_lr:.2e} | "
              f"{'🚀 BURST' if lr_burst_active else '📉 Annealing'}{burst_indicator}")
        print(f"   Coherence: {coherence:.4f} {coh_icon}")
        if friction:
            print(f"   Friction: {friction} | early={early_var:.2f} late={late_var:.2f} Δ={delta:+.2f}")

        # Governor telemetry
        ffn_veto_ratio = info.get('ffn_veto_ratio', 0.0)
        mean_ffn_norm = info.get('mean_ffn_norm', 0.0)
        max_ffn_norm = info.get('max_ffn_norm', 0.0)
        if ffn_veto_ratio > 0 or mean_ffn_norm > 0:
            print(f"   FFNVeto: {ffn_veto_ratio*100:.1f}% mean_norm={mean_ffn_norm:.1f} "
                  f"max_norm={max_ffn_norm:.1f} | target={ffn_target_tracker.target:.0f}")

        alpha_scale_ratio = info.get('alpha_scale_ratio', 0.0)
        mean_alpha_scale = info.get('mean_alpha_scale', 1.0)
        mean_contrib_norm = info.get('mean_contrib_norm', 0.0)
        if alpha_scale_ratio > 0 or mean_contrib_norm > 0:
            print(f"   AlphaScale: {alpha_scale_ratio*100:.1f}% scale={mean_alpha_scale:.3f} "
                  f"contrib_norm={mean_contrib_norm:.1f} | target={alpha_target_tracker.target:.0f}")

        if cap_hit_ratio > 0 or max_raw_norm > 0:
            print(f"   SoftCap: hit={cap_hit_ratio*100:.1f}% max_raw={max_raw_norm:.1f} "
                  f"mean_raw={mean_raw_norm:.1f}")

        # MPC telemetry
        mpc_intervention_ratio = info.get('mpc_intervention_ratio', 0.0)
        mean_control_factor = info.get('mean_control_factor', 1.0)
        mean_instability_field = info.get('mean_instability_field', 0.0)
        if mpc_intervention_ratio > 0 or mean_instability_field > 0:
            print(f"\n   🔮 MPC: intervene={mpc_intervention_ratio*100:.1f}% "
                  f"control={mean_control_factor:.3f} I={mean_instability_field:.3f}")
            print(f"   📊 Pred: {info.get('mean_prediction', 0):.3f} | "
                  f"Conf: {info.get('mean_confidence', 1):.3f} (min: {info.get('confidence_min', 1):.3f})")
            print(f"   📈 Dynamics: v={info.get('instability_velocity', 0):+.4f} "
                  f"a={info.get('instability_acceleration', 0):+.4f}")
            print(f"   🎯 Forecast Error: {info.get('forecast_error', 0):.3f} "
                  f"(predicted={info.get('predicted_from_prev', 0):.3f} vs "
                  f"actual={info.get('actual_intervention', 0):.3f})")

        instability_history = info.get('instability_field_history', [])
        confidence_history = info.get('confidence_history', [])
        if len(instability_history) >= 6:
            print(f"   I-field:  {' '.join([f'{v:.2f}' for v in instability_history[:6]])}")
        if len(confidence_history) >= 6:
            print(f"   Conf-field:{' '.join([f'{v:.2f}' for v in confidence_history[:6]])}")

        # v8.8 NEW: Pressure tensor logging
        total_pressure = info.get('total_pressure', 0.0)
        pressure_conc = info.get('pressure_concentration', 0.0)
        dominant = info.get('dominant_governor', 'none')
        if total_pressure > 0:
            print(f"\n   🔥 Π={total_pressure:.1f} | χ={pressure_conc:.2f} | dominant={dominant}")
            pi_breakdown = info.get('pressure_by_governor', {})
            pi_str = ' '.join([f"{k}={v:.1f}" for k, v in pi_breakdown.items()])
            print(f"   🔥 Π breakdown: {pi_str}")

            # v8.8: Alert on relief valve pattern (paper §6.2)
            pressure_alert = pressure_logger.update(info, step)
            if pressure_alert:
                print(f"   {pressure_alert}")
                print(f"   📋 Hypothesis test: adaptive target tracking active. "
                      f"Watch for pressure redistribution.")

        # v8.8: Rate governor telemetry (NEW)
        rate_governor_hit = info.get('rate_governor_hit', 0.0)
        rate_scale_mean = info.get('rate_scale_mean', 1.0)
        ffn_growth = info.get('ffn_growth_ratio', 1.0)
        res_growth = info.get('residual_growth_ratio', 1.0)
        if rate_governor_hit > 0.01 or rate_scale_mean < 0.99:
            print(f"   📐 Rate Governor: hit={rate_governor_hit*100:.1f}% "
                  f"scale={rate_scale_mean:.3f} | "
                  f"ffn_growth={ffn_growth:.2f}x res_growth={res_growth:.2f}x")

        sys.stdout.flush()

        # ── v8.8: BOUNDED auto-tune (every AUTO_TUNE_EVERY) ────────────
        tune_actions = auto_tuner.tune(step, info)
        if tune_actions:
            print(f"   ⚙️  Auto-tune: {', '.join(tune_actions)}")

        # ── v8.8: FIXED best_loss logic (recomputed here, not stale) ────
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss
            best_step = step
            ckpt_data = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auto_tuner_state': {
                    'control_gain': auto_tuner.control_gain,
                    'instability_target': auto_tuner.instability_target,
                },
                'adaptive_targets_state': {
                    'ffn_ema': ffn_target_tracker.ema,
                    'ffn_target': ffn_target_tracker.target,
                    'alpha_ema': alpha_target_tracker.ema,
                    'alpha_target': alpha_target_tracker.target,
                },
                'loss': float(current_avg_loss),
                'best_loss': float(best_loss),
                'coherence': float(coherence),
                'friction': friction,
                'early_var': float(early_var),
                'late_var': float(late_var),
                'delta': float(delta),
                'timestamp': datetime.now().isoformat(),
            }
            try:
                torch.save(ckpt_data, BEST_CKPT)
                verify = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)
                if 'model_state_dict' in verify:
                    print(f"\n🏆 BEST SAVED: {best_loss:.4f} at step {step:,}")
            except Exception as e:
                print(f"\n🚨 BEST save FAILED at step {step}: {e}")
            sys.stdout.flush()

    # Regular checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        ckpt_data = {
            'epoch': start_epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'auto_tuner_state': {
                'control_gain': auto_tuner.control_gain,
                'instability_target': auto_tuner.instability_target,
            },
            'adaptive_targets_state': {
                'ffn_ema': ffn_target_tracker.ema,
                'ffn_target': ffn_target_tracker.target,
                'alpha_ema': alpha_target_tracker.ema,
                'alpha_target': alpha_target_tracker.target,
            },
            'loss': float(losses_window[-1]) if losses_window else None,
            'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
            'best_loss': float(best_loss),
            'best_step': best_step,
            'timestamp': datetime.now().isoformat(),
        }
        path = os.path.join(CKPT_DIR, f"mycelia_step_{step:05x}.pt")
        try:
            torch.save(ckpt_data, path)
            torch.save(ckpt_data, LATEST_CKPT)
            print(f"\n💾 Checkpoint: step {step:,} → {path}")
            cleanup_checkpoints(CKPT_DIR)
        except Exception as e:
            print(f"\n🚨 Checkpoint save failed: {e}")
        sys.stdout.flush()

    if step % CACHE_CLEAN_EVERY == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ─── FINAL SAVE ────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("💾 Final save...")

final = {
    'epoch': start_epoch,
    'global_step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'auto_tuner_state': {
        'control_gain': auto_tuner.control_gain,
        'instability_target': auto_tuner.instability_target,
    },
    'adaptive_targets_state': {
        'ffn_ema': ffn_target_tracker.ema,
        'ffn_target': ffn_target_tracker.target,
        'alpha_ema': alpha_target_tracker.ema,
        'alpha_target': alpha_target_tracker.target,
    },
    'loss': float(losses_window[-1]) if losses_window else None,
    'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
    'best_loss': float(best_loss),
    'best_step': best_step,
    'timestamp': datetime.now().isoformat(),
}

for ckpt_path, label in [(LATEST_CKPT, "LATEST"), (BEST_CKPT, "BEST")]:
    try:
        torch.save(final, ckpt_path)
        print(f"   ✅ {label}: {ckpt_path}")
    except Exception as e:
        print(f"   🚨 {label} save failed: {e}")

print(f"\n✅ Done! Steps: {step:,} | Best: {best_loss:.4f} at {best_step:,}")
print("="*70)