# ============================================
# MYCELIA Training Loop — v8.8.1
# Emergency rollback + gradual transition from v8.8
#
# v8.8 FAILED because:
#   - Adaptive targets ran away (EMA * 1.5 tracked exploding norms)
#   - Rate governor too aggressive (1.3x threshold on unstable model)
#   - Multiple governors stacked to ~6% effective capacity
#   - No transition from v8.6 regime (1.5M steps at target=50)
#
# v8.8.1 FIXES:
#   1. DISABLE adaptive targets — gradual ramp instead
#   2. DISABLE rate governor for first 5K steps
#   3. RAISE control_factor_floor: 0.4 → 0.7
#   4. ADD governor interaction guard (max 2 simultaneous)
#   5. GRADUAL transition: targets ramp 50→150 over 10K steps
#   6. SMOOTH start: begin at v8.6 values, transition over time
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
ACCUM_STEPS = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 2.0
SAVE_EVERY = 5000
LOG_EVERY = 1000
CACHE_CLEAN_EVERY = 1000

PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
TOTAL_TOKENS_TARGET = 5_000_000_000

ENABLE_LR_BURST = False
CONSENSUS_ROUNDS = 2
AUTO_TUNE_EVERY = 5000

CONTROL_GAIN_MIN = 0.5
CONTROL_GAIN_MAX = 1.5
CONTROL_GAIN_DEFAULT = 1.0

# ── v8.8.1: DISABLE adaptive targets, use gradual transition ──────────────
ADAPTIVE_TARGETS_ENABLED = False  # ← DISABLED (was True in v8.8)

# ── v8.8.1: Gradual transition config ────────────────────────────────────
USE_GRADUAL_TRANSITION = True
TRANSITION_DURATION = 10000       # Ramp over 10K steps
FFN_TARGET_START = 50.0           # v8.6 value
FFN_TARGET_END = 150.0            # v8.8 target
ALPHA_TARGET_START = 100.0      # v8.6 value
ALPHA_TARGET_END = 150.0        # v8.8 target

# ── v8.8.1: Governor interaction guard ─────────────────────────────────────
MAX_SIMULTANEOUS_GOVERNORS = 2

# Pressure concentration alert
PRESSURE_CONCENTRATION_ALERT = 0.85

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"
FINEWEB_PREFIX = "fineweb_cache"
STANFORD_ONLY = ["stanford_philosophy_processed.jsonl"]

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

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
    print("🍄 Mycelia architecture loaded successfully - get ready for the ride!")
except ImportError:
    raise ImportError("MYCELIA_architecture not found!")

# ─── GOVERNOR AUTO-TUNER (v8.8.1: bounded, no adaptive targets) ─────────

class GovernorAutoTuner:
    def __init__(self, model):
        self.model = model
        self.control_gain = CONTROL_GAIN_DEFAULT
        self.instability_target = 0.45
        self.coherence_weight = 1.0
        self.forecast_weight = 1.0
        self.mpc_intervention_ema = 0.5
        self.forecast_error_ema = 0.15
        self.cap_hit_ema = 0.10
        self.curvature_damping_ema = 1.0
        self._aggressive_count = 0
        self._last_tune_step = 0

    def update_telemetry_emas(self, info, alpha=0.05):
        mpc = info.get('mpc_intervention_ratio', 0.0)
        fe = info.get('forecast_error', 0.0)
        ch = info.get('soft_cap_hit_ratio', 0.0)
        cd = info.get('curvature_damping', 1.0)
        self.mpc_intervention_ema = (1-alpha) * self.mpc_intervention_ema + alpha * mpc
        self.forecast_error_ema = (1-alpha) * self.forecast_error_ema + alpha * fe
        self.cap_hit_ema = (1-alpha) * self.cap_hit_ema + alpha * ch
        self.curvature_damping_ema = (1-alpha) * self.curvature_damping_ema + alpha * cd

    def tune(self, step, info):
        if step - self._last_tune_step < AUTO_TUNE_EVERY:
            return []
        self._last_tune_step = step
        actions = []
        mpc = self.mpc_intervention_ema
        fe = self.forecast_error_ema
        ch = self.cap_hit_ema
        cd = self.curvature_damping_ema

        if mpc > 0.70:
            self.instability_target = min(0.7, self.instability_target * 1.10)
            actions.append(f"instability_target↑{self.instability_target:.3f}")
        elif mpc < 0.05:
            self.instability_target = max(0.1, self.instability_target * 0.90)
            actions.append(f"instability_target↓{self.instability_target:.3f}")

        if ch > 0.50 and self.control_gain < CONTROL_GAIN_MAX:
            self.control_gain = min(CONTROL_GAIN_MAX, self.control_gain * 1.05)
            actions.append(f"control_gain↑{self.control_gain:.3f}")
        elif ch < 0.01 and mpc < 0.10 and self.control_gain > CONTROL_GAIN_MIN:
            self.control_gain = max(CONTROL_GAIN_MIN, self.control_gain * 0.95)
            actions.append(f"control_gain↓{self.control_gain:.3f}")

        if fe > 0.30:
            self.coherence_weight = min(2.0, self.coherence_weight * 1.02)
            self.forecast_weight = min(2.0, self.forecast_weight * 1.02)
            actions.append(f"coh/fcst weights↑{self.coherence_weight:.2f}/{self.forecast_weight:.2f}")
        elif fe < 0.05:
            self.coherence_weight = max(0.5, self.coherence_weight * 0.99)
            self.forecast_weight = max(0.5, self.forecast_weight * 0.99)

        for block in self.model.blocks:
            block.instability_target = self.instability_target
            block.control_gain = self.control_gain
            block.coherence_weight = self.coherence_weight
            block.forecast_weight = self.forecast_weight

        return actions

# ─── PRESSURE TENSOR LOGGER ──────────────────────────────────────────────

class PressureTensorLogger:
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
            return f"⚠️  Pressure concentration χ={chi:.2f} (dominant={dominant})"
        return None

# ─── THROUGHPUT TRACKER ────────────────────────────────────────────────

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
                    yield next(f_iter)
            else:
                yield next(f_iter)

def collate(batch):
    return torch.stack(batch)

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
print("🍄 MYCELIA TRAINING v8.8.1")
print("   Gradual Transition + Interaction Guard + No Adaptive Targets")
print("   Stanford (30%) + FineWeb (70%) | No compression")
print("="*70)

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id or 0
print(f"   Vocab: {tokenizer.vocab_size:,}")

print("\n🏗️ Building model...")
cfg = MyceliaConfig()
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window = 128
cfg.compress_ratio = 8
cfg.use_compression = False
cfg.consensus_rounds = CONSENSUS_ROUNDS

# v8.8.1: Start at v8.6 values, will ramp gradually
cfg.ffn_norm_target = FFN_TARGET_START      # 50.0
cfg.alpha_norm_target = ALPHA_TARGET_START  # 100.0
cfg.soft_cap = 400.0
cfg.instability_target = 0.45
cfg.control_gain = CONTROL_GAIN_DEFAULT
cfg.control_factor_floor = 0.7              # ← RAISED from 0.4
cfg.predictive_scale = True

# v8.8.1: Rate governor DISABLED initially
cfg.use_rate_governor = False               # ← DISABLED
cfg.ffn_growth_ratio_max = 2.0             # ← RAISED from 1.3
cfg.residual_growth_ratio_max = 1.5        # ← RAISED from 1.2

# v8.8.1: Gradual transition config
cfg.use_gradual_transition = True
cfg.transition_duration = TRANSITION_DURATION
cfg.ffn_target_end = FFN_TARGET_END        # 150.0
cfg.alpha_target_end = ALPHA_TARGET_END    # 150.0

# v8.8.1: Governor interaction guard
cfg.max_simultaneous_governors = MAX_SIMULTANEOUS_GOVERNORS

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")

# v8.8.1: No adaptive target trackers
auto_tuner = GovernorAutoTuner(model)
pressure_logger = PressureTensorLogger()

opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

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
print(f"📈 Gradual transition: FFN {FFN_TARGET_START:.0f}→{FFN_TARGET_END:.0f} | "
      f"α {ALPHA_TARGET_START:.0f}→{ALPHA_TARGET_END:.0f} over {TRANSITION_DURATION:,} steps")
print(f"🛡️  Rate governor: DISABLED for first {TRANSITION_DURATION//2:,} steps")
print(f"🛡️  LR Burst: {'ENABLED' if ENABLE_LR_BURST else 'DISABLED'}")

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

    # ── v8.8.1a: SCHEDULER RESURRECTION ─────────────────────────────────
    # HF get_cosine_schedule_with_warmup returns LR=0 when last_epoch >= num_training_steps.
    # On resume at step 1.59M with total=305K, the optimizer is frozen dead.
    
    scheduler_alive = False
    
    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            current_lr = scheduler.get_last_lr()[0]
            if current_lr > 0 and current_lr >= MIN_LR * 0.5:
                print(f"   ✅ Scheduler restored (step {scheduler.last_epoch}) | LR={current_lr:.2e}")
                scheduler_alive = True
            else:
                print(f"   🚨 Scheduler state loaded but LR dead: {current_lr:.2e} (threshold: {MIN_LR * 0.5:.2e})")
        except Exception as e:
            print(f"   ⚠️  Scheduler state mismatch: {e}")
    
    if not scheduler_alive:
        new_total = max(step + 1_000_000, total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=new_total,
        )
        for _ in range(step):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"   🔥 Scheduler resurrected: total={new_total:,} | step={scheduler.last_epoch} | LR={current_lr:.2e}")
    
    if scheduler.get_last_lr()[0] <= 0:
        raise RuntimeError(f"CRITICAL: Scheduler resurrection failed. LR={scheduler.get_last_lr()[0]:.2e} at step {step}.")

    # v8.8.1: Set transition start to current step
    for block in model.blocks:
        block.config.transition_start_step = step
    print(f"   📈 Gradual transition starts at step {step:,}")

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

    # ── v8.8.1a: Restore adaptive target EMAs (defensive) ──────────────
    # Only restore if trackers already exist in this scope. If they are defined
    # later in the script, they will start from their constructor defaults.
    if 'adaptive_targets_state' in ckpt:
        if 'ffn_target_tracker' in globals() and 'alpha_target_tracker' in globals():
            try:
                ats = ckpt['adaptive_targets_state']
                ffn_target_tracker.ema = ats.get('ffn_ema', cfg.ffn_norm_target)
                ffn_target_tracker.target = ats.get('ffn_target', cfg.ffn_norm_target)
                alpha_target_tracker.ema = ats.get('alpha_ema', cfg.alpha_norm_target)
                alpha_target_tracker.target = ats.get('alpha_target', cfg.alpha_norm_target)
                print(f"   ✅ Adaptive targets restored")
            except Exception as e:
                print(f"   ⚠️  Adaptive targets load failed: {e}")
        else:
            print(f"   ⏭️  Adaptive targets state in checkpoint but trackers not yet defined (will use defaults)")

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
        pass
else:
    step = 0
    for block in model.blocks:
        block.config.transition_start_step = 0
    print(f"\n{'='*70}\n🚀 FRESH START\n{'='*70}")

# ─── DATA

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

    # v8.8.1: Compute gradual transition targets with cosine smoothstep
    if USE_GRADUAL_TRANSITION:
        progress = min(1.0, (step - cfg.transition_start_step) / TRANSITION_DURATION)
        # Cosine smoothstep: 0.5 - 0.5*cos(π*progress)
        # Zero derivative at both endpoints (gentle start/stop)
        ease = 0.5 - 0.5 * math.cos(math.pi * progress)
        current_ffn_target = FFN_TARGET_START + ease * (FFN_TARGET_END - FFN_TARGET_START)
        current_alpha_target = ALPHA_TARGET_START + ease * (ALPHA_TARGET_END - ALPHA_TARGET_START)
        # v8.8.1: Enable rate governor only after 50% transition progress
        use_rate = progress > 0.5
        for block in model.blocks:
            block.ffn_norm_target = current_ffn_target
            block.alpha_norm_target = current_alpha_target
            block.use_rate_governor = use_rate
    else:
        current_ffn_target = cfg.ffn_norm_target
        current_alpha_target = cfg.alpha_norm_target
        use_rate = cfg.use_rate_governor

    with autocast():
        logits = model(input_ids, padding_mask=(input_ids == PAD_ID),
                       use_compression=False, log_during_train=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1),
                               ignore_index=PAD_ID) / ACCUM_STEPS

    is_bad_loss = torch.isnan(loss) or torch.isinf(loss)
    if is_bad_loss:
        nan_count += 1
        print(f"\n⚠️  NaN/Inf at step {step} (count: {nan_count})")
        if nan_count >= 2:
            for g in opt.param_groups:
                g['lr'] *= 0.5
            print(f"   🚨 LR halved to {opt.param_groups[0]['lr']:.2e}")
        if nan_count >= 3:
            print(f"   🚨🚨 Persistent NaN, resetting Adam momentum")
            opt = AdamW(model.parameters(), lr=opt.param_groups[0]['lr'] * 2,
                        weight_decay=WEIGHT_DECAY)
            nan_count = 0
        opt.zero_grad()
        continue

    nan_count = 0
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
            scheduler.step()
        accum_counter = 0

    losses_window.append(loss.item() * ACCUM_STEPS)
    if len(losses_window) > 1000:
        losses_window.pop(0)

    # v8.8.1: Auto-tune telemetry EMAs
    if hasattr(model, '_last_info') and model._last_info:
        auto_tuner.update_telemetry_emas(model._last_info)

    # Emergency checkpoint
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
                'loss': float(losses_window[-1]) if losses_window else None,
                'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
                'best_loss': float(best_loss),
                'best_step': best_step,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(emergency_ckpt, LATEST_CKPT)
        except Exception as e:
            print(f"\n🚨 Emergency save failed at step {step}: {e}")

    # Logging
    if step % LOG_EVERY == 0 and step > 0:
        current_avg_loss = float(np.mean(losses_window[-100:])) if losses_window else float('inf')
        current_lr = opt.param_groups[0]['lr']
        stats = tracker.log(step)

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

        # v8.8.1: Show transition progress
        if USE_GRADUAL_TRANSITION:
            progress = min(1.0, (step - cfg.transition_start_step) / TRANSITION_DURATION)
            transition_status = f" | 📈 Transition: {progress*100:.0f}% (FFN={current_ffn_target:.0f} α={current_alpha_target:.0f})"
        else:
            transition_status = ""

        print(f"\n📊 Step {step:,} | Loss: {current_avg_loss:.4f} | LR: {current_lr:.2e} | 📉 Annealing{transition_status}")
        print(f"   Coherence: {coherence:.4f} {coh_icon}")
        if friction:
            print(f"   Friction: {friction} | early={early_var:.2f} late={late_var:.2f} Δ={delta:+.2f}")

        ffn_veto_ratio = info.get('ffn_veto_ratio', 0.0)
        mean_ffn_norm = info.get('mean_ffn_norm', 0.0)
        max_ffn_norm = info.get('max_ffn_norm', 0.0)
        if ffn_veto_ratio > 0 or mean_ffn_norm > 0:
            print(f"   FFNVeto: {ffn_veto_ratio*100:.1f}% mean_norm={mean_ffn_norm:.1f} "
                  f"max_norm={max_ffn_norm:.1f} | target={current_ffn_target:.0f}")

        alpha_scale_ratio = info.get('alpha_scale_ratio', 0.0)
        mean_alpha_scale = info.get('mean_alpha_scale', 1.0)
        mean_contrib_norm = info.get('mean_contrib_norm', 0.0)
        if alpha_scale_ratio > 0 or mean_contrib_norm > 0:
            print(f"   AlphaScale: {alpha_scale_ratio*100:.1f}% scale={mean_alpha_scale:.3f} "
                  f"contrib_norm={mean_contrib_norm:.1f} | target={current_alpha_target:.0f}")

        if cap_hit_ratio > 0 or max_raw_norm > 0:
            print(f"   SoftCap: hit={cap_hit_ratio*100:.1f}% max_raw={max_raw_norm:.1f} "
                  f"mean_raw={mean_raw_norm:.1f}")

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

        total_pressure = info.get('total_pressure', 0.0)
        pressure_conc = info.get('pressure_concentration', 0.0)
        dominant = info.get('dominant_governor', 'none')
        if total_pressure > 0:
            print(f"\n   🔥 Π={total_pressure:.1f} | χ={pressure_conc:.2f} | dominant={dominant}")
            pi_breakdown = info.get('pressure_by_governor', {})
            pi_str = ' '.join([f"{k}={v:.1f}" for k, v in pi_breakdown.items()])
            print(f"   🔥 Π breakdown: {pi_str}")
            pressure_alert = pressure_logger.update(info, step)
            if pressure_alert:
                print(f"   {pressure_alert}")

        # v8.8.1: Rate governor status
        rate_governor_hit = info.get('rate_governor_hit', 0.0)
        rate_scale_mean = info.get('rate_scale_mean', 1.0)
        ffn_growth = info.get('ffn_growth_ratio', 1.0)
        res_growth = info.get('residual_growth_ratio', 1.0)
        if use_rate and (rate_governor_hit > 0.01 or rate_scale_mean < 0.99):
            print(f"   📐 Rate Governor: hit={rate_governor_hit*100:.1f}% "
                  f"scale={rate_scale_mean:.3f} | "
                  f"ffn_growth={ffn_growth:.2f}x res_growth={res_growth:.2f}x")
        elif not use_rate:
            print(f"   📐 Rate Governor: DISABLED (transition progress < 50%)")

        sys.stdout.flush()

        # v8.8.1: Governor interaction guard telemetry
        active_govs = sum([
            1 if ffn_veto_ratio > 0.5 else 0,
            1 if alpha_scale_ratio > 0.5 else 0,
            1 if cap_hit_ratio > 0.5 else 0,
            1 if mpc_intervention_ratio > 0.5 else 0,
            1 if (use_rate and rate_governor_hit > 0.5) else 0,
        ])
        if active_govs > MAX_SIMULTANEOUS_GOVERNORS:
            print(f"   ⚠️  GOVERNOR INTERACTION GUARD: {active_govs} governors active, "
                  f"limit is {MAX_SIMULTANEOUS_GOVERNORS}")

        # v8.8.1: Post-transition stress test logging
        if USE_GRADUAL_TRANSITION and progress >= 1.0:
            if not hasattr(model, '_transition_complete_logged'):
                model._transition_complete_logged = True
                print(f"\n{'='*70}")
                print(f"✅ TRANSITION COMPLETE at step {step:,}")
                print(f"   Loss at completion: {current_avg_loss:.4f}")
                print(f"   v8.6 baseline best: 4.1132")
                print(f"   Δ from baseline: {current_avg_loss - 4.1132:+.4f}")
                print(f"   FFN target: {current_ffn_target:.0f} | α target: {current_alpha_target:.0f}")
                print(f"   Pressure χ: {pressure_conc:.2f} | Dominant: {dominant}")
                print(f"   Interpretation:")
                if current_avg_loss < 4.3:
                    print(f"   🟢 REGIME CHANGE HELPED: Loss broke through plateau")
                elif current_avg_loss < 4.6:
                    print(f"   🟡 REGIME CHANGE NEUTRAL: Loss stable, no breakthrough")
                else:
                    print(f"   🔴 REGIME CHANGE HURT: Loss degraded, rollback recommended")
                print(f"{'='*70}\n")
                sys.stdout.flush()

        tune_actions = auto_tuner.tune(step, info)
        if tune_actions:
            print(f"   ⚙️  Auto-tune: {', '.join(tune_actions)}")

        # Best checkpoint
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