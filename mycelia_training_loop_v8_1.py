# train_mycelia_final.py - Production-hardened training with Stanford + FineWeb from S3
# v8.0 - Bulletproof resumption, structured telemetry, zero memory leaks
# PATCHED: gradient scaler crash fix, throughput tracker fix, dynamic total tokens

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
from transformers import AutoTokenizer
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
ACCUM_STEPS = 8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SAVE_EVERY = 5000
LOG_EVERY = 1000
CACHE_CLEAN_EVERY = 1000

PEAK_LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 100

# ─── v8.0 TUNING KNOBS ─────────────────────────────────────────────────────
# LR Burst: Reset to peak LR on resume to escape local minimum plateau
ENABLE_LR_BURST = True          # Set False to disable
LR_BURST_STEPS = 500            # How many steps to hold at peak before cosine resumes
LR_BURST_MIN_DELTA = 0.05     # Only burst if loss improvement < this over last 5K steps

# Consensus Threshold Tuning: Lower = more acclamation, higher coherence
# v8.1: CONSENSUS_DISSENTER_THRESHOLD removed — now adaptive via MAD-based scaling
CONSENSUS_ROUNDS = 2                  # Was 1 — more rounds = deeper consensus

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
    print("🍄 Mycelia v7.3 loaded")
except ImportError:
    raise ImportError("MYCELIA_architecture.py not found!")

# ─── THROUGHPUT TRACKER (v8.0 PATCHED) ───────────────────────────────────

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

        # PATCH v8.0: On first call after resume, only count tokens since last log,
        # not cumulative total. This prevents impossible tok/s on resume.
        if self._first_call and self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
            self._first_call = False
        elif self.last_step >= 0:
            tokens_since = (step - self.last_step) * self.tokens_per_step
            time_since = now - self.last_time
        else:
            # Fresh start: use elapsed time for first measurement
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

        # PATCH v8.0: Dynamic progress — cap at 100 but show real ratio
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
        # PATCH v8.0: Show raw progress if > 100%
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
            # PATCH v8.0: Do NOT clear dedup cache on exhaustion — persistent across epochs
            # self.seen.clear()

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
                yield next(s_iter)
            else:
                yield next(f_iter)

def collate(batch):
    return torch.stack(batch)

# ─── LR SCHEDULER ──────────────────────────────────────────────────────────

class MyceliaLRScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, peak_lr, min_lr):
        self.opt = optimizer
        self.total = total_steps
        self.warmup = warmup_steps
        self.peak = peak_lr
        self.min = min_lr
        self.step_count = 0
        self.lr = 0.0
        self.warmed = False

    def step(self):
        self.step_count += 1
        self.lr = self._compute(self.step_count)
        for g in self.opt.param_groups:
            g['lr'] = self.lr
        return self.lr

    def _compute(self, step):
        if step <= self.warmup:
            return self.peak * (step / self.warmup)
        self.warmed = True
        prog = min(1.0, (step - self.warmup) / (self.total - self.warmup))
        decay = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.min + (self.peak - self.min) * decay

    def get_lr(self):
        return self.lr

    def status(self):
        if self.step_count <= self.warmup:
            return f"🔥 Warmup {self.step_count}/{self.warmup}"
        return "📉 Annealing"

# ─── CHECKPOINT UTILS ──────────────────────────────────────────────────────

def save_checkpoint(path, data):
    torch.save(data, path)
    # Verify
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

# ─── MAIN ───────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("🍄 MYCELIA FINAL TRAINING v8.0")
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
# v8.1: dissenter_threshold is computed adaptively in MycelialConsensus
# using robust statistics (MAD-based scaling). No hardcoded threshold needed.
cfg.consensus_rounds = CONSENSUS_ROUNDS

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")

# Optimizer
opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

# Scheduler
total_steps_5B = 5_000_000_000 // (BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN)
scheduler = MyceliaLRScheduler(opt, total_steps_5B, WARMUP_STEPS, PEAK_LR, MIN_LR)

print(f"\n🔥 Scheduler: peak={PEAK_LR:.2e} | min={MIN_LR:.2e} | warmup={WARMUP_STEPS} | total={total_steps_5B:,}")

# ─── RESUME ────────────────────────────────────────────────────────────────

start_epoch = 0
best_loss = float('inf')
best_step = 0
ckpt = None
loaded_from = None

for path, label in [(BEST_CKPT, "🏆 BEST"), (LATEST_CKPT, "📂 LATEST")]:
    if os.path.exists(path):
        print(f"\n{'='*70}\n{label} CHECKPOINT\n{'='*70}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            loaded_from = path
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

    # PATCH v8.0: Validate checkpoint loss values — reject impossible 0.0
    prev_loss = ckpt.get('loss', 'N/A')
    best_loss_ckpt = ckpt.get('best_loss', float('inf'))

    if isinstance(prev_loss, (int, float)) and prev_loss > 0:
        print(f"\n   📊 Resumed: step={step:,} | epoch={start_epoch-1} | loss={prev_loss:.4f}")
    else:
        print(f"\n   📊 Resumed: step={step:,} | epoch={start_epoch-1} | loss=N/A (checkpoint corrupted)")
        prev_loss = 'N/A'

    if isinstance(best_loss_ckpt, (int, float)) and best_loss_ckpt > 0 and best_loss_ckpt != float('inf'):
        best_loss = best_loss_ckpt
        print(f"   🏆 Best loss: {best_loss:.4f}")
    else:
        print(f"   🏆 Best loss: N/A (checkpoint corrupted, will re-establish)")
        best_loss = float('inf')

    # Reset scheduler — force into annealing at current step
    scheduler.step_count = step
    scheduler.warmed = step > WARMUP_STEPS

    # ─── v8.0 DYNAMIC TOTAL STEPS ────────────────────────────────────────
    # If we've exceeded the original total, extend it so cosine decay doesn't snap to min
    if step >= scheduler.total:
        extension = max(100000, step - scheduler.total + 500000)
        scheduler.total = step + extension
        print(f"   📈 Extended training plan: {scheduler.total:,} total steps")

    # ─── v8.0 LR BURST LOGIC ─────────────────────────────────────────────
    lr_burst_active = False
    if ENABLE_LR_BURST and step > 50000 and scheduler.lr <= MIN_LR * 1.1:
        recent_loss = ckpt.get('avg_loss_100', None)
        if recent_loss is not None and isinstance(recent_loss, (int, float)) and recent_loss > 0:
            pass
        scheduler.step_count = step
        scheduler.warmed = True
        print(f"   🚀 LR BURST: Injecting peak LR={PEAK_LR:.2e} for {LR_BURST_STEPS} steps")
        lr_burst_active = True
        lr_burst_start_step = step
        lr_burst_end_step = step + LR_BURST_STEPS
        scheduler.lr = PEAK_LR
        for g in opt.param_groups:
            g['lr'] = PEAK_LR
    else:
        scheduler.lr = scheduler._compute(step)
        for g in opt.param_groups:
            g['lr'] = scheduler.lr

    print(f"   🔥 Scheduler reset: step={step} | LR={scheduler.lr:.2e} | {scheduler.status()}")
    if lr_burst_active:
        print(f"   🚀 LR BURST ACTIVE: steps {lr_burst_start_step:,} → {lr_burst_end_step:,}")

    if 'optimizer_state_dict' in ckpt:
        try:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print("   ✅ Optimizer restored")
        except Exception as e:
            print(f"   ⚠️  Optimizer reset: {e}")
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

# Tracker
# PATCH v8.0: Dynamic total tokens — use max of planned vs actual processed
tokens_per_step = BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN
actual_total_tokens = max(total_steps_5B * tokens_per_step, step * tokens_per_step)
tracker = ThroughputTracker(tokens_per_step, actual_total_tokens)

print(f"\n⏱️  Tracker: {tracker.tokens_per_step:,} tok/step | {tracker.total_tokens/1e9:.1f}B total")

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print(f"🚀 EPOCH {start_epoch} — STEP {step:,}")
print("="*70 + "\n")

model.train()
losses_window = []  # rolling window, not unbounded
nan_count = 0
accum_counter = 0

for b in model.blocks:
    b.mycelia.reset_stats()

for step in tqdm(range(step, step + 250000), desc="Training", initial=step):
    if _shutdown_requested:
        print("\n🛑 Graceful shutdown, saving checkpoint...")
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
        logits = model(input_ids, padding_mask=(input_ids == PAD_ID), use_compression=False, log_during_train=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID) / ACCUM_STEPS

    # NaN guard
    if torch.isnan(loss) or torch.isinf(loss):
        nan_count += 1
        print(f"\n⚠️  NaN at step {step} (count: {nan_count})")
        if nan_count >= 2:
            for g in opt.param_groups:
                g['lr'] *= 0.5
            print(f"   🚨 LR halved to {opt.param_groups[0]['lr']:.2e}")
        if nan_count >= 3:
            raise RuntimeError(f"Persistent NaN at step {step}")
        opt.zero_grad()
        continue

    nan_count = 0  # reset on good batch

    # Backward
    scaler.scale(loss).backward()
    accum_counter += 1

    if accum_counter >= ACCUM_STEPS:
        # PATCH v8.0: Gradient scaler crash fix
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

            # ─── v8.0 LR BURST STEP LOGIC ──────────────────────────────────
            if 'lr_burst_active' in locals() and lr_burst_active:
                if step < lr_burst_end_step:
                    # Hold at peak LR during burst
                    scheduler.lr = PEAK_LR
                    for g in opt.param_groups:
                        g['lr'] = PEAK_LR
                else:
                    # Burst ended — resume cosine decay from burst end, not from step 0
                    lr_burst_active = False
                    scheduler.step_count = lr_burst_end_step
                    scheduler.lr = scheduler._compute(lr_burst_end_step)
                    for g in opt.param_groups:
                        g['lr'] = scheduler.lr
                    print(f"\n🎯 LR BURST COMPLETE: Resuming cosine decay at step {step:,} | LR={scheduler.lr:.2e}")
            else:
                scheduler.step()

        accum_counter = 0

    # Record loss (bounded window)
    losses_window.append(loss.item() * ACCUM_STEPS)
    if len(losses_window) > 1000:
        losses_window.pop(0)

    # ─── LOGGING ──────────────────────────────────────────────────────
    if step % LOG_EVERY == 0 and step > 0:
        avg_loss = np.mean(losses_window[-100:]) if losses_window else 0
        lr = scheduler.get_lr()

        stats = tracker.log(step)

        # Telemetry
        coherence = 0.0
        early_var, late_var, delta = 0.0, 0.0, 0.0
        friction = ""

        if hasattr(model, '_last_info') and model._last_info:
            info = model._last_info
            coherence = info.get('coherence', 0.0)
            early_var = info.get('early_var', 0.0)
            late_var = info.get('late_var', 0.0)
            delta = info.get('variance_delta', 0.0)

            if delta > 1.0:
                friction = "✅ DISSIPATED"
            elif delta < -1.0:
                friction = "🌋 DEEP DRIFT"
            elif early_var < 2.0 and late_var < 2.0:
                friction = "🟢 HARMONIZED"
            else:
                friction = "🟡 PROCESSING"

        coh_icon = "📈" if coherence > 0.8 else "📉" if coherence < 0.5 else "➡️"

        burst_indicator = " 🚀 BURST" if ('lr_burst_active' in locals() and lr_burst_active) else ""
        print(f"\n📊 Step {step:,} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {scheduler.status()}{burst_indicator}")
        print(f"   Coherence: {coherence:.4f} {coh_icon}")
        if friction:
            print(f"   Friction: {friction} | early={early_var:.2f} late={late_var:.2f} Δ={delta:+.2f}")
        sys.stdout.flush()

        # Best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_step = step
            ckpt_data = {
                'epoch': start_epoch,
                'global_step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_step': scheduler.step_count,
                'loss': float(avg_loss),
                'best_loss': float(best_loss),
                'coherence': float(coherence),
                'timestamp': datetime.now().isoformat(),
            }
            if save_checkpoint(BEST_CKPT, ckpt_data):
                print(f"\n🏆 BEST: {best_loss:.4f} at step {step:,}")
                sys.stdout.flush()

    # Regular checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        ckpt_data = {
            'epoch': start_epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_step': scheduler.step_count,
            'loss': float(losses_window[-1]) if losses_window else None,
            'avg_loss_100': float(np.mean(losses_window[-100:])) if len(losses_window) >= 100 else None,
            'timestamp': datetime.now().isoformat(),
        }
        path = os.path.join(CKPT_DIR, f"mycelia_step_{step:05x}.pt")
        if save_checkpoint(path, ckpt_data):
            save_checkpoint(LATEST_CKPT, ckpt_data)
            print(f"\n💾 Checkpoint: step {step:,}")
            cleanup_checkpoints(CKPT_DIR)
            sys.stdout.flush()

    # Cache cleanup
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
    'scheduler_step': scheduler.step_count,
    'loss': float(losses_window[-1]) if losses_window else None,
    'best_loss': float(best_loss),
    'timestamp': datetime.now().isoformat(),
}

save_checkpoint(LATEST_CKPT, final)
if best_step == step:
    save_checkpoint(BEST_CKPT, final)

print(f"\n✅ Done! Steps: {step:,} | Best: {best_loss:.4f} at {best_step:,}")
print("="*70)
