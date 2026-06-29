#!/usr/bin/env python
# train_mycelia_final.py - Full training with Stanford + FineWeb from S3

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import boto3
import io
import sys
import gc
import json
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ─── IMPORT ARCHITECTURE
try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("🍄 Mycelia v7.2 loaded")
except ImportError:
    raise ImportError("MYCELIA_architecture.py not found!")

# ═══════════════════════════════════════════════════════════════════════════
# ─── CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MAX_SEQ_LEN = 512
BATCH_SIZE = 2
ACCUM_STEPS = 8
LR = 3e-4  # ← BUMPED to 3e-4 (was 5e-5)
WEIGHT_DECAY = 0.01
MAX_STEPS = 250000 # 4 x as many steps (instead of 50.000)
GRAD_CLIP = 1      # ← INCREASE from 0.5 to 1.0 for higher LR
SAVE_EVERY = 5000
LOG_EVERY = 1000

# ─── LR SCHEDULER PARAMETERS

PEAK_LR = 3e-4      # ← Peak learning rate
MIN_LR = 1e-5       # ← Minimum (keep as is)
WARMUP_STEPS = 100  # ← Keep 100 for smooth transition

# ─── DATA SOURCES

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"
FINEWEB_PREFIX = "fineweb_cache"

STANFORD_ONLY = [
    "stanford_philosophy_processed.jsonl",
]

# ─── DIRECTORIES

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

# ═══════════════════════════════════════════════════════════════════════════
# ─── THROUGHPUT TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class ThroughputTracker:
    def __init__(self, global_batch_size=16, seq_len=512, total_tokens=5_000_000_000):
        self.tokens_per_step = global_batch_size * seq_len
        self.total_tokens = total_tokens
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.step_counter = 0
        self.last_step = 0
        self.recent_tokens = []
        self.recent_times = []
        self.window_size = 100
    
    def update(self, current_step):
        self.step_counter += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        total_tokens_processed = self.step_counter * self.tokens_per_step
        overall_tps = total_tokens_processed / elapsed if elapsed > 0 else 0
        
        tokens_since_last = (current_step - self.last_step) * self.tokens_per_step
        time_since_last = current_time - self.last_update_time
        
        if time_since_last > 0:
            self.recent_tokens.append(tokens_since_last)
            self.recent_times.append(time_since_last)
            if len(self.recent_tokens) > self.window_size:
                self.recent_tokens.pop(0)
                self.recent_times.pop(0)
        
        if self.recent_tokens and self.recent_times:
            smoothed_tps = sum(self.recent_tokens) / sum(self.recent_times)
        else:
            smoothed_tps = overall_tps
        
        remaining_tokens = max(0, self.total_tokens - (current_step * self.tokens_per_step))
        eta_seconds = remaining_tokens / smoothed_tps if smoothed_tps > 0 else 0
        
        self.last_update_time = current_time
        self.last_step = current_step
        
        return {
            'step': current_step,
            'overall_tps': overall_tps,
            'smoothed_tps': smoothed_tps,
            'total_tokens': self.total_tokens,  # ← FIXED: Added this line
            'total_tokens_processed': total_tokens_processed,
            'tokens_processed_gb': total_tokens_processed / 1_000_000_000,
            'remaining_tokens': remaining_tokens,
            'remaining_tokens_gb': remaining_tokens / 1_000_000_000,
            'eta_seconds': eta_seconds,
            'eta_hours': eta_seconds / 3600,
            'eta_days': eta_seconds / 86400,
            'elapsed_hours': elapsed / 3600,
            'elapsed_days': elapsed / 86400,
            'progress_pct': (current_step * self.tokens_per_step / self.total_tokens) * 100,
        }
    
    def print_status(self, current_step):
        stats = self.update(current_step)
        eta_str = str(timedelta(seconds=int(stats['eta_seconds']))) if stats['eta_seconds'] > 0 else "N/A"
        elapsed_str = str(timedelta(seconds=int(stats['elapsed_hours'] * 3600)))
        
        print(f"\n⏱️ THROUGHPUT:")
        print(f"   Step: {stats['step']:,}")
        print(f"   Speed: {stats['smoothed_tps']:.0f} tok/s")
        print(f"   Processed: {stats['tokens_processed_gb']:.2f} GB / {stats['total_tokens']/1_000_000_000:.1f} GB")
        print(f"   Progress: {stats['progress_pct']:.1f}%")
        print(f"   ETA: {eta_str}")
        print(f"   Elapsed: {elapsed_str}")
        return stats

# ═══════════════════════════════════════════════════════════════════════════
# ─── DATASET: STANFORD (S3 JSONL)
# ═══════════════════════════════════════════════════════════════════════════

class StanfordDataset(IterableDataset):
    """Stream Stanford philosophy data from S3 JSONL."""
    
    def __init__(self, bucket: str, prefix: str, tokenizer, max_seq_len: int = 512):
        self.bucket = bucket
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_tokens = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        self.seen_hashes = set()
    
    def _stream_file(self, key: str):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            for line in obj['Body'].iter_lines():
                if line:
                    try:
                        yield json.loads(line.decode('utf-8'))
                    except:
                        pass
        except Exception as e:
            print(f"   ⚠️ Error reading {key}: {e}")
    
    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()
    
    def _stream_texts(self):
        for key in STANFORD_ONLY:
            for row in self._stream_file(f"{self.prefix}/{key}"):
                text = row.get("text") or row.get("content") or ""
                if not text or len(text) < 50:
                    continue
                h = self._get_hash(text)
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)
                yield text
    
    def _tokenize_stream(self):
        for text in self._stream_texts():
            try:
                tokens = self.tokenizer.encode(text, allowed_special="all")
            except:
                tokens = self.tokenizer.encode(text)
            for tok in tokens:
                yield tok
            yield self.tokenizer.eos_token_id or 0
    
    def __iter__(self):
        import random
        while True:  # ← Loop forever!
            buffer = []
            for tok in self._tokenize_stream():
                buffer.append(tok)
                while len(buffer) >= self.target_tokens:
                    seq = buffer[:self.target_tokens]
                    buffer = buffer[self.target_tokens:]
                    yield torch.tensor(seq, dtype=torch.long)

            # If we run out of data, restart the stream
            # This makes the dataset infinite
            self.seen_hashes.clear()  # Reset dedup cache
            continue

# ═══════════════════════════════════════════════════════════════════════════
# ─── DATASET: FINEWEB (S3 NPY CHUNKS)
# ═══════════════════════════════════════════════════════════════════════════

class S3FineWebDatasetChunked(IterableDataset):
    """Pre-load ALL FineWeb chunks into memory once, then train."""

    def __init__(self, bucket: str, prefix: str, tokenizer, max_seq_len: int = 512):
        self.bucket = bucket
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_tokens = max_seq_len + 1
        self.s3 = boto3.client('s3', region_name='eu-central-1')

        print("   📥 Loading ALL FineWeb chunks into memory...")
        sys.stdout.flush()

        # List all chunks
        chunks = []
        continuation_token = None
        
        while True:
            if continuation_token:
                resp = self.s3.list_objects_v2(
                    Bucket=self.bucket, 
                    Prefix=self.prefix, 
                    ContinuationToken=continuation_token
                )
            else:
                resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            
            for obj in resp.get('Contents', []):
                if obj['Key'].endswith('.npy'):
                    chunks.append(obj['Key'])
            
            if not resp.get('IsTruncated', False):
                break
            continuation_token = resp.get('NextContinuationToken')
        
        chunks = sorted(chunks)
        print(f"   📚 Found {len(chunks)} total chunks")
        sys.stdout.flush()
        
        # Load ALL chunks into memory as NumPy arrays (NOT Python lists!)
        import io
        import numpy as np
        
        self.all_tokens = []
        total_tokens = 0
        
        # Only load FIRST 500 chunks to avoid OOM
        # Change this to len(chunks) if you have enough memory
        max_chunks = min(500, len(chunks))
        chunks_to_load = chunks[:max_chunks]
        
        print(f"   📚 Loading {len(chunks_to_load)} chunks into memory...")
        sys.stdout.flush()
        
        for i, chunk_key in enumerate(chunks_to_load):
            resp = self.s3.get_object(Bucket=self.bucket, Key=chunk_key)
            data = resp['Body'].read()
            buffer = io.BytesIO(data)
            chunk_tokens = np.load(buffer)  # NumPy array
            self.all_tokens.append(chunk_tokens)  # Store as NumPy array
            total_tokens += len(chunk_tokens)
            
            if (i + 1) % 100 == 0:
                mem_gb = total_tokens * 4 / (1024**3)
                print(f"      Loaded {i+1} chunks ({total_tokens:,} tokens, {mem_gb:.2f} GB)")
                sys.stdout.flush()
        
        print(f"   ✅ Loaded {len(self.all_tokens)} NumPy arrays ({total_tokens:,} tokens)")
        print(f"   📊 Memory used: {total_tokens * 4 / (1024**3):.2f} GB")
        sys.stdout.flush()
    
    def __iter__(self):
        buffer = []
        # Iterate through NumPy arrays (fast, no S3 requests!)
        for chunk_array in self.all_tokens:
            for tok in chunk_array:
                buffer.append(int(tok))
                while len(buffer) >= self.target_tokens:
                    seq = buffer[:self.target_tokens]
                    buffer = buffer[self.target_tokens:]
                    yield torch.tensor(seq, dtype=torch.long)
        
        if len(buffer) >= 256:
            while len(buffer) < self.target_tokens:
                buffer.append(0)
            yield torch.tensor(buffer[:self.target_tokens], dtype=torch.long)
            
            del batch_tokens
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"      ✅ Batch complete, memory cleared")
            sys.stdout.flush()
        
        print(f"\n   ✅ All {len(self.chunks)} chunks processed!")

# ═══════════════════════════════════════════════════════════════════════════
# ─── DATASET: MIXED (Stanford + FineWeb)
# ═══════════════════════════════════════════════════════════════════════════

class MixedDataset(IterableDataset):
    def __init__(self, stanford_dataset, fineweb_dataset, stanford_weight: float = 0.3):
        self.stanford = stanford_dataset
        self.fineweb = fineweb_dataset
        self.stanford_weight = stanford_weight
    
    def __iter__(self):
        stanford_iter = iter(self.stanford)
        fineweb_iter = iter(self.fineweb)
        import random
        while True:
            if random.random() < self.stanford_weight:
                try:
                    yield next(stanford_iter)
                except StopIteration:
                    stanford_iter = iter(self.stanford)
                    yield next(stanford_iter)
            else:
                try:
                    yield next(fineweb_iter)
                except StopIteration:
                    fineweb_iter = iter(self.fineweb)
                    yield next(fineweb_iter)

def collate(batch):
    return torch.stack(batch)

# ═══════════════════════════════════════════════════════════════════════════
# ─── CLEANUP FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_old_checkpoints(ckpt_dir, keep=2, verbose=True):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "mycelia_step_*.pt")), key=os.path.getmtime)
    for old in ckpts[:-keep]:
        try:
            os.remove(old)
            if verbose:
                print(f"   🗑️ Removed: {os.path.basename(old)}")
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# ─── MODEL SETUP
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🍄 MYCELIA FINAL TRAINING")
print("   Stanford (30%) + FineWeb (70%)")
print(f"   Model: 181M params")
print("="*70)

# ─── TOKENIZER

print("\n📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id or 0
print(f"   Vocab: {tokenizer.vocab_size:,}")

# ─── MODEL

print("\n🏗️ Building model...")
cfg = MyceliaConfig()
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window = 128
cfg.compress_ratio = 8
cfg.use_compression = False

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
print(f"   {sum(p.numel() for p in model.parameters()):,} params on {device}")

# ─── OPTIMIZER

opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

# ─── CUSTOM LR SCHEDULER ──────────────────────────────────────────────────

class MyceliaLRScheduler:
    def __init__(self, optimizer, total_steps=610351, warmup_steps=100, 
                 peak_lr=3e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0
        self.current_lr = 0.0
        self.warmup_start_step = 0
        self._warmed_up = False  # ← ADD THIS
    
    def reset_warmup(self):
        """Reset warmup for resumed training."""
        self.warmup_start_step = self.current_step
        self._warmed_up = False  # ← RESET ON RESUME
        print(f"   🔥 Warmup reset at step {self.warmup_start_step}")
        print(f"   Warmup will run for {self.warmup_steps} steps")
    
    def step(self):
        self.current_step += 1
        self.current_lr = self._get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        return self.current_lr
    
    def _get_lr(self, step):
        steps_since_warmup = step - self.warmup_start_step
        
        if steps_since_warmup < self.warmup_steps and not self._warmed_up:
            # Linear warmup: 0 → peak_lr
            progress = steps_since_warmup / self.warmup_steps
            return self.peak_lr * progress
        
        self._warmed_up = True  # ← Mark as warmed up
        
        # Cosine decay: peak_lr → min_lr
        decay_step = max(0, steps_since_warmup - self.warmup_steps)
        total_decay_steps = self.total_steps - self.warmup_steps
        progress = min(1.0, decay_step / total_decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
    
    def get_lr(self):
        return self.current_lr
    
    def get_warmup_status(self):
        steps_since_warmup = self.current_step - self.warmup_start_step
        if steps_since_warmup < self.warmup_steps and not self._warmed_up:
            return f"🔥 Warmup {steps_since_warmup+1}/{self.warmup_steps}"
        else:
            return f"📉 Annealing"

# Calculate total steps for 5B tokens
total_steps_for_5B = 5_000_000_000 // (BATCH_SIZE * ACCUM_STEPS * MAX_SEQ_LEN)

lr_scheduler = MyceliaLRScheduler(
    optimizer=opt,
    total_steps=total_steps_for_5B,
    warmup_steps=100,
    peak_lr=3e-4,
    min_lr=1e-5
)

print(f"\n🔥 LR Scheduler initialized:")
print(f"   Peak LR: {lr_scheduler.peak_lr:.2e}")
print(f"   Min LR: {lr_scheduler.min_lr:.2e}")
print(f"   Warmup steps: {lr_scheduler.warmup_steps}")
print(f"   Total steps: {lr_scheduler.total_steps:,}")

# ═══════════════════════════════════════════════════════════════════════════
# ─── CHECKPOINT RESUMPTION
# ═══════════════════════════════════════════════════════════════════════════

global_step = 0
start_epoch = 0
best_loss = float('inf')

# ─── CHECK FOR BEST CHECKPOINT FIRST ──────────────────────────────────────

BEST_CKPT = os.path.join(CKPT_DIR, "mycelia_best.pt")

if os.path.exists(BEST_CKPT):
    print("\n" + "="*70)
    print("🏆 LOADING BEST CHECKPOINT")
    print("="*70)
    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    loaded_from_best = True
elif os.path.exists(LATEST_CKPT):
    print("\n" + "="*70)
    print("📂 LOADING LATEST CHECKPOINT")
    print("="*70)
    ckpt = torch.load(LATEST_CKPT, map_location='cpu')
    loaded_from_best = False
else:
    print("\n" + "="*70)
    print("🚀 NO CHECKPOINT — STARTING FRESH")
    print("="*70)
    ckpt = None
    loaded_from_best = False

if ckpt is not None:
    # ─── LOAD MODEL WEIGHTS ──────────────────────────────────────────────
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    print(f"   ✅ Model weights loaded (strict=False)")
    
    # ─── WARM-START input_pos ──────────────────────────────────────────────
    if hasattr(model.compressor, 'input_pos') and hasattr(model.compressor, 'latent_pos'):
        with torch.no_grad():
            latent = model.compressor.latent_pos.data
            window = model.compressor.window
            if latent.shape[1] >= window:
                model.compressor.input_pos.data = latent[:, :window, :].clone()
                print(f"   🔥 Warm-started input_pos")
    
    # ─── RESTORE TRAINING STATE ──────────────────────────────────────────
    global_step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    prev_loss = ckpt.get('loss', 'N/A')
    
    print(f"\n   📊 CHECKPOINT DATA RESTORED:")
    print(f"      Global Step:  {global_step:,}")
    print(f"      Epoch:        {start_epoch}")
    print(f"      Loss:         {prev_loss:.4f}" if prev_loss != 'N/A' else f"      Loss:         {prev_loss}")
    print(f"      Best Loss:    {best_loss:.4f}" if best_loss != float('inf') else f"      Best Loss:    N/A")
    if loaded_from_best:
        print(f"      Source:       🏆 BEST CHECKPOINT")
    else:
        print(f"      Source:       📂 LATEST CHECKPOINT")
    
    # ─── RESET WARMUP ──────────────────────────────────────────────────────
    print(f"\n🔥 Resetting warmup for resumed training...")
    lr_scheduler = MyceliaLRScheduler(
        optimizer=opt,
        total_steps=total_steps_for_5B,
        warmup_steps=100,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR
    )
    # Set current step so warmup starts from here
    lr_scheduler.current_step = ckpt.get('lr_scheduler_step', global_step)
    lr_scheduler.warmup_start_step = global_step
    print(f"   Warmup will start from step {global_step}")
    
    # ─── RESTORE OPTIMIZER STATE ──────────────────────────────────────────
    if 'optimizer_state_dict' in ckpt:
        try:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"   ✅ Optimizer state restored")
        except:
            print(f"   ⚠️ Optimizer state incompatible, reinitializing")
    
    # ─── INCREMENT EPOCH ──────────────────────────────────────────────────
    start_epoch += 1
    
    print("="*70)
else:
    # ─── FRESH START ──────────────────────────────────────────────────────
    print(f"\n   Starting from step 0, epoch 0")
    print("="*70)

# ─── DATASET

print("\n📖 Loading datasets...")
print("   📚 Stanford Philosophy (S3 JSONL)...")
stanford_dataset = StanfordDataset(
    bucket=S3_BUCKET,
    prefix=HQ_PREFIX,
    tokenizer=tokenizer,
    max_seq_len=MAX_SEQ_LEN
)

print("   📚 FineWeb-Edu (S3 NPY chunks)...")
fineweb_dataset = S3FineWebDatasetChunked(
    bucket=S3_BUCKET,
    prefix=FINEWEB_PREFIX,
    tokenizer=tokenizer,
    max_seq_len=MAX_SEQ_LEN
)

print("   🔀 Mixed dataset: 30% Stanford, 70% FineWeb")
mixed_dataset = MixedDataset(
    stanford_dataset=stanford_dataset,
    fineweb_dataset=fineweb_dataset,
    stanford_weight=0.3
)

loader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=0)
data_iter = iter(loader)

print(f"\n   ✅ Data ready!")

# ─── THROUGHPUT TRACKER

tracker = ThroughputTracker(
    global_batch_size=BATCH_SIZE * ACCUM_STEPS,
    seq_len=MAX_SEQ_LEN,
    total_tokens=5_000_000_000
)

print(f"\n⏱️ Throughput tracker initialized:")
print(f"   Tokens per step: {tracker.tokens_per_step:,}")
print(f"   Total dataset: {tracker.total_tokens / 1_000_000_000:.1f}B tokens")

# ═══════════════════════════════════════════════════════════════════════════
# ─── TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"🚀 EPOCH {start_epoch} — STEP {global_step:,}")
print("   Logging every 1000 steps | No compression")
print("   FineWeb 70% | Stanford 30%")
print("="*70 + "\n")

model.train()
losses = []
nan_count = 0

# ─── ADD BEST LOSS TRACKING HERE ──────────────────────────────────────────
best_loss = float('inf')
best_step = 0

for b in model.blocks:
    b.mycelia.reset_stats()

for step in tqdm(range(MAX_STEPS), desc=f"Training", initial=global_step):
    try:
        batch = next(data_iter)
    except StopIteration:
        print("\n🔄 Dataset exhausted, resetting...")
        data_iter = iter(loader)
        batch = next(data_iter)

    batch = batch.to(device)
    input_ids = batch[:, :-1].contiguous()
    target_ids = batch[:, 1:].contiguous()

    # ─── FORWARD (NO COMPRESSION)
    with autocast():
        padding_mask = (input_ids == PAD_ID)
        logits = model(input_ids, padding_mask=padding_mask, use_compression=False, log_during_train=False)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=PAD_ID
        ) / ACCUM_STEPS
        
        # ─── CAPTURE COHERENCE (Leading Indicator)
    # Get coherence from the last block's telemetry
    if hasattr(model, '_last_info'):
        coherence = model._last_info.get('coherence', 0.0)
    else:
        coherence = 0.0

    # ─── NAN GUARD
    if torch.isnan(loss) or torch.isinf(loss):
        nan_count += 1
        if nan_count < 3:
            opt.zero_grad()
            continue
        else:
            raise RuntimeError(f"Loss NaN at step {global_step}")

    # ─── BACKWARD
    scaler.scale(loss).backward()

    if (step + 1) % ACCUM_STEPS == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        
        # ─── UPDATE LR WITH CUSTOM SCHEDULER
        current_lr = lr_scheduler.step()

    losses.append(loss.item() * ACCUM_STEPS)
    global_step += 1

    # ─── LOGGING (EVERY 1000 STEPS)
    if step % LOG_EVERY == 0 and step > 0:
        avg_loss = np.mean(losses[-100:]) if losses else 0
        current_lr = lr_scheduler.get_lr()
        warmup_status = lr_scheduler.get_warmup_status()
        
        # Throughput stats
        stats = tracker.print_status(global_step)
        
        print(f"\n📊 Step {global_step:,} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | {warmup_status}")
        print(f"   Coherence: {coherence:.4f} {'📈' if coherence > 0.8 else '📉' if coherence < 0.5 else '➡️'}")
        print(f"   Speed: {stats['smoothed_tps']:.0f} tok/s | ETA: {stats['eta_hours']:.1f}h | {stats['progress_pct']:.1f}%")

        # ─── ADD BEST CHECKPOINT SAVING HERE ──────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_step = global_step
            
            best_ckpt = {
                'epoch': start_epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'lr_scheduler_step': lr_scheduler.current_step,
                'loss': avg_loss,
                'best_loss': best_loss,
                'coherence': coherence,
                'timestamp': datetime.now().isoformat(),
            }
            
            best_path = os.path.join(CKPT_DIR, "mycelia_best.pt")
            torch.save(best_ckpt, best_path)
            print(f"\n🏆 NEW BEST LOSS: {best_loss:.4f} at step {global_step:,}")
        # ──────────────────────────────────────────────────────────────────────────

    # ─── CHECKPOINT (REGULAR)
    if step % SAVE_EVERY == 0 and step > 0:
        hex_s = f"{global_step:05x}"
        path = os.path.join(CKPT_DIR, f"mycelia_step_{hex_s}.pt")

        checkpoint = {
            'epoch': start_epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'lr_scheduler_step': lr_scheduler.current_step,
            'loss': losses[-1],
            'avg_loss_100': float(np.mean(losses[-100:])) if len(losses) >= 100 else None,
            'timestamp': datetime.now().isoformat(),
        }

        torch.save(checkpoint, path)
        torch.save(checkpoint, LATEST_CKPT)
        print(f"\n💾 Checkpoint: step {global_step:,}")

        cleanup_old_checkpoints(CKPT_DIR, keep=2, verbose=True)

    # ─── CACHE CLEANUP
    if step % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════
# ─── FINAL SAVE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("💾 Saving final checkpoint...")
print("="*70)

final_ckpt = {
    'epoch': start_epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'scheduler_state_dict': sched.state_dict(),
    'loss': losses[-1] if losses else None,
    'avg_loss_100': float(np.mean(losses[-100:])) if len(losses) >= 100 else None,
    'timestamp': datetime.now().isoformat(),
}

torch.save(final_ckpt, LATEST_CKPT)
torch.save(final_ckpt, BEST_CKPT)

print(f"\n✅ Training complete!")
print(f"   Total steps: {global_step:,}")
print(f"   Final loss: {losses[-1]:.4f}" if losses else "   Final loss: N/A")
print(f"   Checkpoint: {LATEST_CKPT}")
print("="*70)