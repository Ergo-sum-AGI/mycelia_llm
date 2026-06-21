# ============================================
# MYCELIA v7.1 - PRODUCTION TRAINING PIPELINE
# T4-Optimized | Single HQ Source | Internal TCM-Priority Mixing
# Data: s3://.../massif-llm-highquality/ (consolidated)
# ============================================

import os
import gc
import json
import re
import boto3
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from typing import List, Dict, Iterator, Optional, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ─── IMPORT ARCHITECTURE ──────────────────────────────────────────────────────
try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("Successfully linked to MYCELIA v7.1 architecture.")
except ImportError:
    raise ImportError("Please ensure 'MYCELIA_architecture.py' is in the same directory!")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SEQ_LEN = 4096
BATCH_SIZE = 2
ACCUM_STEPS = 8
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_STEPS = 50000
GRAD_CLIP = 1.0
SAVE_EVERY = 500
COMPRESS_FREQ = 4
COMPRESS_WINDOW = 256
COMPRESS_RATIO = 8

# ─── SINGLE S3 SOURCE (consolidated high-quality) ────────────────────────────
S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"

# Internal file priorities within the HQ folder
# These files get higher sampling weight
TCM_PRIORITY_FILES = [
    "tcm_nuclear_processed.jsonl",
    "tcm_books_processed.jsonl",
    "tcm_shizhen.jsonl",
]

# Secondary files (lower weight)
SECONDARY_FILES = [
    "buddhism.jsonl",
    "vedas.jsonl",
    "gsm8k.jsonl",
    "web_math.jsonl",
]

# Data quality thresholds
MIN_TOKENS = 50
MAX_TOKENS = 6000
DEDUP_HASH_SIZE = 100000

# ─── DIRECTORIES ──────────────────────────────────────────────────────────────
CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/tmp'), 'mycelia_checkpoints')
OUT_DIR = os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', '/tmp'), 'mycelia_output')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")

# ═══════════════════════════════════════════════════════════════════════════════
# QWEN TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════
print("\\nLoading Qwen tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    print("   Loaded Qwen2.5-1.5B tokenizer")
except Exception as e:
    print(f"   Qwen2.5 failed ({e}), trying Qwen-7B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    print("   Loaded Qwen-7B tokenizer")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

PAD_ID = tokenizer.pad_token_id or 0
VOCAB_SIZE = tokenizer.vocab_size
print(f"   Vocab size: {VOCAB_SIZE} | Pad token: {tokenizer.pad_token}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

class DataQualityFilter:
    """Cleans and filters raw JSONL chunks before tokenization."""
    
    JUNK_PATTERNS = [
        r'what is the daily horoscope',
        r'horoscope for \\w+ today',
        r'click here to learn more',
        r'copyright \\d{4}',
        r'all rights reserved',
        r'page \\d+ of \\d+',
        r'table of contents',
        r'\\[?\\s*edit\\s*\\]?',
        r'^\\s*references\\s*$',
        r'^\\s*see also\\s*$',
        r'^\\s*external links\\s*$',
        r'^\\s*appendix\\s*$',
        r'^\\s*index\\s*$',
        r'\\b\\d{4}-\\d{2}-\\d{2}\\b',
        r'\\bISBN\\b',
        r'\\bDOI\\b',
        r'\\bhttp[s]?://',
    ]
    
    JUNK_RE = [re.compile(p, re.IGNORECASE) for p in JUNK_PATTERNS]
    
    def __init__(self, dedup_size: int = 100000):
        self.seen_hashes = set()
        self.dedup_size = dedup_size
        self.stats = {
            'total_rows': 0,
            'rejected_short': 0,
            'rejected_long': 0,
            'rejected_junk': 0,
            'rejected_dup': 0,
            'rejected_repetitive': 0,
            'accepted': 0,
        }
    
    def _is_junk(self, text: str) -> bool:
        for pattern in self.JUNK_RE:
            if pattern.search(text):
                return True
        return False
    
    def _is_repetitive(self, text: str) -> bool:
        lines = [l.strip() for l in text.split('\\n') if l.strip()]
        if len(lines) < 3:
            return False
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.3:
            return True
        words = text.split()
        if len(words) > 20:
            from collections import Counter
            fivegrams = [' '.join(words[i:i+5]) for i in range(len(words)-4)]
            if fivegrams:
                most_common = Counter(fivegrams).most_common(1)[0][1]
                if most_common > len(fivegrams) * 0.3:
                    return True
        return False
    
    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text[:500].encode()).hexdigest()
    
    def filter_row(self, row: Dict) -> Optional[str]:
        self.stats['total_rows'] += 1
        
        text = (row.get("text") or row.get("content") or row.get("body") 
                or row.get("instruction") or row.get("input") 
                or row.get("question") or row.get("prompt") 
                or row.get("answer") or row.get("output") or "").strip()
        
        if not text:
            return None
        
        char_len = len(text)
        if char_len < 50:
            self.stats['rejected_short'] += 1
            return None
        if char_len > 50000:
            self.stats['rejected_long'] += 1
            return None
        
        if self._is_junk(text):
            self.stats['rejected_junk'] += 1
            return None
        
        if self._is_repetitive(text):
            self.stats['rejected_repetitive'] += 1
            return None
        
        h = self._get_hash(text)
        if h in self.seen_hashes:
            self.stats['rejected_dup'] += 1
            return None
        
        self.seen_hashes.add(h)
        if len(self.seen_hashes) > self.dedup_size:
            self.seen_hashes.pop()
        
        self.stats['accepted'] += 1
        return text
    
    def print_stats(self):
        print("\\n" + "="*60)
        print("DATA QUALITY FILTER STATS")
        print("="*60)
        total = self.stats['total_rows']
        if total == 0:
            print("   No rows processed")
            return
        for k, v in self.stats.items():
            pct = 100 * v / total if total > 0 else 0
            print(f"   {k:20s}: {v:6d} ({pct:5.1f}%)")
        print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-SOURCE S3 DATASET WITH INTERNAL PRIORITY MIXING
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidatedHQDataset(IterableDataset):
    """
    Streams from a single S3 folder (massif-llm-highquality/)
    with internal file-level priority mixing.
    
    TCM files get 70% weight, secondary files get 30% weight.
    """
    
    def __init__(self, bucket: str, prefix: str, tokenizer,
                 tcm_priority_files: List[str],
                 secondary_files: List[str],
                 max_seq_len: int = 4096,
                 compress_window: int = 256,
                 quality_filter: Optional[DataQualityFilter] = None,
                 tcm_weight: float = 0.7):
        self.bucket = bucket
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.quality_filter = quality_filter or DataQualityFilter()
        self.max_seq_len = max_seq_len
        self.compress_window = compress_window
        self.effective_len = (max_seq_len // compress_window) * compress_window
        self.target_tokens = self.effective_len + 1
        self.tcm_weight = tcm_weight
        
        import botocore.config
        s3_config = botocore.config.Config(response_checksum_validation="when_required")
        s3 = boto3.client('s3', region_name=S3_REGION, config=s3_config)
        
        # List all files in the HQ folder
        all_files = self._list_files()
        print(f"\\n📁 Found {len(all_files)} files in s3://{bucket}/{prefix}")
        
        # Categorize files
        self.tcm_files = []
        self.secondary_files = []
        self.other_files = []
        
        for f in all_files:
            basename = os.path.basename(f)
            if basename in tcm_priority_files:
                self.tcm_files.append(f)
                print(f"   🌿 [TCM] {basename}")
            elif basename in secondary_files:
                self.secondary_files.append(f)
                print(f"   📚 [SEC] {basename}")
            else:
                self.other_files.append(f)
                print(f"   📄 [OTH] {basename}")
        
        print(f"\\n📊 Source breakdown:")
        print(f"   TCM priority: {len(self.tcm_files)} files")
        print(f"   Secondary:    {len(self.secondary_files)} files")
        print(f"   Other:        {len(self.other_files)} files")
        
        # Build weighted source list
        self.sources = self._build_weighted_sources()
    
    def _list_files(self) -> List[str]:
        """List all JSONL files under the prefix."""
        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            files = [o['Key'] for o in resp.get('Contents', []) 
                     if o['Key'].endswith('.jsonl')]
            return sorted(files)
        except Exception as e:
            print(f"   ⚠️  Error listing {self.prefix}: {e}")
            return []
    
    def _build_weighted_sources(self) -> List[Tuple[str, str]]:
        """Build weighted round-robin source list."""
        sources = []
        
        # Weight: 70% TCM, 30% other
        w_tcm = int(10 * self.tcm_weight)
        w_other = 10 - w_tcm
        
        all_non_tcm = self.secondary_files + self.other_files
        
        max_len = max(len(self.tcm_files), len(all_non_tcm), 1)
        
        for i in range(max_len * 3):  # 3x for variety
            slot = i % 10
            if slot < w_tcm and self.tcm_files:
                sources.append(('tcm', self.tcm_files[i % len(self.tcm_files)]))
            elif all_non_tcm:
                sources.append(('other', all_non_tcm[i % len(all_non_tcm)]))
        
        return sources
    
    def _stream_file(self, key: str):
        """Stream JSONL lines from a specific S3 file."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            for line in obj['Body'].iter_lines():
                if line:
                    try:
                        yield json.loads(line.decode('utf-8'))
                    except:
                        pass
        except Exception as e:
            print(f"   Error reading {key}: {e}")
    
    def _stream_all(self):
        """Stream from all sources in weighted order."""
        for source_type, key in self.sources:
            for row in self._stream_file(key):
                text = self.quality_filter.filter_row(row)
                if text is not None:
                    yield text
    
    def _tokenize_stream(self):
        """Tokenize text stream into continuous token buffer."""
        for text in self._stream_all():
            try:
                tokens = self.tokenizer.encode(text, allowed_special="all")
            except:
                tokens = self.tokenizer.encode(text)
            
            for tok in tokens:
                yield tok
            yield self.tokenizer.eos_token_id or PAD_ID
    
    def __iter__(self):
        """Pack tokens into fixed-length sequences aligned to compress_window."""
        buffer = []
        for tok in self._tokenize_stream():
            buffer.append(tok)
            
            while len(buffer) >= self.target_tokens:
                seq = buffer[:self.target_tokens]
                buffer = buffer[self.target_tokens:]
                yield torch.tensor(seq, dtype=torch.long)
        
        if len(buffer) >= self.compress_window + 1:
            while len(buffer) < self.target_tokens:
                buffer.append(PAD_ID)
            yield torch.tensor(buffer[:self.target_tokens], dtype=torch.long)


def collate(batch):
    return torch.stack(batch)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ═══════════════════════════════════════════════════════════════════════════════
print("\\nBuilding model...")
cfg = MyceliaConfig()
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = VOCAB_SIZE
cfg.compress_window = COMPRESS_WINDOW
cfg.compress_ratio = COMPRESS_RATIO

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
total_params = sum(p.numel() for p in model.parameters())
print(f"   {total_params:,} parameters on {device}")

# ─── OPTIMIZER ────────────────────────────────────────────────────────────────
opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=MAX_STEPS, eta_min=1e-6)
scaler = GradScaler()

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUMPTION
# ═══════════════════════════════════════════════════════════════════════════════

global_step = 0
start_epoch = 0

if os.path.exists(LATEST_CKPT):
    print("\\n" + "="*70)
    print("CHECKPOINT FOUND — RESUMING TRAINING")
    print("="*70)
    ckpt = torch.load(LATEST_CKPT, map_location=device)
    
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    opt.load_state_dict(kt['optimizer_state_dict'])
    sched.load_state_dict(ckpt['scheduler_state_dict'])
    
    global_step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0)
    prev_loss = ckpt.get('loss', 'N/A')
    
    print(f"   Previous: Epoch {start_epoch}, Step {global_step}")
    print(f"   Previous loss: {prev_loss}")
    print(f"   >>> CONTINUING FROM WHERE WE LEFT OFF <<<")
    print("="*70)
    
    start_epoch += 1
else:
    print("\\n" + "="*70)
    print("NO CHECKPOINT — STARTING FRESH")
    print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# DATALOADER
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\\nLoading consolidated HQ data stream for epoch {start_epoch}...")
quality_filter = DataQualityFilter(dedup_size=DEDUP_HASH_SIZE)

dataset = ConsolidatedHQDataset(
    bucket=S3_BUCKET,
    prefix=HQ_PREFIX,
    tokenizer=tokenizer,
    tcm_priority_files=TCM_PRIORITY_FILES,
    secondary_files=SECONDARY_FILES,
    max_seq_len=MAX_SEQ_LEN,
    compress_window=COMPRESS_WINDOW,
    quality_filter=quality_filter,
    tcm_weight=0.7  # 70% TCM, 30% other
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=0)
data_iter = iter(loader)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("\\n" + "="*70)
print(f"EPOCH {start_epoch} — TRAINING")
print("="*70)

model.train()
losses = []

for step in tqdm(range(MAX_STEPS), desc=f"Epoch {start_epoch}", initial=global_step):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
    
    batch = batch.to(device)
    input_ids = batch[:, :-1].contiguous()
    target_ids = batch[:, 1:].contiguous()
    
    B, T = input_ids.shape
    use_comp = (cfg.use_compression and step % COMPRESS_FREQ == 0 and step > 0)
    
    with autocast():
        logits = model(input_ids, use_compression=use_comp, log_during_train=(step % 100 == 0))
        
        if use_comp and T > COMPRESS_WINDOW:
            compressed_prefix = COMPRESS_WINDOW // COMPRESS_RATIO
            suffix_len = T - COMPRESS_WINDOW
            effective_seq_len = compressed_prefix + suffix_len
            target_ids = target_ids[:, :effective_seq_len]
            logits = logits[:, :effective_seq_len, :]
        
        assert logits.shape[1] == target_ids.shape[1], \
            f"Shape mismatch: logits {logits.shape}, targets {target_ids.shape}"
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=PAD_ID
        ) / ACCUM_STEPS
    
    scaler.scale(loss).backward()
    
    if (step + 1) % ACCUM_STEPS == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()
    
    losses.append(loss.item() * ACCUM_STEPS)
    global_step += 1
    
    if step % 10 == 0:
        avg = np.mean(losses[-100:]) if losses else 0
        lr = sched.get_last_lr()[0]
        comp_status = "ON" if use_comp else "OFF"
        tqdm.write(f"Step {global_step:5d} | Loss: {avg:.4f} | LR: {lr:.2e} | Comp: {comp_status}")
    
    if step % SAVE_EVERY == 0 and step > 0:
        hex_s = f"{global_step:05x}"
        path = os.path.join(CKPT_DIR, f"mycelia_step_{hex_s}.pt")
        
        checkpoint = {
            'epoch': start_epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'loss': losses[-1],
            'avg_loss_100': float(np.mean(losses[-100:])) if len(losses) >= 100 else None,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, path)
        torch.save(checkpoint, LATEST_CKPT)
        print(f"\\n💾 Checkpoint: step {global_step}, epoch {start_epoch}")
    
    if step % 50 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\\n" + "="*70)
print("SAVING FINAL CHECKPOINT")
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
hex_s = f"{global_step:05x}"
torch.save(final_ckpt, os.path.join(CKPT_DIR, f"mycelia_step_{hex_s}.pt"))

torch.save({
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'epoch': start_epoch,
    'global_step': global_step,
    'final_loss': losses[-1] if losses else None,
}, os.path.join(OUT_DIR, f"mycelia_epoch_{start_epoch}_final.pt"))

quality_filter.print_stats()

print(f"\\n✅ Epoch {start_epoch} complete!")
print(f"   Total steps: {global_step}")
print(f"   Final loss: {losses[-1]:.4f}" if losses else "   Final loss: N/A")
print(f"   Checkpoint: {LATEST_CKPT}")
print(f"\\n   >>> RUN AGAIN FOR EPOCH {start_epoch + 1} <<<")
print("="*70)
