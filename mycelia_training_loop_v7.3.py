# ============================================
# MYCELIA v7.2 - PRODUCTION TRAINING PIPELINE
# T4-Optimized | Single HQ Source | Dynamic Compression
# ============================================
# ─── IMPORTS
# ============================================
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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

# ─── IMPORT ARCHITECTURE
try:
    from MYCELIA_architecture import MyceliaLM, MyceliaConfig
    print("Successfully linked to MYCELIA v7.2 architecture.")
except ImportError:
    raise ImportError("Please ensure 'MYCELIA_architecture.py' is in the same directory!")
from mycelia_jupyter_logger import MyceliaJupyterLogger
print("✅ Logger imported successfully")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SEQ_LEN = 512
BATCH_SIZE = 2
ACCUM_STEPS = 8
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 50000
GRAD_CLIP = 0.5
SAVE_EVERY = 500

COMPRESS_WINDOW = 128
COMPRESS_RATIO = 8

# ═══════════════════════════════════════════════════════════════════════════════
# ─── SINGLE S3 SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

S3_BUCKET = "sagemaker-eu-central-1-119287771635"
HQ_PREFIX = "massif-llm-highquality"

TCM_PRIORITY_FILES = [
    "tcm_books_processed.jsonl",
]

SECONDARY_FILES = [
    "tcm_nuclear_processed.jsonl",
    "tcm_shizhen.jsonl",
    "stanford_philosophy_processed.jsonl",
    "vedas.jsonl",
    "buddhism.jsonl",
]

MIN_TOKENS = 50
MAX_TOKENS = 6000
DEDUP_HASH_SIZE = 100000

# ═══════════════════════════════════════════════════════════════════════════════
# ─── DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

CKPT_DIR = os.path.join(os.environ.get('SM_MODEL_DIR', '/home/ec2-user/SageMaker'), 'mycelia_checkpoints')
OUT_DIR = os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', '/home/ec2-user/SageMaker'), 'mycelia_output')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "mycelia_latest.pt")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── QWEN TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

print("\nLoading Qwen tokenizer...")
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
# ─── DATA QUALITY FILTERING
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
        print("\n" + "="*60)
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
# ─── SINGLE-SOURCE S3 DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidatedHQDataset(IterableDataset):
    def __init__(self, bucket: str, prefix: str, tokenizer,
                 tcm_priority_files: List[str],
                 secondary_files: List[str],
                 max_seq_len: int = 4096,
                 compress_window: int = 128,
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
        self.s3 = boto3.client('s3', region_name='eu-central-1', config=s3_config)
        all_files = self._list_files()
        print(f"\n📁 Found {len(all_files)} files in s3://{bucket}/{prefix}")

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

        print(f"\n📊 Source breakdown:")
        print(f"   TCM priority: {len(self.tcm_files)} files")
        print(f"   Secondary:    {len(self.secondary_files)} files")
        print(f"   Other:        {len(self.other_files)} files")

        self.sources = self._build_weighted_sources()

    def _list_files(self) -> List[str]:
        try:
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            files = [o['Key'] for o in resp.get('Contents', []) 
                     if o['Key'].endswith('.jsonl')]
            return sorted(files)
        except Exception as e:
            print(f"   ⚠️  Error listing {self.prefix}: {e}")
            return []

    def _build_weighted_sources(self) -> List[Tuple[str, str]]:
        sources = []
        w_tcm = int(10 * self.tcm_weight)
        w_other = 10 - w_tcm
        
        all_non_tcm = self.secondary_files + self.other_files
        
        max_len = max(len(self.tcm_files), len(all_non_tcm), 1)
        
        for i in range(max_len * 3):
            slot = i % 10
            if slot < w_tcm and self.tcm_files:
                sources.append(('tcm', self.tcm_files[i % len(self.tcm_files)]))
            elif all_non_tcm:
                sources.append(('other', all_non_tcm[i % len(all_non_tcm)]))
        
        return sources
    
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
            print(f"   Error reading {key}: {e}")
    
    def _stream_all(self):
        while True:
            for source_type, key in self.sources:
                for row in self._stream_file(key):
                    text = self.quality_filter.filter_row(row)
                    if text is not None:
                        yield text

    def _tokenize_stream(self):
        for text in self._stream_all():
            try:
                tokens = self.tokenizer.encode(text, allowed_special="all")
            except:
                tokens = self.tokenizer.encode(text)

            for tok in tokens:
                yield tok
            yield self.tokenizer.eos_token_id or PAD_ID
            
    def __iter__(self):
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
# ─── DYNAMIC COMPRESSION SCHEDULER & LWES (end_prob=0.01 = DISABLED)
# ═══════════════════════════════════════════════════════════════════════════════

def get_compression_probability(step, total_steps=50000, start_prob=0.8, end_prob=0.01):
    decay = min(1.0, step / total_steps)
    prob = start_prob - (start_prob - end_prob) * decay
    return max(end_prob, prob)


def calculate_lwes_sweet_spot(loss_compressed, loss_uncompressed, cumulative_gb, 
                               gamma=3.0, beta=10.0):
    if loss_compressed == 0 or loss_uncompressed == 0:
        return 0.0
    loss_fidelity_ratio = loss_uncompressed / loss_compressed
    penalty_component = loss_fidelity_ratio ** gamma
    vram_utility_bonus = (cumulative_gb / beta) + 1.0
    return penalty_component * vram_utility_bonus


class MyceliaArchitectController:
    def __init__(self, checkpoint_dir, patience_steps=1500, min_score_target=0.20):
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience_steps
        self.min_score_target = min_score_target
        
        self.best_sweet_spot_score = -float('inf')
        self.best_step = 0
        self.steps_without_improvement = 0
        self.early_stop_triggered = False
        
        self.history = {'steps': [], 'scores': [], 'losses': [], 'savings': []}
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def evaluate_and_action(self, current_step, model, optimizer, sweet_spot_score, 
                            loss=None, savings_gb=None):
        should_stop = False
        should_save = False
        
        self.history['steps'].append(current_step)
        self.history['scores'].append(sweet_spot_score)
        if loss is not None:
            self.history['losses'].append(loss)
        if savings_gb is not None:
            self.history['savings'].append(savings_gb)
        
        if sweet_spot_score > self.best_sweet_spot_score:
            self.best_sweet_spot_score = sweet_spot_score
            self.best_step = current_step
            self.steps_without_improvement = 0
            should_save = True
            self._save_checkpoint(current_step, model, optimizer, sweet_spot_score)
            print(f"\n🌟 [NEW SWEET SPOT] Step {current_step} | LWES: {sweet_spot_score:.4f}")
        else:
            self.steps_without_improvement += 1
        
        if current_step > 5000:  # Warm-up period
            if sweet_spot_score < self.min_score_target and sweet_spot_score > 0:
                print(f"\n🛑 [CRITICAL STOP] Step {current_step}")
                print(f"   Sweet Spot Score collapsed to {sweet_spot_score:.4f}")
                print(f"   Minimum threshold: {self.min_score_target:.4f}")
                self.early_stop_triggered = True
                should_stop = True
            elif self.steps_without_improvement >= self.patience:
                print(f"\n🛑 [EARLY STOP] Step {current_step}")
                print(f"   No LWES improvement for {self.patience} steps.")
                print(f"   Best Score: {self.best_sweet_spot_score:.4f} (step {self.best_step})")
                self.early_stop_triggered = True
                should_stop = True
        
        return should_stop, should_save
    
    def _save_checkpoint(self, step, model, optimizer, score):
        path = os.path.join(self.checkpoint_dir, f"mycelia_sweetspot_step_{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sweet_spot_score': score,
        }, path)
        print(f"   💾 Sweet spot checkpoint: {path}")
    
    def print_summary(self):
        print("\n" + "="*70)
        print("🍄 MYCELIA ARCHITECT CONTROLLER SUMMARY")
        print("="*70)
        print(f"   Best Sweet Spot Score: {self.best_sweet_spot_score:.4f}")
        print(f"   Best Step: {self.best_step}")
        print(f"   Total Steps Evaluated: {len(self.history['steps'])}")
        print(f"   Status: {'⏹️ Early Stopped' if self.early_stop_triggered else '✅ Active'}")
        print("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── CLEANUP FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_old_checkpoints(ckpt_dir, controller=None, verbose=True):
    """Clean up old checkpoints, keeping only recent ones."""
    import glob
    import re
    import os
    
    # 1. Clean regular step checkpoints (keep 2 most recent)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "mycelia_step_*.pt")), key=os.path.getmtime)
    for old in ckpts[:-2]:
        try:
            os.remove(old)
            if verbose:
                print(f"   🗑️ Removed old checkpoint: {os.path.basename(old)}")
        except:
            pass
    
    # 2. Clean sweet spot checkpoints (keep best + most recent)
    sweet_ckpts = glob.glob(os.path.join(ckpt_dir, "mycelia_sweetspot_step_*.pt"))
    
    if len(sweet_ckpts) > 2 and controller is not None:
        step_pattern = re.compile(r'mycelia_sweetspot_step_(\d+)\.pt')
        sweet_steps = []
        for ckpt in sweet_ckpts:
            match = step_pattern.search(ckpt)
            if match:
                sweet_steps.append(int(match.group(1)))
        
        most_recent_step = max(sweet_steps) if sweet_steps else None
        best_step = controller.best_step
        
        keep_steps = {best_step}
        if most_recent_step and most_recent_step != best_step:
            keep_steps.add(most_recent_step)
        
        for ckpt in sweet_ckpts:
            match = step_pattern.search(ckpt)
            if match:
                step = int(match.group(1))
                if step not in keep_steps:
                    try:
                        os.remove(ckpt)
                        if verbose:
                            print(f"   🗑️ Removed old sweet spot: {os.path.basename(ckpt)}")
                    except:
                        pass
                        
# ═══════════════════════════════════════════════════════════════════════════════
# ─── MODEL SETUP
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding model...")
cfg = MyceliaConfig()
cfg.max_seq_len = MAX_SEQ_LEN
cfg.vocab_size = 151643
cfg.compress_window = COMPRESS_WINDOW
cfg.compress_ratio = COMPRESS_RATIO

model = MyceliaLM(cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
device = next(model.parameters()).device
total_params = sum(p.numel() for p in model.parameters())
print(f"   {total_params:,} parameters on {device}")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── OPTIMIZER & SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=MAX_STEPS, eta_min=1e-6)
scaler = GradScaler()

# ═══════════════════════════════════════════════════════════════════════════════
# ─── CHECKPOINT RESUMPTION
# ═══════════════════════════════════════════════════════════════════════════════

global_step = 0
start_epoch = 0

if os.path.exists(LATEST_CKPT):
    print("\n" + "="*70)
    print("CHECKPOINT FOUND — RESUMING TRAINING")
    print("="*70)

    ckpt = torch.load(LATEST_CKPT, map_location='cpu')
    
    print(f"   📋 Checkpoint keys: {list(ckpt.keys())}")

    # ─── LOAD MODEL WEIGHTS ──────────────────────────────────────────────────
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    print(f"   ✅ Model weights loaded (strict=False)")

    # ─── WARM-START input_pos FROM latent_pos ──────────────────────────────
    if hasattr(model.compressor, 'input_pos') and hasattr(model.compressor, 'latent_pos'):
        with torch.no_grad():
            latent = model.compressor.latent_pos.data
            window = model.compressor.window
            if latent.shape[1] >= window:
                model.compressor.input_pos.data = latent[:, :window, :].clone()
                print(f"   🔥 Warm-started input_pos from latent_pos ({window} positions)")
            else:
                repeats = window // latent.shape[1] + 1
                expanded = latent.repeat(1, repeats, 1)[:, :window, :]
                model.compressor.input_pos.data = expanded.clone()
                print(f"   🔥 Interpolated input_pos from latent_pos ({window} positions)")

    # ─── RESTORE TRAINING STATE ──────────────────────────────────────────────
    global_step = ckpt.get('global_step', 0)
    start_epoch = ckpt.get('epoch', 0)
    prev_loss = ckpt.get('loss', 'N/A')
    avg_loss_100 = ckpt.get('avg_loss_100', 'N/A')

    print(f"\n   📊 CHECKPOINT DATA RESTORED:")
    print(f"      Global Step:  {global_step:,}")
    print(f"      Epoch:        {start_epoch}")
    print(f"      Loss:         {prev_loss:.4f}" if prev_loss != 'N/A' and prev_loss is not None else f"      Loss:         {prev_loss}")
    # Fix for avg_loss_100 - properly indented inside the if block
    if avg_loss_100 != 'N/A' and avg_loss_100 is not None:
        print(f"      Avg Loss 100: {avg_loss_100:.4f}")
    else:
        print(f"      Avg Loss 100: {avg_loss_100}")
    print(f"      Timestamp:    {ckpt.get('timestamp', 'N/A')}")

    # ─── OPTIMIZER ────────────────────────────────────────────────────────────
    print(f"\n   🔄 Reinitializing optimizer (architecture changed - new parameters)")
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"   ✅ Optimizer initialized with LR={LR}, weight_decay={WEIGHT_DECAY}")

    # ─── SCHEDULER ────────────────────────────────────────────────────────────
    try:
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"   ✅ Scheduler state loaded")
    except:
        print(f"   ⚠️ Scheduler state not found, reinitializing")
        sched = CosineAnnealingLR(opt, T_max=MAX_STEPS, eta_min=1e-6)

    print(f"\n   ✅ CONTINUING FROM STEP {global_step:,} (Epoch {start_epoch})")
    print("="*70)
    
    start_epoch += 1
    
else:
    print("\n" + "="*70)
    print("NO CHECKPOINT — STARTING FRESH")
    print("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── DATALOADER
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading consolidated HQ data stream for epoch {start_epoch}...")
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
    tcm_weight=0.7
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=0)
data_iter = iter(loader)
logger = MyceliaJupyterLogger(refresh_rate_steps=10)

# ─── INITIALIZE CONTROLLER ────────────────────────────────────────────────────

controller = MyceliaArchitectController(
    checkpoint_dir=CKPT_DIR,
    patience_steps=1500,
    min_score_target=0.20
)

print("\n🍄 Mycelia Automated Guardrails Engaged.")
print(f"   Patience: {controller.patience} steps")
print(f"   Minimum score threshold: {controller.min_score_target:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"EPOCH {start_epoch} — CONTINUING TRAINING FROM STEP {global_step:,}")
print("="*70)

model.train()
losses = []
comp_on_losses = []
comp_off_losses = []
nan_count = 0
NAN_PATIENCE = 2

for b in model.blocks:
    b.mycelia.reset_stats()

import random

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

    # ─── DYNAMIC COMPRESSION SCHEDULER ──────────────────────────────────────
    current_compress_prob = get_compression_probability(
        global_step, 
        total_steps=MAX_STEPS,
        start_prob=0.8,
        end_prob=0.01
    )

    use_comp = (cfg.use_compression and 
                random.random() < current_compress_prob and 
                T > COMPRESS_WINDOW)

    if global_step % 100 == 0:
        tqdm.write(f"  📊 Step {global_step}: Compress prob = {current_compress_prob:.3f} | use_comp = {use_comp}")

    with autocast():
        padding_mask = (input_ids == PAD_ID)
        logits = model(input_ids, padding_mask=padding_mask, use_compression=use_comp, log_during_train=(step % 100 == 0))

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

    # ─── NAN/INF GUARD ──────────────────────────────────────────────────────
    if torch.isnan(loss) or torch.isinf(loss):
        nan_count += 1
        if nan_count < NAN_PATIENCE:
            tqdm.write(f"⚠️ Skipping batch at step {global_step} (loss is NaN/Inf)")
            opt.zero_grad()
            continue
        else:
            raise RuntimeError(f"Loss is NaN/Inf at step {global_step}")

    # ─── BACKWARD PASS ──────────────────────────────────────────────────────
    scaler.scale(loss).backward()

    if (step + 1) % ACCUM_STEPS == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

    losses.append(loss.item() * ACCUM_STEPS)

    if use_comp:
        comp_on_losses.append(loss.item() * ACCUM_STEPS)
    else:
        comp_off_losses.append(loss.item() * ACCUM_STEPS)

    global_step += 1

    # ─── LOGGING ──────────────────────────────────────────────────────────────
    if step % 10 == 0:
        avg = np.mean(losses[-100:]) if losses else 0
        lr = sched.get_last_lr()[0]
        comp_status = "ON" if use_comp else "OFF"
        tqdm.write(f"Step {global_step:5d} | Loss: {avg:.4f} | LR: {lr:.2e} | Comp: {comp_status} | Prob: {current_compress_prob:.3f}")

        if hasattr(model, '_last_info'):
            total_kept = 0
            total_vetoed = 0
            for block in model.blocks:
                stats = block.mycelia.get_stats()
                total_kept += stats['kept']
                total_vetoed += stats['vetoed']

            stats_dict = {
                'total': total_kept + total_vetoed,
                'kept': total_kept,
                'vetoed': total_vetoed,
            }
            
            cumulative_gb = model._last_info.get('cumulative_gb', 0.0)
            
            # Calculate rolling means for compression ON/OFF
            mean_on = np.mean(comp_on_losses[-100:]) if len(comp_on_losses) >= 50 else 0
            mean_off = np.mean(comp_off_losses[-100:]) if len(comp_off_losses) >= 50 else 0
            
            # Only calculate LWES if we have sufficient data from BOTH states
            if mean_on > 0 and mean_off > 0:
                sweet_spot_score = calculate_lwes_sweet_spot(
                    loss_compressed=mean_on,
                    loss_uncompressed=mean_off,
                    cumulative_gb=cumulative_gb,
                    gamma=3.0,
                    beta=10.0
                )
                model._last_info['sweet_spot_score'] = sweet_spot_score
            else:
                sweet_spot_score = 0.0
                model._last_info['sweet_spot_score'] = 0.0
            
            logger.update(step=global_step, stats_dict=stats_dict, info_dict=model._last_info)
            
            # ─── Controller Evaluation (only with sufficient data) ──────────
            if len(comp_on_losses) >= 50 and len(comp_off_losses) >= 50 and sweet_spot_score > 0:
                stop_training, _ = controller.evaluate_and_action(
                    current_step=global_step,
                    model=model,
                    optimizer=opt,
                    sweet_spot_score=sweet_spot_score,
                    loss=avg,
                    savings_gb=cumulative_gb
                )
                
                if stop_training:
                    cleanup_old_checkpoints(CKPT_DIR, controller, verbose=True)
                    tqdm.write("\n💾 Final stable parameters saved. Exiting cleanly.")
                    break
            else:
                if step % 100 == 0:
                    tqdm.write(f"   ⏳ Collecting data... (ON: {len(comp_on_losses)}, OFF: {len(comp_off_losses)})")

    # ─── CHECKPOINT SAVING ──────────────────────────────────────────────────
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
        tqdm.write(f"\n💾 Checkpoint: step {global_step}, epoch {start_epoch}")

    # ─── CACHE CLEANUP ──────────────────────────────────────────────────────
    if step % 50 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ─── CLEANUP OLD CHECKPOINTS ───────────────────────────────────────────────
cleanup_old_checkpoints(CKPT_DIR, controller, verbose=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── FINAL SAVE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
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

# ═══════════════════════════════════════════════════════════════════════════════
# ─── COMPRESSION LOSS RATIO REPORT
# ═══════════════════════════════════════════════════════════════════════════════

if comp_on_losses and comp_off_losses:
    mean_on = sum(comp_on_losses) / len(comp_on_losses)
    mean_off = sum(comp_off_losses) / len(comp_off_losses)
    ratio = mean_on / mean_off if mean_off > 0 else 0.0
    print("\n" + "="*70)
    print("COMPRESSION LOSS RATIO REPORT")
    print("="*70)
    print(f"   Compression ON  steps: {len(comp_on_losses):4d} | Mean loss: {mean_on:.4f}")
    print(f"   Compression OFF steps: {len(comp_off_losses):4d} | Mean loss: {mean_off:.4f}")
    print(f"   Loss Ratio (ON / OFF): {ratio:.4f}")
    if ratio > 1.0:
        print(f"   Effect: Compression adds {(ratio - 1) * 100:.1f}% penalty to loss")
    elif ratio < 1.0:
        print(f"   Effect: Compression reduces loss by {(1 - ratio) * 100:.1f}%")
    else:
        print(f"   Effect: Compression has neutral impact on loss")
    print("="*70)
else:
    print("\n   [Compression ratio report skipped — insufficient mixed data]")

# ─── CONTROLLER SUMMARY ──────────────────────────────────────────────────────

controller.print_summary()

# ═══════════════════════════════════════════════════════════════════════════════
# ─── END COMPRESSION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n✅ Epoch {start_epoch} complete!")
print(f"   Total steps: {global_step:,}")
print(f"   Final loss: {losses[-1]:.4f}" if losses else "   Final loss: N/A")
print(f"   Checkpoint: {LATEST_CKPT}")
print(f"\n   >>> RUN AGAIN FOR EPOCH {start_epoch + 1} <<<")
print("="*70)