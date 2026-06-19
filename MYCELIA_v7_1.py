# ============================================
# MYCELIA LM v7.1 — S3-AWARE TCM TRAINING
# Sinusoidal PE | Interleaved Compression | MASSIF Telemetry | S3 JSONL Streaming
# ============================================

import os, sys, warnings, json, time, math, random, gc, re
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Iterator
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm

# ─── S3 DEPENDENCIES ──────────────────────────────────────────────────────────
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("⚠️ boto3 not installed. S3 features disabled. Install with: pip install boto3")

# ─── ENVIRONMENT DETECTION ────────────────────────────────────────────────────
IN_SAGEMAKER = os.environ.get('SM_TRAINING_ENV', 'false').lower() == 'true'
SAGEMAKER_CHANNEL = os.environ.get('SM_CHANNEL_TRAIN', None)
SAGEMAKER_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', None)

print("=" * 80)
print("🍄 MYCELIA LM v7.1 — S3-AWARE TCM TRAINING")
print("   Sinusoidal Positions | Interleaved Compression | S3 JSONL Streaming")
print("=" * 80)
print(f"\n🔍 Environment: {'SageMaker' if IN_SAGEMAKER else 'Local/Colab'}")
print(f"   boto3 available: {'✅' if HAS_BOTO3 else '❌'}")
if IN_SAGEMAKER:
    print(f"   SM_CHANNEL_TRAIN: {SAGEMAKER_CHANNEL}")
    print(f"   SM_CHANNEL_VALIDATION: {SAGEMAKER_VALIDATION}")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

@dataclass
class MyceliaConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    vocab_size: int = 151936  # Qwen-7B vocab size
    max_seq_len: int = 512
    fib_weights: Tuple = (5, 8, 13, 21, 34, 55)
    dissenter_threshold: float = 2.5
    dubito_threshold: float = 7.0
    consensus_rounds: int = 2
    # Compressor
    use_compression: bool = True
    compress_ratio: int = 8
    compress_window: int = 128
    compress_freq: int = 4
    # S3 / Data
    s3_bucket: str = "sagemaker-eu-central-1-119287771635"
    s3_prefix_train: str = "massif-llm/clean-v1/train"
    s3_prefix_val: str = "massif-llm/clean-v1/val"
    s3_fallback_files: List[str] = field(default_factory=lambda: [
        "massif-llm/tcm_nuclear_110mb.jsonl"
    ])
    local_data_dir: str = "/tmp/mycelia_data"  # SageMaker / local cache
    prefetch_buffer_size: int = 1000  # Lines to buffer in memory
    dataset_split_train: float = 0.95

config = MyceliaConfig()

# ─── S3 INVENTORY & FETCHING ──────────────────────────────────────────────────

class S3DataInventory:
    """
    Full S3 inventory with logging. Lists, validates, and streams JSONL files.
    """
    def __init__(self, bucket: str, region: str = "eu-central-1"):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client('s3', region_name=region) if HAS_BOTO3 else None
        self.inventory: List[Dict] = []
        self.total_bytes = 0
        self.total_lines = 0

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   [{timestamp}] 📦 S3 | {msg}")

    def list_prefix(self, prefix: str, max_keys: int = 1000) -> List[Dict]:
        """List all objects under a prefix with metadata."""
        if not self.s3:
            self._log("❌ boto3 unavailable — cannot list S3")
            return []

        self._log(f"🔍 Scanning s3://{self.bucket}/{prefix}")
        objects = []
        paginator = self.s3.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    size = obj['Size']
                    last_modified = obj['LastModified']
                    is_jsonl = key.endswith('.jsonl') or key.endswith('.json')
                    entry = {
                        'key': key,
                        'size_mb': size / (1024 * 1024),
                        'last_modified': last_modified.isoformat(),
                        'type': 'jsonl' if is_jsonl else 'other',
                        'status': 'pending'
                    }
                    objects.append(entry)
                    self.total_bytes += size
        except ClientError as e:
            self._log(f"❌ S3 error: {e}")
            return []

        self.inventory.extend(objects)
        jsonl_count = sum(1 for o in objects if o['type'] == 'jsonl')
        self._log(f"✅ Found {len(objects)} objects ({jsonl_count} JSONL) under {prefix}")
        self._log(f"   Total size: {self.total_bytes / (1024*1024):.2f} MB")
        return objects

    def list_fallback(self, keys: List[str]) -> List[Dict]:
        """Check specific fallback files."""
        objects = []
        for key in keys:
            try:
                head = self.s3.head_object(Bucket=self.bucket, Key=key)
                size = head['ContentLength']
                entry = {
                    'key': key,
                    'size_mb': size / (1024 * 1024),
                    'last_modified': head['LastModified'].isoformat(),
                    'type': 'jsonl' if key.endswith('.jsonl') else 'other',
                    'status': 'fallback'
                }
                objects.append(entry)
                self.total_bytes += size
                self._log(f"✅ Fallback ready: {key} ({entry['size_mb']:.1f} MB)")
            except ClientError:
                self._log(f"⚠️ Fallback missing: {key}")
        self.inventory.extend(objects)
        return objects

    def print_inventory(self):
        """Pretty-print the full inventory."""
        print("\n" + "=" * 70)
        print("📋 S3 DATA INVENTORY")
        print("=" * 70)
        print(f"{'Key':<50} {'Size (MB)':>10} {'Type':>8} {'Status':>10}")
        print("-" * 70)
        for item in self.inventory:
            short_key = item['key'][-47:] if len(item['key']) > 50 else item['key']
            print(f"{short_key:<50} {item['size_mb']:>10.2f} {item['type']:>8} {item['status']:>10}")
        print("-" * 70)
        print(f"{'TOTAL':<50} {self.total_bytes/(1024*1024):>10.2f} MB")
        print("=" * 70)

    def stream_jsonl_lines(self, keys: List[str]) -> Iterator[Dict]:
        """
        Stream JSONL lines from S3 with progress tracking.
        Yields parsed JSON dicts one at a time.
        """
        for key in keys:
            self._log(f"⬇️ Streaming: {key}")
            lines_yielded = 0
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=key)
                body = response['Body']
                # Stream line by line to avoid loading entire file into RAM
                buffer = ""
                for chunk in body.iter_chunks(chunk_size=8192):
                    buffer += chunk.decode('utf-8', errors='replace')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parsed = json.loads(line)
                            lines_yielded += 1
                            if lines_yielded % 10000 == 0:
                                self._log(f"   ... {lines_yielded:,} lines from {key.split('/')[-1]}")
                            yield parsed
                        except json.JSONDecodeError:
                            continue
                # Handle any remaining buffer
                if buffer.strip():
                    try:
                        yield json.loads(buffer.strip())
                        lines_yielded += 1
                    except json.JSONDecodeError:
                        pass
                self._log(f"✅ Completed: {key} — {lines_yielded:,} lines")
                self.total_lines += lines_yielded
            except ClientError as e:
                self._log(f"❌ Failed to stream {key}: {e}")

    def download_to_local(self, key: str, local_dir: str) -> str:
        """Download a single file to local disk."""
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(key))
        if os.path.exists(local_path):
            self._log(f"📁 Already cached: {local_path}")
            return local_path
        self._log(f"💾 Downloading to {local_path}...")
        try:
            self.s3.download_file(self.bucket, key, local_path)
            self._log(f"✅ Saved: {local_path}")
            return local_path
        except ClientError as e:
            self._log(f"❌ Download failed: {e}")
            return ""


# ─── DATA RESOLUTION ──────────────────────────────────────────────────────────

def resolve_data_sources(config: MyceliaConfig) -> Tuple[List[str], List[str], S3DataInventory]:
    """
    Resolve training and validation data sources.
    Priority: SageMaker channels > S3 prefixes > fallback files
    Returns: (train_keys, val_keys, inventory_instance)
    """
    inventory = S3DataInventory(config.s3_bucket)
    train_keys, val_keys = [], []

    print("\n" + "=" * 70)
    print("🔍 RESOLVING DATA SOURCES")
    print("=" * 70)

    # ── Priority 1: SageMaker Channels ───────────────────────────────────────
    if IN_SAGEMAKER and SAGEMAKER_CHANNEL and os.path.exists(SAGEMAKER_CHANNEL):
        print(f"   🏭 SageMaker channel detected: {SAGEMAKER_CHANNEL}")
        # SageMaker mounts channels as local directories
        train_local = SAGEMAKER_CHANNEL
        val_local = SAGEMAKER_VALIDATION if (SAGEMAKER_VALIDATION and os.path.exists(SAGEMAKER_VALIDATION)) else None

        def scan_local_dir(path: str) -> List[str]:
            files = []
            for root, _, filenames in os.walk(path):
                for f in filenames:
                    if f.endswith('.jsonl') or f.endswith('.json'):
                        files.append(os.path.join(root, f))
            return sorted(files)

        train_files = scan_local_dir(train_local)
        print(f"   ✅ Train files in channel: {len(train_files)}")
        for f in train_files[:5]:
            print(f"      • {os.path.basename(f)}")
        if len(train_files) > 5:
            print(f"      ... and {len(train_files)-5} more")

        # For SageMaker, we return local file paths as "keys" (special handling in loader)
        train_keys = train_files
        if val_local:
            val_keys = scan_local_dir(val_local)
        print(f"   ✅ Validation files: {len(val_keys)}")

        # Build pseudo-inventory
        for f in train_files:
            inventory.inventory.append({
                'key': f, 'size_mb': os.path.getsize(f)/(1024*1024),
                'type': 'jsonl', 'status': 'sagemaker_channel'
            })
        return train_keys, val_keys, inventory

    # ── Priority 2: S3 Prefixes ──────────────────────────────────────────────
    print(f"   🌐 Checking S3 prefixes...")
    train_objs = inventory.list_prefix(config.s3_prefix_train)
    val_objs = inventory.list_prefix(config.s3_prefix_val)

    train_keys = [o['key'] for o in train_objs if o['type'] == 'jsonl']
    val_keys = [o['key'] for o in val_objs if o['type'] == 'jsonl']

    # ── Priority 3: Fallback files ───────────────────────────────────────────
    if not train_keys:
        print(f"   ⚠️ No train data in prefixes. Checking fallbacks...")
        fallback_objs = inventory.list_fallback(config.s3_fallback_files)
        train_keys = [o['key'] for o in fallback_objs if o['type'] == 'jsonl']

    inventory.print_inventory()

    if not train_keys:
        raise RuntimeError("❌ CRITICAL: No training data found in SageMaker channels, S3 prefixes, or fallbacks!")

    print(f"\n   📊 Training sources: {len(train_keys)} file(s)")
    print(f"   📊 Validation sources: {len(val_keys)} file(s)")
    return train_keys, val_keys, inventory


# ─── TCM TEXT FORMATTER ───────────────────────────────────────────────────────

class TCMTextFormatter:
    """
    Formats TCM dataset rows into training text.
    Handles ShenNong format (question/answer) and generic formats.
    """
    CHAT_MARKERS = {
        'user_start': '<|im_start|>user\n',
        'user_end': '<|im_end|>\n',
        'assistant_start': '<|im_start|>assistant\n',
        'assistant_end': '<|im_end|>\n',
    }

    def __init__(self, tokenizer, chat_template: bool = True):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def format_row(self, row: Dict) -> str:
        """
        Convert a JSON row to training text.
        Tries common field names for TCM datasets.
        """
        # Try ShenNong / ChatMed format
        question = row.get('question') or row.get('input') or row.get('query') or row.get('prompt', '')
        answer = row.get('answer') or row.get('output') or row.get('response') or row.get('completion', '')
        requirements = row.get('requirements', '')

        # Fallback: single text field
        if not question and not answer:
            text = row.get('text') or row.get('content') or row.get('instruction', '')
            if text:
                return str(text)
            return ""

        # Build formatted text
        if self.chat_template:
            parts = []
            if question:
                parts.append(f"{self.CHAT_MARKERS['user_start']}{question}{self.CHAT_MARKERS['user_end']}")
            if requirements:
                parts.append(f"{self.CHAT_MARKERS['user_start']}[Requirements] {requirements}{self.CHAT_MARKERS['user_end']}")
            if answer:
                parts.append(f"{self.CHAT_MARKERS['assistant_start']}{answer}{self.CHAT_MARKERS['assistant_end']}")
            return "".join(parts)
        else:
            # Simple concatenation for causal LM
            parts = []
            if question:
                parts.append(f"Question: {question}")
            if requirements:
                parts.append(f"Requirements: {requirements}")
            if answer:
                parts.append(f"Answer: {answer}")
            return "\n".join(parts)

    def tokenize(self, text: str, max_length: int) -> Optional[torch.Tensor]:
        """Tokenize text to fixed length tensor."""
        if not text or len(text.strip()) < 10:
            return None
        try:
            tokens = self.tokenizer.encode(text, allowed_special="all")
        except Exception:
            try:
                tokens = self.tokenizer.encode(text)
            except Exception:
                return None

        target = max_length + 1
        if len(tokens) > target:
            # Random crop for training variety
            start = random.randint(0, len(tokens) - target)
            tokens = tokens[start:start + target]
        elif len(tokens) < target:
            pad_id = self.tokenizer.pad_token_id or 0
            tokens = tokens + [pad_id] * (target - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)


# ─── S3 JSONL DATASET ─────────────────────────────────────────────────────────

class S3JSONLDataset(IterableDataset):
    """
    Streams JSONL from S3 (or local files) with buffering and formatting.
    """
    def __init__(self, data_keys: List[str], formatter: TCMTextFormatter,
                 inventory: S3DataInventory, max_length: int = 512,
                 buffer_size: int = 1000, is_local: bool = False):
        self.data_keys = data_keys
        self.formatter = formatter
        self.inventory = inventory
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.is_local = is_local  # True for SageMaker channel paths
        self.line_buffer: List[torch.Tensor] = []
        self.stats = {'rows_seen': 0, 'rows_valid': 0, 'buffer_refills': 0}

    def _stream_lines(self) -> Iterator[Dict]:
        """Stream raw JSON lines from S3 or local files."""
        if self.is_local:
            # Local file streaming (SageMaker)
            for path in self.data_keys:
                print(f"   📁 Reading local: {path}")
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue
        else:
            # S3 streaming
            yield from self.inventory.stream_jsonl_lines(self.data_keys)

    def _fill_buffer(self):
        """Fill the token buffer from streamed lines."""
        self.stats['buffer_refills'] += 1
        new_tokens = []
        for row in self._stream_lines():
            self.stats['rows_seen'] += 1
            text = self.formatter.format_row(row)
            tokens = self.formatter.tokenize(text, self.max_length)
            if tokens is not None:
                new_tokens.append(tokens)
                self.stats['rows_valid'] += 1
                if len(new_tokens) >= self.buffer_size:
                    break
        random.shuffle(new_tokens)
        self.line_buffer = new_tokens
        if self.stats['buffer_refills'] == 1 or self.stats['buffer_refills'] % 10 == 0:
            print(f"   🔄 Buffer refill #{self.stats['buffer_refills']}: "
                  f"{self.stats['rows_seen']:,} rows → {self.stats['rows_valid']:,} valid "
                  f"({100*self.stats['rows_valid']/max(1,self.stats['rows_seen']):.1f}%)")

    def __iter__(self):
        while True:
            if not self.line_buffer:
                self._fill_buffer()
                if not self.line_buffer:
                    print("   ⚠️ Buffer empty — dataset exhausted or all rows invalid")
                    # Restart from beginning for infinite training
                    self.stats['rows_seen'] = 0
                    self.stats['rows_valid'] = 0
                    self._fill_buffer()
            yield self.line_buffer.pop()


# ─── CURRICULUM GENERATOR (retained for diversity) ────────────────────────────

class MycelialGenerator:
    def __init__(self):
        self.categories = ["algebra", "recursion", "consensus", "contradiction",
                          "impermanence", "self_reference", "nonduality"]
        self.sample_count = 0

    def generate(self):
        self.sample_count += 1
        cat = random.choice(self.categories)
        if cat == "algebra":
            a = random.randint(1, 50)
            return f"Expand: (x + {a})² = x² + {2*a}x + {a*a}"
        elif cat == "recursion":
            d = random.randint(1, 8)
            cur = "observe"
            lines = []
            for _ in range(d):
                lines.append(cur)
                cur = f"observe {cur}"
            return "\n".join(lines)
        elif cat == "consensus":
            x = random.randint(1, 20)
            return f"Agent A: x={x}\nAgent B: x={x}\nAgent C: x={x}\nConsensus: All agree."
        elif cat == "contradiction":
            x = random.randint(1, 20)
            return f"Agent A: x={x}\nAgent B: x={x+1}\nConsensus: Disagreement detected."
        elif cat == "impermanence":
            s = random.choice(["river", "cloud", "leaf", "shadow", "wave", "flower"])
            return f"The {s} changes.\nThe {s} changes again.\nNo moment of the {s} is identical."
        elif cat == "self_reference":
            d = random.randint(1, 6)
            t = "The observer."
            for _ in range(d):
                t = f"The observer observes: [{t}]"
            return t
        else:
            a, b = random.choice([("wave", "ocean"), ("shadow", "light"),
                                  ("question", "answer"), ("self", "world")])
            return f"The {a} depends on the {b}.\nThe {b} depends on the {a}.\nNeither exists independently."


# ─── HYBRID DATASET (TCM + Curriculum) ────────────────────────────────────────

class HybridS3Dataset(IterableDataset):
    def __init__(self, tokenizer, train_keys, val_keys, inventory,
                 max_length=512, curriculum_ratio=0.1, use_chat_template=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.curriculum_ratio = curriculum_ratio

        self.formatter = TCMTextFormatter(tokenizer, chat_template=use_chat_template)
        is_local = IN_SAGEMAKER and SAGEMAKER_CHANNEL

        self.tcm_train = S3JSONLDataset(
            train_keys, self.formatter, inventory, max_length,
            buffer_size=config.prefetch_buffer_size, is_local=is_local
        )
        self.tcm_val = S3JSONLDataset(
            val_keys, self.formatter, inventory, max_length,
            buffer_size=config.prefetch_buffer_size // 2, is_local=is_local
        ) if val_keys else None

        self.curriculum = MycelialGenerator()
        self.curriculum_cache = []
        print(f"   🌱 Pre-caching curriculum samples...")
        for _ in range(200):
            text = self.curriculum.generate()
            tokens = self.formatter.tokenize(text, max_length)
            if tokens is not None:
                self.curriculum_cache.append(tokens)
        print(f"   ✅ {len(self.curriculum_cache)} curriculum samples ready")

    def __iter__(self):
        while True:
            if random.random() < self.curriculum_ratio:
                # Curriculum sample
                if random.random() < 0.15 and self.curriculum_cache:
                    yield random.choice(self.curriculum_cache)
                else:
                    text = self.curriculum.generate()
                    tokens = self.formatter.tokenize(text, self.max_length)
                    if tokens is not None:
                        yield tokens
            else:
                # TCM sample from S3
                try:
                    yield next(iter(self.tcm_train))
                except StopIteration:
                    continue


# ─── COLLATE ──────────────────────────────────────────────────────────────────

def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)


# ═══════════════════════════════════════════════════════════════════════════════
# MYCELIA ARCHITECTURE (unchanged from v7)
# ═══════════════════════════════════════════════════════════════════════════════

def get_sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
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
        fib_weights = config.fib_weights[:self.n_heads]
        self.register_buffer('fib_weights', torch.tensor(fib_weights, dtype=torch.float32) / sum(fib_weights))

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
        assert x.shape[1] <= 1024, f"Sequence length {x.shape[1]} exceeds max"
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
        self.latent_pos = nn.Parameter(torch.randn(1, 64, config.d_model) * 0.02)

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
        x = self.embedding(input_ids)
        x = x + get_sinusoidal_pe(T, self.config.d_model, x.device)

        if use_compression and T > self.config.compress_window:
            prefix_len = self.config.compress_window
            prefix = x[:, :prefix_len, :]
            suffix = x[:, prefix_len:, :]
            latent = self.compressor(prefix)
            x = torch.cat([latent, suffix], dim=1)
            actual_T = x.shape[1]
        else:
            actual_T = T

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
    def generate(self, prompt: str, max_new_tokens: int = 30, temperature: float = 0.7):
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


# ═══════════════════════════════════════════════════════════════════════════════
# MASSIF TELEMETRY (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class MASSIFTelemetry:
    def __init__(self):
        self.metrics = {
            'step': [], 'loss': [], 'dubito': [], 'coherence': [],
            'persistence': [], 'persistence_avg': [], 'flip_detected': [],
            'flip_count': 0, 'norm': [], 'norm_growth': [],
            'alignment': [], 'curvature': [], 'learning_rate': [],
            'vram_allocated': [], 'timestamp': [], 'compressed': []
        }
        self.flip_history = []
        self.start_time = time.time()
        self.persist_buffer = []
        self.last_hidden = None

    def update(self, step, model, loss, lr, compressed=False):
        timestamp = datetime.now().isoformat()
        hidden = model.get_hidden_states()

        self.metrics['step'].append(step)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        self.metrics['timestamp'].append(timestamp)
        self.metrics['compressed'].append(1 if compressed else 0)

        avg_dubito = np.mean(model.dubito_history[-100:]) if model.dubito_history else 0
        avg_coherence = np.mean(model.consensus_stats[-100:]) if model.consensus_stats else 0
        self.metrics['dubito'].append(avg_dubito)
        self.metrics['coherence'].append(avg_coherence)

        if hidden is not None:
            norm = torch.norm(hidden).item()
            self.metrics['norm'].append(norm)
            if len(self.metrics['norm']) > 1:
                self.metrics['norm_growth'].append(norm - self.metrics['norm'][-2])
            else:
                self.metrics['norm_growth'].append(0.0)

        if hidden is not None and self.last_hidden is not None:
            eps = 1e-8
            h_norm = hidden / (torch.norm(hidden) + eps)
            last_norm = self.last_hidden / (torch.norm(self.last_hidden) + eps)
            v_t = h_norm - last_norm
            v_t_minus = last_norm - self.last_hidden / (torch.norm(self.last_hidden) + eps) if len(self.metrics['norm']) > 2 else v_t

            if torch.norm(v_t) > eps and torch.norm(v_t_minus) > eps:
                I_t = (v_t @ v_t_minus) / (torch.norm(v_t) * torch.norm(v_t_minus) + eps)
                self.metrics['persistence'].append(I_t.item())
                self.persist_buffer.append(I_t.item())
                if len(self.persist_buffer) > 3:
                    self.persist_buffer.pop(0)
                if len(self.persist_buffer) == 3:
                    avg_persist = np.mean(self.persist_buffer)
                    self.metrics['persistence_avg'].append(avg_persist)
                    if avg_persist > 0:
                        self.metrics['flip_detected'].append(1)
                        self.metrics['flip_count'] += 1
                        self.flip_history.append(step)
                    else:
                        self.metrics['flip_detected'].append(0)
            else:
                self.metrics['persistence'].append(0.0)
                self.metrics['persistence_avg'].append(0.0)
                self.metrics['flip_detected'].append(0)
        else:
            self.metrics['persistence'].append(0.0)
            self.metrics['persistence_avg'].append(0.0)
            self.metrics['flip_detected'].append(0)

        self.metrics['alignment'].append(min(1.0, avg_coherence * 1.1))
        if len(self.metrics['persistence']) > 1:
            p_prev = self.metrics['persistence'][-2]
            p_curr = self.metrics['persistence'][-1]
            if p_prev != 0 and p_curr != 0:
                self.metrics['curvature'].append(math.acos(min(1.0, max(-1.0, p_curr))))
            else:
                self.metrics['curvature'].append(0.0)
        else:
            self.metrics['curvature'].append(0.0)

        if torch.cuda.is_available():
            self.metrics['vram_allocated'].append(torch.cuda.memory_allocated(0) / 1e9)
        else:
            self.metrics['vram_allocated'].append(0.0)

        if hidden is not None:
            self.last_hidden = hidden.detach().clone()

    def get_summary(self):
        if not self.metrics['step']:
            return {}
        return {
            'total_steps': len(self.metrics['step']),
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else None,
            'avg_loss_100': np.mean(self.metrics['loss'][-100:]) if len(self.metrics['loss']) >= 100 else None,
            'final_dubito': self.metrics['dubito'][-1] if self.metrics['dubito'] else None,
            'final_coherence': self.metrics['coherence'][-1] if self.metrics['coherence'] else None,
            'flip_count': self.metrics['flip_count'],
            'flip_rate': self.metrics['flip_count'] / len(self.metrics['step']) if self.metrics['step'] else 0,
            'avg_persistence': np.mean(self.metrics['persistence']) if self.metrics['persistence'] else 0,
            'avg_norm': np.mean(self.metrics['norm']) if self.metrics['norm'] else 0,
            'avg_alignment': np.mean(self.metrics['alignment']) if self.metrics['alignment'] else 0,
            'avg_curvature': np.mean(self.metrics['curvature']) if self.metrics['curvature'] else 0,
            'compressed_ratio': np.mean(self.metrics['compressed']) if self.metrics['compressed'] else 0,
        }

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"   📊 MASSIF telemetry saved to {filepath}")

    def print_status(self, step, compressed=False):
        if not self.metrics['step']:
            return
        idx = -1
        comp_marker = " [C]" if compressed else ""
        line = f"  📊 Step 0x{step:05x}{comp_marker} | Loss: {self.metrics['loss'][idx]:.4f}"
        line += f" | Dubito: {self.metrics['dubito'][idx]:.2f} | Coh: {self.metrics['coherence'][idx]:.3f}"
        line += f" | Persist: {self.metrics['persistence'][idx]:.3f} | Flips: {self.metrics['flip_count']}"
        line += f" | ||h||: {self.metrics['norm'][idx]:.2f} | R_t: {self.metrics['alignment'][idx]:.3f}"
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = os.environ.get('SM_MODEL_DIR', "/content/drive/MyDrive/mycelia_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "mycelia_latest.pt")

SAVE_EVERY = 500
ACCUMULATION_STEPS = 4
CURRENT_DATASET = "tcm_shennong_s3"

def to_hex(n: int, digits: int = 5) -> str:
    return f"{n:0{digits}x}"

def from_hex(h: str) -> int:
    return int(h, 16)

def save_checkpoint(model, optimizer, global_step, losses, dataset_tag, filepath):
    checkpoint = {
        'global_step': global_step,
        'dataset_tag': dataset_tag,
        'config': model.config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1] if losses else None,
        'avg_loss_100': float(np.mean(losses[-100:])) if len(losses) >= 100 else None,
        'timestamp': datetime.now().isoformat(),
    }
    temp_path = filepath + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, filepath)
    return filepath

def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        return None, 0
    print(f"⏳ Found checkpoint at {filepath}! Loading...")
    checkpoint = torch.load(filepath, map_location=device)

    if 'config' in checkpoint:
        old_cfg = checkpoint['config']
        if hasattr(old_cfg, 'max_seq_len') and old_cfg.max_seq_len != config.max_seq_len:
            print(f"⚠️ Checkpoint seq_len {old_cfg.max_seq_len} != current {config.max_seq_len}")
            print("   🗑️ Incompatible checkpoint - starting fresh")
            return None, 0

    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"   🌱 Fresh init for: {[k.split('.')[0] for k in missing[:3]]}...")
    if unexpected:
        print(f"   🗑️ Ignored old keys: {[k.split('.')[0] for k in unexpected[:3]]}...")

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint.get('global_step', 0)
    dataset_tag = checkpoint.get('dataset_tag', 'unknown')
    print(f"🚀 Resuming from global step {global_step} (0x{to_hex(global_step)}) | Dataset: {dataset_tag}")
    return checkpoint, global_step

def cleanup_old_checkpoints(keep_last: int = 5):
    checkpoints = []
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("mycelia_step_") and f.endswith(".pt"):
            try:
                hex_part = f.split("_")[2].split(".")[0]
                step_num = from_hex(hex_part)
                checkpoints.append((step_num, f))
            except (ValueError, IndexError):
                continue
    checkpoints.sort()
    for _, old_file in checkpoints[:-keep_last]:
        os.remove(os.path.join(CHECKPOINT_DIR, old_file))
        print(f"   🗑️ Cleaned up old checkpoint: {old_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n🔧 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    print("   ✅ Qwen tokenizer loaded!")
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
VOCAB_SIZE = tokenizer.vocab_size
config.vocab_size = VOCAB_SIZE
print(f"   ✅ Vocab size: {VOCAB_SIZE} | pad_token: {tokenizer.pad_token}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SETUP & TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🌿 MYCELIA LM v7.1 — S3-AWARE TRAINING SETUP")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Resolve data sources ─────────────────────────────────────────────────────
print("\n📡 Resolving data sources...")
train_keys, val_keys, inventory = resolve_data_sources(config)

# ── Create model ─────────────────────────────────────────────────────────────
print("\nCreating model...")
model = MyceliaLM(config).to(device)
print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} params")

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-6)
scaler = GradScaler()

# ── Load checkpoint ──────────────────────────────────────────────────────────
checkpoint, global_step = load_checkpoint(model, optimizer, LATEST_PATH)
if checkpoint is None:
    print("🌱 No compatible checkpoint found. Starting from scratch.")
    global_step = 0

# ── Create dataset & dataloader ──────────────────────────────────────────────
print("\nCreating S3-aware hybrid dataset...")
dataset = HybridS3Dataset(
    tokenizer, train_keys, val_keys, inventory,
    max_length=config.max_seq_len,
    curriculum_ratio=0.1,
    use_chat_template=True
)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)

massif = MASSIFTelemetry()

print(f"\n🚀 Training configuration:")
print(f"   Global step starts at: {global_step} (0x{to_hex(global_step)})")
print(f"   Compression: {config.compress_ratio}x every {config.compress_freq} steps")
print(f"   Curriculum: {dataset.curriculum_ratio*100}%")
print(f"   S3 train sources: {len(train_keys)}")
print(f"   S3 val sources: {len(val_keys)}")
print(f"   Seq length: {config.max_seq_len}")
print(f"   Effective batch: {8 * ACCUMULATION_STEPS}")
print(f"   Mixed precision: ✅ AMP\n")

# ── Training loop ────────────────────────────────────────────────────────────
model.train()
losses = []
local_step = 0
max_steps = 5000

progress = tqdm(range(max_steps), desc="🍄 Mycelia v7.1", position=0, leave=True)

for local_step in progress:
    try:
        batch = next(iter(dataloader))
        input_ids = batch.to(device)
        if input_ids.shape[1] < 3:
            continue

        expected_len = config.max_seq_len + 1
        assert input_ids.shape[1] == expected_len, f"Batch shape: {input_ids.shape[1]} != {expected_len}"

        inputs = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        use_compression = (config.use_compression and (local_step % config.compress_freq == 0) and local_step > 0)

        with autocast():
            logits = model(inputs, use_compression=use_compression, log_during_train=True)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID
            )
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (local_step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        losses.append(loss.item() * ACCUMULATION_STEPS)
        global_step += 1
        current_lr = scheduler.get_last_lr()[0]
        massif.update(global_step, model, loss.item() * ACCUMULATION_STEPS, current_lr, compressed=use_compression)

        if local_step % 10 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else loss.item()
            progress.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'global': f'0x{to_hex(global_step)}',
                'flips': massif.metrics['flip_count'],
                'comp': 'C' if use_compression else '-'
            })

        if local_step % 100 == 0 and local_step > 0:
            massif.print_status(global_step, compressed=use_compression)
            if torch.cuda.is_available():
                print(f"     VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")

        if local_step % SAVE_EVERY == 0 and local_step > 0:
            hex_step = to_hex(global_step)
            step_path = os.path.join(CHECKPOINT_DIR, f"mycelia_step_{hex_step}.pt")
            save_checkpoint(model, optimizer, global_step, losses, CURRENT_DATASET, step_path)
            save_checkpoint(model, optimizer, global_step, losses, CURRENT_DATASET, LATEST_PATH)
            cleanup_old_checkpoints(keep_last=5)
            print(f"💾 Checkpoint saved at global step {global_step} (0x{hex_step})")

        if local_step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        break
    except Exception as e:
        print(f"\n  ❌ Error at local step {local_step}, global {global_step}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ── Final save ───────────────────────────────────────────────────────────────
print("\n💾 Saving final checkpoint...")
hex_step = to_hex(global_step)
step_path = os.path.join(CHECKPOINT_DIR, f"mycelia_step_{hex_step}.pt")
save_checkpoint(model, optimizer, global_step, losses, CURRENT_DATASET, step_path)
save_checkpoint(model, optimizer, global_step, losses, CURRENT_DATASET, LATEST_PATH)
cleanup_old_checkpoints(keep_last=5)
print(f"✅ Final checkpoint saved at global step {global_step} (0x{hex_step})")

# ── Save model & telemetry ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"   Local steps completed: {local_step + 1}")
print(f"   Global step: {global_step} (0x{to_hex(global_step)})")
if losses:
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Average loss (last 100): {np.mean(losses[-100:]):.4f}" if len(losses) >= 100 else "")

summary = massif.get_summary()
print(f"\n📊 MASSIF Telemetry Summary:")
for k, v in summary.items():
    print(f"   {k}: {v}")

# Save outputs
output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '.')
model_path = os.path.join(output_dir, "mycelia_model_v7_1.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'loss': losses[-1] if losses else None,
    'massif_summary': summary
}, model_path)
print(f"\n   ✅ Model saved to {model_path}")

telemetry_path = os.path.join(output_dir, "massif_telemetry_v7_1.json")
massif.save(telemetry_path)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🎯 TEST GENERATION")
print("=" * 80)

model.eval()

test_prompts = [
    "我腹痛，没有其他症状，有什么中药可以推荐吗？",
    "The mycelial network reaches consensus when",
    "患者发热恶寒，头痛身疼，舌苔薄白，脉浮紧",
    "The observer observes that",
]

for prompt in test_prompts:
    print(f"\n📖 Prompt: '{prompt[:60]}...'")
    output = model.generate(prompt, max_new_tokens=60, temperature=0.7)
    print(f"   → {output[:200]}...")

print("\n" + "=" * 80)
print("🍄 Mycelia v7.1: S3 TCM pipeline active. ShenNong corpus streaming.")
print("=" * 80)