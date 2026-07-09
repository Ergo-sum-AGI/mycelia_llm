#!/usr/bin/env python
# mycelia_landscape_mapper.py - Map the hyperdimensional landscape of MyceliaLM

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from MYCELIA_architecture import MyceliaLM, MyceliaConfig
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

CKPT_PATH = "/home/ec2-user/SageMaker/mycelia_checkpoints/mycelia_latest.pt"
OUTPUT_DIR = "/home/ec2-user/SageMaker/landscape_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("🧬 MYCELIA LANDSCAPE MAPPER")
print("   Mapping the hyperdimensional topology")
print("="*70)

print("\n📂 Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(CKPT_PATH, map_location=device)
config = MyceliaConfig()
model = MyceliaLM(config).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ─── LOAD TOKENIZER ─────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── PROBE PROMPTS ──────────────────────────────────────────────────────────

test_prompts = [
    "The nature of consciousness is",
    "In traditional Chinese medicine, Qi is",
    "According to Stanford philosophy",
    "The mind-body problem concerns",
]

# ─── EXTRACT ACTIVATIONS ───────────────────────────────────────────────────

def extract_activations(model, input_ids, prompt_text):
    """Extract neuron activations across all layers."""
    
    activations = []
    
    def hook_fn(module, input, output):
        # MycelialBlock returns (x, info) - a tuple
        # The first element is the tensor we want
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach().cpu())
    
    # Register hooks on each block
    hooks = []
    for block in model.blocks:
        hook = block.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids, use_compression=False, log_during_train=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations

# ─── EXTRACT ATTENTION PATTERNS ────────────────────────────────────────────

def get_attention_patterns(model, input_ids):
    """Extract attention patterns from all heads."""
    patterns = []
    
    def attn_hook(module, input, output):
        # MycelialAttention returns (out, head_outputs) - a tuple
        # The second element is the head outputs we want
        if isinstance(output, tuple):
            output = output[1]  # head_outputs
        patterns.append(output.detach().cpu())
    
    # Register hook on the attention module of the last block
    last_block = model.blocks[-1]
    hook = last_block.attn.register_forward_hook(attn_hook)
    
    with torch.no_grad():
        _ = model(input_ids, use_compression=False, log_during_train=False)
    
    hook.remove()
    return patterns

# ─── ANALYZE LANDSCAPE ─────────────────────────────────────────────────────

all_layer_data = []
all_prompt_names = []

for prompt in test_prompts:
    print(f"\n📝 Analyzing: {prompt[:40]}...")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Extract activations
    layer_acts = extract_activations(model, input_ids, prompt)
    
    all_layer_data.append(layer_acts)
    all_prompt_names.append(prompt[:30] + "...")

# ─── COMPUTE LANDSCAPE STATISTICS ──────────────────────────────────────────

print("\n" + "="*70)
print("📊 HYPERDIMENSIONAL LANDSCAPE STATISTICS")
print("="*70)

for prompt_idx, (layer_acts, prompt_name) in enumerate(zip(all_layer_data, all_prompt_names)):
    print(f"\n📝 Prompt: {prompt_name}")
    print(f"   Layers captured: {len(layer_acts)}")
    
    for layer_idx, act in enumerate(layer_acts):
        # Shape: (B, T, D) -> (T, D)
        act_flat = act.squeeze(0).numpy()
        
        # Compute statistics
        mean_act = np.mean(act_flat)
        std_act = np.std(act_flat)
        max_act = np.max(act_flat)
        min_act = np.min(act_flat)
        
        # Sparsity (percentage of near-zero activations)
        sparsity = np.mean(np.abs(act_flat) < 0.01) * 100
        
        # Entropy of activation distribution
        hist, _ = np.histogram(act_flat, bins=50)
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        
        # Correlation with previous layer (if not first)
        if layer_idx > 0:
            prev_act = all_layer_data[prompt_idx][layer_idx-1].squeeze(0).numpy()
            flat_prev = prev_act.flatten()
            flat_curr = act_flat.flatten()
            min_len = min(len(flat_prev), len(flat_curr))
            corr = np.corrcoef(flat_prev[:min_len], flat_curr[:min_len])[0, 1]
        else:
            corr = 0.0
        
        print(f"\n   Layer {layer_idx+1}:")
        print(f"      Mean: {mean_act:.4f} | Std: {std_act:.4f}")
        print(f"      Range: [{min_act:.4f}, {max_act:.4f}]")
        print(f"      Sparsity: {sparsity:.1f}%")
        print(f"      Entropy: {entropy:.4f}")
        print(f"      Correlation with prev layer: {corr:.4f}")

# ─── CLUSTER ANALYSIS ──────────────────────────────────────────────────────

print("\n" + "="*70)
print("🧩 CLUSTER ANALYSIS (Neuron Topology)")
print("="*70)

for prompt_idx, (layer_acts, prompt_name) in enumerate(zip(all_layer_data, all_prompt_names)):
    # Use the last layer for clustering
    last_act = layer_acts[-1].squeeze(0).numpy()  # (T, D)
    
    # Flatten tokens for clustering
    X = last_act  # (T, D)
    
    if X.shape[0] < 2:
        print(f"\n⚠️ Not enough tokens for clustering on: {prompt_name}")
        continue
    
    # K-means clustering
    n_clusters = min(3, X.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Compute cluster statistics
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    cluster_variances = []
    for i in range(n_clusters):
        if np.sum(labels == i) > 0:
            cluster_var = np.var(X[labels == i], axis=0).mean()
            cluster_variances.append(cluster_var)
    
    print(f"\n📝 Prompt: {prompt_name}")
    print(f"   Tokens: {X.shape[0]}")
    print(f"   Clusters: {n_clusters}")
    print(f"   Cluster sizes: {cluster_sizes}")
    print(f"   Cluster variances: {[f'{v:.4f}' for v in cluster_variances]}")
    print(f"   Inertia (compactness): {kmeans.inertia_:.4f}")

# ─── PCA PROJECTION ──────────────────────────────────────────────────────

print("\n" + "="*70)
print("📉 PCA PROJECTION (2D Visualization)")
print("="*70)

# Combine all layer activations for PCA
all_activations = []
layer_labels = []
prompt_labels = []

for prompt_idx, (layer_acts, prompt_name) in enumerate(zip(all_layer_data, all_prompt_names)):
    for layer_idx, act in enumerate(layer_acts):
        act_flat = act.squeeze(0).numpy().flatten()  # (T*D,)
        all_activations.append(act_flat)
        layer_labels.append(layer_idx)
        prompt_labels.append(prompt_idx)

# Pad to same length
max_len = max(len(a) for a in all_activations)
padded_acts = np.array([np.pad(a, (0, max_len - len(a)), 'constant') for a in all_activations])

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(padded_acts)

print(f"\n   PCA explained variance: {pca.explained_variance_ratio_}")
print(f"   Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# TSNE (optional, slower)
print("\n   Running t-SNE (may take a moment)...")
tsne = TSNE(n_components=2, perplexity=min(30, len(padded_acts)-1), random_state=42)
tsne_result = tsne.fit_transform(padded_acts)

# ─── HEAD ATTENTION PATTERNS ──────────────────────────────────────────────

print("\n" + "="*70)
print("👁️ ATTENTION HEAD TOPOLOGY")
print("="*70)

def compute_attention_entropy(attn_matrix):
    """Compute entropy of attention distribution with robust handling."""
    eps = 1e-10
    attn_safe = attn_matrix + eps
    row_sums = attn_safe.sum(axis=1, keepdims=True)
    attn_normalized = attn_safe / row_sums
    entropy = -np.sum(attn_normalized * np.log(attn_normalized + 1e-12))
    return entropy

for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attn_patterns = get_attention_patterns(model, input_ids)
    
    if attn_patterns:
        attn = attn_patterns[0]  # (B, n_heads, T, T)
        attn_mean = attn.mean(dim=(0, 1)).numpy()  # (T, T)
        attn_std = attn.std(dim=(0, 1)).numpy()
        
        print(f"\n📝 Prompt: {prompt[:30]}...")
        print(f"   Attention matrix shape: {attn.shape}")
        
        # ← FIXED: Robust entropy calculation
        attn_entropy = compute_attention_entropy(attn_mean)
        print(f"   Mean attention entropy: {attn_entropy:.4f}")
        print(f"   Attention sparsity: {np.mean(attn_mean < 0.01) * 100:.1f}%")
        print(f"   Diagonality (self-attention): {np.mean(np.diag(attn_mean)):.4f}")