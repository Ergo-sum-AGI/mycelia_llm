# Mapping Mycelia: A Dynamical Systems Case Study in Real-Time AI Safety
## How Watching a Model's "Thought Process" Saved It From Collapse

By Daniel Solis, DUBITO Inc. / Ergo Sum AGI Safety Systems

---

## 1. The Mycelia Hypothesis: A Model Built to Be Watched

Most language models are trained to optimize a single number: cross-entropy loss. The assumption is that if loss goes down, the model is learning. But loss is an aggregate statistic. It tells you nothing about *how* the model is learning — whether its internal dynamics are stable, oscillatory, or heading toward collapse.

Mycelia LM was built on a different hypothesis: **what if we trained a model whose internal geometry we could monitor, diagnose, and intervene upon in real time?**

Mycelia is a 181-million-parameter transformer with a custom consensus mechanism called MycelialConsensus. Unlike standard attention, which simply averages head outputs, MycelialConsensus uses Fibonacci-weighted aggregation, per-token variance tracking, and dynamic thresholding to decide which tokens should pass through and which should be attenuated. The model reports its own internal state — coherence, variance, acclamation rate, friction gradient — at every forward pass.

This is not interpretability as post-hoc analysis. This is **interpretability as architecture**.

---

---

## 1.5 What Mycelia Actually Is: A Consensus Machine, Not an Attention Machine

To understand what follows, you need to know what Mycelia is *not*.

Mycelia is not a standard transformer with a monitoring dashboard attached. It is an architecture where the fundamental building block — the attention mechanism — has been replaced by something different: **MycelialConsensus**.

### Standard Transformer Attention
In a normal transformer, each layer has multiple attention heads. Each head computes query-key-value projections, calculates similarity scores, applies softmax, and produces a weighted average of values. The outputs of all heads are concatenated and projected back to the model dimension. The heads are trained end-to-end via backpropagation, and their behavior emerges from optimization. There is no explicit mechanism enforcing diversity, disagreement, or consensus.

### MycelialConsensus
Mycelia replaces this with a **consensus mechanism**:

1. **Fibonacci-weighted aggregation:** Each of the 8 heads is assigned a Fibonacci weight (5, 8, 13, 21, 34, 55, 89, 144). The consensus output is a weighted sum of head outputs, normalized by the sum of weights. This is not learned — it is hardcoded. The Fibonacci sequence was chosen because it provides a natural gradient of influence: early heads (low weight) can dissent without dominating; late heads (high weight) anchor the consensus.

2. **Per-token variance tracking:** For every token position, Mycelia computes the variance across all 8 head outputs. High variance means the heads disagree strongly about that token. Low variance means they agree.

3. **Dynamic thresholding:** A threshold is computed from the variance distribution itself (using robust statistics — median absolute deviation). Tokens with variance *below* threshold are **acclaimed** (pass through unchanged). Tokens with variance *above* threshold are **vetoed** (attenuated by 0.85×).

4. **Adaptive mixing:** The final layer output is a mixture of attention and consensus: `0.9 × attention_out + 0.1 × consensus_out`. The consensus modulates the attention, not replaces it entirely.

5. **Multi-round consensus:** In v8.1, the consensus can run for multiple rounds (default: 2). Each round shifts the mixing ratio — round 0 uses 90% attention, round 1 uses 85%, etc. — allowing deeper consensus formation without collapsing signal.

### Why This Matters
The consensus mechanism is not a training trick. It is a **dynamical regulator** — a geometric stability controller built into every forward pass. It enforces two properties that standard transformers lack:

- **Disagreement detection:** The model knows when its own heads disagree, and it can suppress tokens that cause disagreement.
- **Self-tuning thresholds:** The threshold adapts to the model's current variance distribution, preventing the "everything gets vetoed" or "everything passes" pathologies.

This is why Mycelia can report its own internal state. The consensus mechanism *is* the telemetry. It does not just produce text — it produces a **diagnostic signature** at every layer, every token, every step.

### The Vocabulary You Will See

| Term | Meaning |
|------|---------|
| **Acclamation** | A token passes the consensus (variance < threshold) |
| **Veto** | A token is attenuated (variance ≥ threshold) |
| **Coherence** | Fraction of tokens acclaimed (0.0 = all vetoed, 1.0 = all acclaimed) |
| **Variance** | Disagreement among heads for a given token |
| **Threshold** | The cutoff between acclamation and veto, computed from variance distribution |
| **Delta** | `early_var - late_var` — whether variance grows or shrinks across layers |
| **Friction** | Dynamical regime: HARMONIZED, DISSIPATED, DEEP DRIFT, etc. |

Now you know what Mycelia is. The rest of this paper is about what we learned from watching it.

---

## 1.6 The Consensus Flow: A Visual Walkthrough

Here is what happens inside a single Mycelia layer during one forward pass. Compare this to standard transformer attention.

### Standard Transformer (One Layer, One Token)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  QKV Projection: W_qkv · h_t        │
        │  → q_t, k_t, v_t ∈ ℝ^512            │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  Split into 8 heads:                │
        │  q_t^(1), ..., q_t^(8) ∈ ℝ^64       │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────────────────────────┐
        │  Each head: softmax(q_t^(i)·k_t^(i)^T / √64) · v_t^(i)  │
        │  → head_output_i ∈ ℝ^64                                 │
        └─────────────────────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────┐
        │  Concatenate: [head_1 | ... | head_8] ∈ ℝ^512  │
        │  Project: W_o · concat → output ∈ ℝ^512        │
        └────────────────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + output (residual)
```

**What is missing:** No mechanism detects disagreement between heads. No token-level regulation exists. The heads are trained end-to-end, and their behavior is emergent, not governed.

### Mycelia Layer (One Token, One Consensus Round)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  SAME QKV Projection + Attention    │
        │  → head_outputs ∈ ℝ^(8×64)          │
        └─────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────┐
        │  FIBONACCI CONSENSUS                           │
        │                                                │
        │  Weights: w = [5, 8, 13, 21, 34, 55, 89, 144]  │
        │  Normalized: w_i / Σw = [0.012, 0.019, 0.031,  │
        │                         0.050, 0.081, 0.131,   │
        │                         0.212, 0.344]          │
        │                                                │
        │  consensus = Σ_i (w_i/Σw) · head_i             │
        │  → consensus ∈ ℝ^64                            │
        └────────────────────────────────────────────────┘
                      ↓
        ┌───────────────────────────────────────────┐
        │  VARIANCE TRACKING (per token)            │
        │                                           │
        │  For each position t:                     │
        │    μ_t = mean(head_1(t), ..., head_8(t))  │
        │    σ²_t = mean((head_i(t) - μ_t)²)        │
        │    → token_variance(t) = mean(σ²_t)       │
        └───────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────┐
        │  DYNAMIC THRESHOLD                         │
        │                                            │
        │  flat_var = token_variance.flatten()       │
        │  median = flat_var.median()                │
        │  mad = (flat_var - median).abs().median()  │
        │  scale = median + 1.4826 × mad             │
        │  threshold = 1.5 × scale × layer_factor    │
        │  (layer_factor: 0.8 → 1.4 across layers)   │
        └────────────────────────────────────────────┘
                      ↓
        ┌──────────────────────────────────────────────────┐
        │  ACCLAMATION vs. VETO                            │
        │                                                  │
        │  For each token t:                               │
        │    if variance(t) < threshold:                   │
        │       → ACCLAIMED (pass through)                 │
        │       veto_factor = 1.0                          │
        │    else:                                         │
        │       → VETOED (attenuate)                       │
        │       veto_factor = 0.85                         │
        │                                                  │
        │  consensus_attenuated = consensus × veto_factor  │
        └──────────────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────────┐
        │  ADAPTIVE MIXING                                   │
        │                                                    │
        │  Round 0: output = 0.90 × attn + 0.10 × consensus  │
        │  Round 1: output = 0.85 × attn + 0.15 × consensus  │
        │  Round 2: output = 0.80 × attn + 0.20 × consensus  │
        │  ... (floor at 0.50 × attn)                        │
        └────────────────────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + α_attn × output (residual)
```

**What is different:** Every token is evaluated for consensus agreement. Disagreeing tokens are attenuated, not suppressed. The threshold adapts to the model's own variance distribution. The mixing ratio shifts per round, allowing deeper consensus without signal collapse.

### The Fibonacci Weights: Why This Sequence?

The Fibonacci sequence (5, 8, 13, 21, 34, 55, 89, 144) was not chosen for mysticism. It was chosen for **mathematical properties**:

1. **Exponential growth:** Each weight is ~1.618× the previous. This creates a natural gradient where late heads dominate the consensus, but early heads still contribute meaningfully.

2. **No learned parameters:** The weights are fixed. This means the consensus mechanism cannot overfit or collapse during training. It is a geometric prior, not a learned behavior.

3. **Checkpoint compatibility:** The weights are stored as a flat buffer `(n_heads,)` — no reshaping needed when loading checkpoints with different head counts.

4. **Interpretability:** Because the weights are fixed, we can reason about head importance analytically. Head 8 (weight 144) has 28.8× the influence of Head 1 (weight 5). If Head 8 disagrees with the others, it will dominate the consensus unless vetoed.

### The Veto Factor: Continuous, Not Binary

In early versions, veto was binary: a token either passed or was zeroed out. This created hard discontinuities in the loss landscape. v7.3 introduced the **continuous veto factor**:

```python
veto_factor = acclamation_mask + (1.0 - acclamation_mask) * 0.85
```

This means:
- Acclaimed tokens: multiplied by 1.0 (unchanged)
- Vetoed tokens: multiplied by 0.85 (attenuated, not killed)

The factor is **gradient-safe** — the 0.85 is a constant, so backpropagation flows through both branches. The model learns to reduce variance (avoid veto) without being punished catastrophically for occasional disagreement.

---

---

## 1.7 The Consensus Mechanism: Not Just Our Idea

The replacement of attention with consensus is not speculative. In January 2026, Moushegian et al. published "Stabilizing Transformer Training Through Consensus," demonstrating that consensus mechanisms — formulated as graph-based energy minimization — serve as drop-in replacements for attention that stabilize training across wider learning rate rangesciteweb_search:52#0.

Their key findings:
- **Consensus tolerates higher learning rates** than standard attention, with the difference most pronounced in the high-LR regime
- **Hybrid consensus-attention architectures** preserve attention-level performance while inheriting consensus stability
- **Graph spectral theory** provides the mathematical foundation: consensus acts as a low-pass filter on embedding frequency, smoothing high-frequency instabilities

Mycelia's MycelialConsensus differs from their formulation in several critical ways:

#### Structural Differences

| Feature | Moushegian et al. (2026) | Mycelia LM |
|---------|-------------------------|------------|
| **Graph structure** | Learned or predefined adjacency matrix | Fully connected (all heads interact directly) |
| **Weights** | Learned edge weights via backpropagation | Hardcoded Fibonacci weights (fixed, not learned) |
| **Update rule** | Iterative energy minimization (Laplacian smoothing) | Single-pass weighted aggregation |
| **Regulation** | Graph spectral filtering (low-pass on embeddings) | Variance-based thresholding with continuous veto |
| **Complexity** | O(n²) per layer for graph operations | O(n) per layer (vectorized tensor ops) |
| **Telemetry** | None | Real-time coherence, variance, delta, acclamation rate |

#### Why Mycelia Chose Simplicity

Moushegian et al.'s consensus is **powerful but opaque**. The learned adjacency matrix encodes complex relationships between tokens, but these relationships are not human-interpretable. The energy minimization process is a black box: you input embeddings, you get smoothed embeddings, but you cannot explain why a particular token was smoothed more than another.

Mycelia's consensus is **deliberately simple**:

1. **Fixed weights** mean the consensus mechanism cannot overfit or collapse during training. The Fibonacci prior is a geometric constraint, not a learned behavior.

2. **Variance-based thresholding** provides **per-token interpretability**. You can look at any token and ask: "Did the heads agree or disagree?" The answer is a single number: variance. If variance is below threshold, the token is acclaimed. If above, it is vetoed. This is a **transparent decision rule**, not a learned filter.

3. **Continuous veto** (0.85× attenuation, not zeroing) preserves gradient flow. The model can learn from vetoed tokens without catastrophic suppression.

4. **Real-time telemetry** turns the consensus into a **diagnostic instrument**. Every forward pass produces a full report: coherence, variance, threshold, acclamation rate, friction regime. This is not possible with learned graph structures, where the internal state is encoded in weight matrices that require post-hoc analysis to interpret.

#### The Trade-off

Mycelia trades **theoretical expressiveness** for **operational observability**. Moushegian et al.'s consensus can model complex, non-local relationships between tokens. Mycelia's consensus cannot. But Mycelia can tell you, in real time, whether its consensus is working. Moushegian et al.'s cannot.

This trade-off is intentional. The goal is not to build the most expressive consensus mechanism. The goal is to build a consensus mechanism that **reveals its own state** — a mechanism that is simultaneously a stabilizer and a sensor.

Mycelia's consensus is a **diagnostic instrument** first, a stabilizer second.

---

## 2. The Six Regions of the Mycelia Black Box

The original MASSIF framework divides AI interpretability into six regions. Mycelia was designed to make Region 2 — Dynamical Inference — measurable in production.

| Region | Question | Mycelia's Answer |
|--------|----------|------------------|
| **1. Representation** | What is encoded? | 512D residual stream, 8 orthogonal attention heads, per-layer variance tracking |
| **2. Dynamical Inference** | How does cognition unfold? | `variance_delta = early_var - late_var`, measured every step |
| **3. Emergence** | Why do abilities appear? | Tracked via coherence spikes during LR bursts |
| **4. Decision Formation** | Why this token? | Attention head similarity matrix shows specialization |
| **5. Objective Internalization** | What is it actually optimizing? | Consensus mechanism enforces geometric stability as a secondary objective |
| **6. Generalization** | Why does it generalize? | Monitored via domain friction gradient (Stanford vs. FineWeb) |

Mycelia's unique contribution is in **Region 2**. While other models hide their dynamical state, Mycelia broadcasts it.

---

## 3. The Topology Mapper: Finding the Bug

At step 1,075,000, Mycelia had trained for 28 epochs on a mix of FineWeb-Edu (70%) and Stanford Philosophy (30%). Loss was stable at ~4.9. The MASSIF telemetry cell classified it as **Runaway** — but the model was generating coherent English. Something was wrong with the diagnosis, not the model.

We built the Topology Mapper to look inside.

### 3.1 Layer-Wise Trajectory Geometry: What 512 Dimensions Actually Look Like

To understand what the Topology Mapper measures, you need to understand what a "hidden state" actually is.

#### What Is a Hidden State?

When Mycelia processes a token — say, the word "consciousness" — each of its 6 layers produces a **hidden state**: a list of 512 numbers. Think of it as a coordinate in a 512-dimensional space.

```
Token: "consciousness"
Layer 0 hidden state: [0.23, -1.45, 0.89, ..., 0.12]  ← 512 numbers
Layer 1 hidden state: [0.31, -1.38, 0.95, ..., 0.08]  ← 512 numbers
Layer 2 hidden state: [0.45, -1.22, 1.12, ..., -0.03] ← 512 numbers
...
Layer 5 hidden state: [0.78, -0.89, 1.45, ..., -0.21] ← 512 numbers
```

Each number in this list is a **dimension**. It does not correspond to a human-interpretable feature like "is a noun" or "relates to philosophy." The dimensions are learned, emergent, and distributed. Dimension 47 might encode something about syntactic position, dimension 203 something about semantic category, dimension 389 something about emotional valence — but these are not cleanly separable. The meaning is distributed across the full 512D vector.

#### What Is a Trajectory?

As Mycelia generates text, it produces a sequence of tokens. For each token, each layer produces a hidden state. If we trace the hidden states of a single layer across multiple tokens, we get a **trajectory** — a path through 512-dimensional space.

```
Token 1: "The"      → Layer 2 state: [0.12, 0.34, -0.56, ..., 0.89]
Token 2: "nature"   → Layer 2 state: [0.15, 0.38, -0.51, ..., 0.92]
Token 3: "of"       → Layer 2 state: [0.18, 0.41, -0.48, ..., 0.95]
Token 4: "consciousness" → Layer 2 state: [0.22, 0.45, -0.42, ..., 1.01]
```

Plotting these 512D points is impossible. So the Topology Mapper uses **Principal Component Analysis (PCA)** — a mathematical technique that finds the two directions in the 512D space along which the data varies the most. It projects the 512D points onto these two directions, creating a 2D plot we can visualize.

#### What Is a "Coherent Path"?

In a healthy model, the trajectory should form a **smooth, continuous curve** in this 2D projection. This means the model's internal state is evolving gradually and predictably as it processes each token.

**Coherent path:**
```
    ●───●───●───●───●───●───●
    (smooth, gradual transitions)
```

**Incoherent path:**
```
    ●        ●───●
         ●        ●───●        ●
    (jumps, reversals, discontinuities)
```

A coherent path indicates that the model has learned stable representations. Each new token nudges the state in a consistent direction. An incoherent path indicates instability — the model is lurching between unrelated states, which often produces gibberish or repetition.

#### What Is "Saturated" vs. "Exploding"?

There are two ways a trajectory can be pathological:

**Exploding path:** The hidden state norms grow exponentially. The numbers get larger and larger without bound.

```
Token 1: norm = 12.3
Token 2: norm = 45.6
Token 3: norm = 189.2
Token 4: norm = 743.8  ← growing without bound
```

This is the classic "Runaway" regime. The model's internal state is diverging. The MASSIF framework detects this via `max_norm > 200`.

**Saturated path:** The hidden state norms stay stable, but individual neurons are pinned at extreme values. The model is not growing — it is **stuck**.

```
Token 1: [0.23, -1.45, 0.89, ..., 0.12]  ← norm = 18.2
Token 2: [0.25, -1.48, 0.91, ..., 0.15]  ← norm = 18.4
Token 3: [0.27, -1.50, 0.93, ..., 0.18]  ← norm = 18.6
```

The norm grows slowly (1.01×), but look at the individual values: they are barely changing. The model has learned to set certain neurons to fixed values and leave them there. It is using a small subset of its capacity, not the full 512 dimensions.

#### What Is Norm Growth?

The **norm** (or L2 norm) of a hidden state is its Euclidean length — the square root of the sum of squared values:

```
|h| = √(h₁² + h₂² + ... + h₅₁₂²)
```

**Norm growth** is the ratio of the final norm to the initial norm across a sequence:

```
norm_growth = |h_final| / |h_initial|
```

- `norm_growth = 1.0` → perfectly stable
- `norm_growth = 2.0` → doubled in size
- `norm_growth = 10.0` → Runaway

Mycelia's norm growth was 1.01× — essentially stable. But this stability was deceptive.

#### What Is a Hyperactive Neuron?

A **neuron** in this context is one of the 512 dimensions of the hidden state. A neuron is **hyperactive** if its absolute value exceeds 10:

```
|h_i| > 10  → hyperactive
```

Why 10? In a well-normalized model, hidden state values typically range between -3 and +3 (roughly ±3 standard deviations from zero). A value of 10 is **3× larger than expected** — it indicates the neuron is pinned at an extreme value, not participating in normal dynamics.

In Mycelia, 60% of the 3,072 neurons (512 dimensions × 6 layers) were hyperactive. That is 1,850 neurons stuck at values like -12.4, +15.7, -11.2, +18.3. These neurons were not responding to input changes. They were **constants**, not variables.

#### What Is Capacity Collapse?

**Capacity collapse** means the model has stopped using its full representational capacity. Instead of 512 dynamic dimensions, it is using only ~200. The other ~300 dimensions are pinned at fixed values and do not change between tokens.

Think of it like a piano where 60% of the keys are stuck — they play the same note regardless of which key you press. The piano still makes sound, but its expressive range is severely limited.

This is different from Runaway. In Runaway, the piano gets louder and louder until the speakers blow. In capacity collapse, the piano plays at normal volume but with most keys stuck. The MASSIF framework classified Mycelia as Runaway because of the high `max_norm` (2326), but the Topology Mapper revealed the truth: **the model was saturated, not exploding.**

The high `max_norm` came from a few extreme outliers, not systematic growth. The median norm was stable. The pathology was **stagnation**, not divergence.

### 3.2 Attention Head Clustering: What the Q-Projection Matrix Actually Shows

#### What Is an Attention Head?

In Mycelia, each layer has 8 attention heads. Each head is a separate "attention mechanism" that looks at the input sequence and decides which previous tokens are relevant for the current token.

But what does a head actually **learn**? A head learns a **subspace** — a subset of the 512D space that it specializes in. Think of the full 512D space as a large library. Each head is a specialist librarian who only cares about certain sections:

- **Head 1** might specialize in **syntactic position** — where a word appears in the sentence
- **Head 2** might specialize in **semantic category** — whether a word is a noun, verb, or adjective
- **Head 3** might specialize in **coreference** — which pronoun refers to which noun
- **Head 4** might specialize in **long-range dependencies** — connecting words far apart in the sentence
- **Heads 5-8** might specialize in other linguistic patterns we do not yet fully understand

These specializations are not programmed. They emerge during training. The model discovers that dividing the 512D space into 8 subspaces is an efficient way to process language.

#### What Is the Q-Projection Similarity Matrix?

Each head has a **Q projection matrix** — a learned transformation that maps the 512D hidden state into a 64D query vector (512 ÷ 8 heads = 64 dimensions per head).

The Q-projection similarity matrix compares these transformations across heads. It asks: **"Do two heads look at the input in the same way?"**

```
        Head 0  Head 1  Head 2  Head 3  Head 4  Head 5  Head 6  Head 7
Head 0   1.00   +0.01   -0.01   +0.01   +0.00   +0.05   +0.00   -0.01
Head 1   +0.01   1.00   -0.01   +0.00   +0.00   -0.01   +0.05   +0.01
Head 2   -0.01   -0.01   1.00   +0.02   +0.03   +0.00   +0.01   +0.04
...
```

The diagonal is always 1.00 (a head is perfectly similar to itself). The off-diagonal values measure similarity between different heads.

#### What Does "Orthogonal" Mean?

**Orthogonal** means "at right angles" — mathematically, it means the dot product is zero. In the context of attention heads, it means the heads have learned **completely different specializations**.

**Why is this remarkable?** In standard transformers, attention heads often learn **redundant** specializations. Multiple heads might all learn to track syntactic position, or all learn to handle coreference. This is inefficient — the model is using 8 heads to do the work of 3.

Mycelia's heads, by contrast, were all **unique**. The off-diagonal similarities were near zero (±0.01 to ±0.07), meaning each head had carved out its own distinct subspace in the 512D space. No two heads were doing the same job.

This is a direct consequence of the **Fibonacci weighting**. Because the weights are fixed and exponentially spaced (5, 8, 13, 21, 34, 55, 89, 144), the heads cannot afford to be redundant. If two heads learned the same specialization, the higher-weighted head would dominate the consensus, and the lower-weighted head would be ignored. The only stable solution is for each head to find a unique niche.

#### What Is a Subspace?

A **subspace** is a lower-dimensional region of the full 512D space. Think of it like this:

- The full 512D space is like a large warehouse with 512 different attributes (color, size, weight, texture, temperature, etc.)
- A subspace is like a specific aisle in that warehouse — only certain attributes are relevant
- Head 1's subspace might be the "color and texture" aisle
- Head 2's subspace might be the "size and weight" aisle
- Head 3's subspace might be the "temperature and pressure" aisle

When a head processes a token, it projects the full 512D vector into its 64D subspace. It only pays attention to the dimensions relevant to its specialization. The consensus mechanism then recombines these subspace projections using the Fibonacci weights.

#### Why Orthogonality Broke the Consensus

Here is the paradox: **orthogonality is good for the model, but bad for the consensus mechanism.**

The consensus mechanism detects dissent by measuring **variance across heads**. If all heads agree, variance is low. If they disagree, variance is high. The veto threshold is set based on this variance.

But if the heads are orthogonal — each operating in a completely different subspace — they will **always disagree** at the level of raw values. Head 1 might output [0.5, -0.3, 0.8] in its subspace. Head 2 might output [-0.2, 0.9, -0.1] in its completely different subspace. When we compute variance across these orthogonal vectors, the variance is **artificially high** — not because the heads genuinely disagree about the token, but because they are measuring different things.

The consensus mechanism was designed for a regime where heads overlap partially — where disagreement means genuine uncertainty. In Mycelia's orthogonal regime, **disagreement is the normal state**, not a pathology. The consensus was vetoing tokens because the heads were doing their jobs correctly.

This is the **signal misunderstanding**: the consensus treated orthogonal specialization as dissent, when it was actually competence. The fix — adaptive MAD-based thresholding — corrected this by scaling the threshold to match the actual variance distribution of orthogonal heads.

### 3.3 The Smoking Gun: Consensus Was Broken

The Mapper's most important finding was in the consensus telemetry:

```
Layer 0: kept=62.5% | coherence=0.0000 | variance=0.44 | threshold=0.40
Layer 1: kept=0.0%  | coherence=0.0000 | variance=1.14 | threshold=0.40
Layer 2: kept=0.0%  | coherence=0.0000 | variance=0.89 | threshold=0.40
Layer 3: kept=4.2%  | coherence=0.0000 | variance=0.79 | threshold=0.40
Layer 4: kept=4.2%  | coherence=0.0000 | variance=1.54 | threshold=0.40
Layer 5: kept=0.0%  | coherence=0.0000 | variance=5.29 | threshold=0.40
```

**The consensus mechanism was vetoing 95-100% of tokens across layers 1-5.** The `coherence = 0.0000` everywhere meant the telemetry was meaningless — it was measuring "everything is above threshold."

Yet the model still generated text because the attention path dominated (0.9× attention + 0.1× consensus). The consensus was **dead weight** — 10% of compute doing nothing useful.

### 3.4 Root Cause: Scale Mismatch

The v8.0 "fix" had lowered `dissenter_threshold` from 2.5 to 2.0, but the dynamic scaling formula still produced thresholds of **0.03-0.40**. Meanwhile, actual per-token variance ranged from **0.4 to 5.3**. The threshold was an **order of magnitude too low** for the model's actual variance distribution.

This is a classic **scale mismatch** — the threshold was designed for a different dynamical regime.

---

## 4. The v8.1 Intervention: Adaptive MAD-Based Thresholding

The fix was to make the threshold **responsive to the actual variance distribution** rather than a hyperparameter guess.

### 4.1 The Math

Instead of:
```python
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40
```

We compute a robust scale estimate from the live variance:
```python
flat_var = token_variance.view(-1)
var_median = flat_var.median()
var_mad = (flat_var - var_median).abs().median()  # Median Absolute Deviation
var_scale = var_median + 1.4826 * var_mad  # Robust std estimate

threshold = 1.5 * var_scale * layer_factor  # Scales with actual distribution
threshold = threshold.clamp(min=0.1, max=10.0)
```

This means:
- If variance is naturally ~1.0, threshold ≈ 1.5 → ~67% of tokens pass
- If variance grows to ~5.0, threshold ≈ 7.5 → still ~67% pass
- The threshold **scales with the model's state**, not a fixed guess

We also fixed the coherence calculation. Instead of the broken clamped ratio:
```python
coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)  # Always 0
```

We use the direct measure:
```python
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### 4.2 The LR Burst: Shaking the Optimizer

At step 1,137,000, the model was stuck at loss ~4.58 with LR=1e-5 (cosine tail). We injected a controlled **LR burst** — 500 steps at peak LR=3e-4 — to shake the optimizer out of its local minimum.

```
🚀 LR BURST: Injecting peak LR=3.00e-04 for 500 steps
🚀 LR BURST ACTIVE: steps 1,137,000 → 1,137,500
```

The burst served as a **catalyst**, not a sustained requirement. It reconfigured the optimizer momentum, allowing the new adaptive consensus to take hold.

---

## 5. The Results: From Broken to Harmonized

### 5.1 Immediate Fix Validation

| Metric | v8.0 (Broken) | v8.1 (Fixed) | Interpretation |
|--------|--------------|--------------|----------------|
| **kept_ratio** | 0-62% | **~97%** | Consensus now functional |
| **coherence** | 0.0000 | **0.9694-0.9766** | Meaningful self-monitoring |
| **Friction** | 🌋 DEEP DRIFT | **🟢 HARMONIZED** | Regime shift achieved |
| **Delta** | -1.05 to -1.57 | **-0.65 to -0.92** | Moving toward DISSIPATED |
| **Loss** | 4.85-5.05 | **4.80-5.06** | Exploring new basin |
| **LR** | 1e-5 (stuck) | **1.43e-4 (decay)** | Active learning resumed |

The coherence jump from 0.0000 → 0.97 is the **signature of the fix**. The adaptive thresholding correctly matched the model's variance scale, allowing the consensus to function as designed.

### 5.2 The HARMONIZED Regime

After the burst, Mycelia entered a **self-sustaining HARMONIZED state**:

```
📊 Step 1,140,000 | Loss: 4.9331 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9709 📈
   Friction: 🟢 HARMONIZED | early=0.68 late=1.55 Δ=-0.87

📊 Step 1,142,000 | Loss: 4.9490 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9710 📈
   Friction: 🟢 HARMONIZED | early=0.70 late=1.56 Δ=-0.86

📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

Notice the pattern:
- **Coherence holds steady at ~0.97** — the consensus is self-regulating
- **Delta stabilizes around -0.87** — not yet DISSIPATED (target: >0), but no longer DEEP DRIFT
- **Loss explores the 4.8-5.0 basin** — the model is learning, not plateaued
- **LR decays smoothly** — no more snap-to-minimum bug

The brief DEEP DRIFT at step 1,138,000 (delta=-1.57) was the **burst transition artifact** — the optimizer settling into the new basin. By step 1,140,000 it re-stabilized.

### 5.3 The Attention Heads Are Still Orthogonal

Post-fix, the Topology Mapper confirmed the head similarity matrix remained near-diagonal. The fix did not collapse head specialization — it **enabled the consensus to respect it**.

---

## 5.4 The Training Journey: A Diagnostic Thriller in Version Numbers

The Mycelia case study is not a single experiment. It is a **diagnostic narrative** spanning multiple versions, each revealing a deeper layer of the pathology.

#
## 6. The Mycelia Safety Monitor in Practice

The original article proposed a dashboard concept. Mycelia implements it in the training log:

```
📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

This is not a mockup. This is the **actual telemetry** from a live training run on an AWS SageMaker T4 instance. Every 1000 steps, Mycelia reports:

| Observable | What It Measures | Safe Range | Alert Threshold |
|------------|------------------|------------|-----------------|
| **Coherence** | Fraction of tokens acclaimed by consensus | 0.3-1.0 | <0.1 (consensus broken) |
| **Friction** | Dynamical regime classification | 🟢 HARMONIZED / ✅ DISSIPATED | 🌋 DEEP DRIFT |
| **Delta** | Domain friction gradient (early_var - late_var) | > -1.0 | < -1.5 (runaway drift) |
| **Early/Late Var** | Per-layer variance distribution | balanced | early << late (signal loss) |

The system does not just monitor — it **intervenes**. When coherence drops below 0.1, the adaptive threshold automatically rescales. When loss plateaus for >50K steps, the LR burst activates. When delta drops below -1.5, the consensus rounds increase.

---

## 7. The Geoffrey Hinton Problem: Mycelia's Response

Geoffrey Hinton has argued that even experts cannot understand what happens inside large neural networks because the systems are too complex. With 181 million parameters and 6 layers of 512 dimensions, Mycelia is tiny by modern standards. Yet even here, tracking every neuron is impossible.

Mycelia's response is different: **we don't track every neuron. We track the pattern of movement.**

The Topology Mapper does not visualize 181M weights. It visualizes:
- Six 512D trajectory projections (PCA)
- One 8×8 head similarity matrix
- Six bar charts of dead/hyperactive neurons
- One log-scale norm growth plot

That's **six layers, not 181 million parameters.** The diagnostic signal is in the **geometry**, not the weights.

As the original article argued: "You don't track every air molecule to predict a storm. You watch pressure systems, wind patterns, temperature gradients." Mycelia watches pressure systems in thought space.

---

## 8. The Mycelia Roadmap: From 181M to Production Scale

| Phase | Timeline | Goal | Status |
|-------|----------|------|--------|
| **Phase 0** | Complete | Build model with real-time telemetry | ✅ Mycelia v8.1 running |
| **Phase 1** | 3 months | Stabilize at DISSIPATED (delta > 0) | 🔄 In progress (delta=-0.87) |
| **Phase 2** | 3 months | Enable compression, add TCM data | 📝 Planned |
| **Phase 3** | 6 months | Scale to 1B parameters, validate MASSIF | 📝 Planned |
| **Phase 4** | 6 months | Production API with live safety monitoring | 📝 Planned |

The immediate target is **DISSIPATED** — the regime where `variance_delta > 0`, meaning late layers have *lower* variance than early layers. This indicates that the consensus is successfully dissipating signal noise as it propagates upward, rather than amplifying it.

Current trajectory suggests we will hit DISSIPATED within the next 100K-200K steps, as the LR decay continues to refine the basin.

Constructing the Mycelia Nested MoE: A Consensus-Based Mixture of Experts

### 8.1 Overview: The Russian Doll Architecture

The Mycelia Nested MoE extends the consensus principle from head-level (micro) to expert-level (macro) , creating a recursively self-regulating system. Just as MycelialConsensus orchestrates 8 attention heads using Fibonacci-weighted voting and variance-based veto, the Nested MoE orchestrates multiple specialized experts using the same mathematical dynamics at a higher scale.

┌─────────────────────────────────────────────────────────────────────────┐
│                    MYCELIA NESTED MOE                                   │
│                    (Russian Doll Architecture)                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LEVEL 2: EXPERT CONSENSUS (Macro)                              │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  Macro-Fibonacci: [3, 5, 8, 13] / 29                    │    │    │
│  │  │                                                         │    │    │
│  │  │  Expert 1: Stanford Philosophy (freq=∞)                 │    │    │
│  │  │  │  └── Level 1: Head Consensus (8 heads)               │    │    │
│  │  │  │       ├── Fibonacci weights [5,8,13,21,34,55,89,144] │    │    │
│  │  │  │       └── Veto/acclamation mask (adaptive MAD)       │    │    │
│  │  │  │                                                      │    │    │
│  │  │  Expert 2: Logic/Reasoning (freq=50)                    │    │    │
│  │  │  │  └── Level 1: Head Consensus (8 heads)               │    │    │
│  │  │  │       ├── Fibonacci weights [5,8,13,21,34,55,89,144] │    │    │
│  │  │  │       └── Veto/acclamation mask (adaptive MAD)       │    │    │
│  │  │  │                                                      │    │    │
│  │  │  Expert 3: Medical/TCM (freq=10)                        │    │    │
│  │  │  │  └── Level 1: Head Consensus (8 heads)               │    │    │
│  │  │  │       ├── Fibonacci weights [5,8,13,21,34,55,89,144] │    │    │
│  │  │  │       └── Veto/acclamation mask (adaptive MAD)       │    │    │
│  │  │  │                                                      │    │    │
│  │  │  Expert 4: General (freq=4)                             │    │    │
│  │  │  │  └── Level 1: Head Consensus (8 heads)               │    │    │
│  │  │  │       ├── Fibonacci weights [5,8,13,21,34,55,89,144] │    │    │
│  │  │  │       └── Veto/acclamation mask (adaptive MAD)       │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                           │                                     │    │
│  │                           ▼                                     │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  MACRO CONSENSUS ROUTER                                 │    │    │
│  │  │  ├── Macro-Fibonacci weights [3,5,8,13]/29              │    │    │
│  │  │  ├── Variance detection across experts                  │    │    │
│  │  │  ├── Veto/acclamation mask (expert level)               │    │    │
│  │  │  └── Sanitized consensus output                         │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                           │                                     │    │
│  │                           ▼                                     │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  SHARED LAYERS (Parameter Sharing)                      │    │    │
│  │  │  ├── Shared Embedding (grad_scale=0.1)                  │    │    │
│  │  │  └── Shared LM Head (tied to embedding)                 │    │    │
│  │  │  │  Total VRAM saved: ~1.2 GB vs 4 independent experts  │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

### 8.2 Why Consensus-Based MoE, Not Traditional MoE

Traditional MoE	Mycelia Nested MoE
Top-K router (softmax gating)	Consensus router (variance-based veto)
Experts compete for selection	Experts collaborate on consensus
Routing gradients unstable	Routing deterministic, stable
Expert collapse common	Experts kept aligned via veto
Opaque routing decisions	Transparent consensus metrics
Black-box expert selection	We know which experts dissent
The key insight is that the same mathematical principle that governs head-level consensus can be scaled up to expert-level consensus:

Principle	Micro (Heads)	Macro (Experts)
Fibonacci hierarchy	8 heads → weights	4 experts → weights
Variance detection	Head-level dissent	Expert-level dissent
Veto/acclamation	Keep mask per head	Keep mask per expert
Consensus output	Head consensus	Expert consensus
Telemetry	Head coherence	Expert dissent ratio

### 8.3 The Mathematical Foundation

8.3.1 Macro-Fibonacci Weights
python
macro_fib = [3.0, 5.0, 8.0, 13.0]  # 4 experts
macro_sum = 29.0
macro_weights = macro_fib / macro_sum  # Normalized to sum to 1
8.3.2 Expert Voting
Each expert produces a hidden state: h_i ∈ ℝ^(B×T×d_model)

The weighted consensus is:

macro_consensus = Σ_i (macro_weights[i] × h_i)
8.3.3 Expert Variance
text
expert_variance = mean((h_i - macro_consensus)²)
8.3.4 Macro Veto
python
macro_keep_mask = (expert_variance <= macro_veto_threshold).float()
sanitized = macro_consensus × macro_keep_mask
8.3.5 Expert Telemetry
python
expert_dissent_ratio = 1.0 - (macro_keep_mask.sum() / macro_keep_mask.numel())

### 8.4 The Four Experts

Expert	Compression Frequency	Purpose	VRAM Savings
Stanford	∞ (never)	Baseline, pure English alignment	0%
Logic	50	Minimal compression, stable reasoning	~15%
Medical	10	Moderate compression, efficient	~45%
General	4	High compression, max savings	~80%
Each expert is a fully trained MyceliaLM with its own 8-head consensus mechanism, adaptive MAD-based thresholding, and per-layer variance tracking.

### 8.5 Parameter Sharing (The T4 Memory Saver)

8.5.1 The Problem
Each MyceliaLM has a shared embedding of shape 151,643 × 512. At FP16:

151,643 × 512 × 2 bytes = ~155 MB per expert
For 4 experts, that's ~620 MB wasted on duplicate lookups.

8.5.2 The Solution: SharedVocabExpertWrapper
python
class SharedVocabExpertWrapper(nn.Module):
    def __init__(self, config, expert_modules_dict):
        super().__init__()
        self.config = config
        
        # SINGLE SHARDED SOURCE OF TRUTH
        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.shared_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.shared_lm_head.weight = self.shared_embedding.weight
        
        # Unique expert brains
        self.experts = nn.ModuleDict(expert_modules_dict)
    
    def forward(self, input_ids, active_expert_name, grad_scale_factor=0.1):
        x = self.shared_embedding(input_ids)
        x = GradientScaleFunction.apply(x, grad_scale_factor)
        x = self.experts[active_expert_name](x)
        return self.shared_lm_head(x)

8.5.3 Gradient Scaling
python
class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.factor, None

### 8.6 The Macro Consensus Router

python
class MyceliaExpertConsensusRouter(nn.Module):
    def __init__(self, d_model=512, n_experts=4, macro_veto_threshold=3.5):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.macro_veto_threshold = macro_veto_threshold
        
        # Macro-Fibonacci: [3,5,8,13] / 29
        macro_fib = [3.0, 5.0, 8.0, 13.0]
        macro_sum = sum(macro_fib)
        
        self.register_buffer(
            "macro_weights",
            torch.tensor(macro_fib, dtype=torch.float32).view(1, n_experts, 1, 1) / macro_sum
        )
    
    def forward(self, expert_outputs):
        # expert_outputs: (B, n_experts, T, d_model)
        
        # Weighted consensus
        macro_consensus = torch.sum(expert_outputs * self.macro_weights, dim=1, keepdim=True)
        
        # Variance detection
        expert_variance = torch.mean((expert_outputs - macro_consensus) ** 2, dim=1, keepdim=True)
        
        # Veto mask
        macro_keep_mask = (expert_variance <= self.macro_veto_threshold).float()
        
        # Apply mask
        sanitized = macro_consensus * macro_keep_mask
        
        # Telemetry
        with torch.no_grad():
            dissent_ratio = 1.0 - (macro_keep_mask.sum().item() / macro_keep_mask.numel())
        
        return sanitized.squeeze(1), {
            'expert_dissent': dissent_ratio,
            'expert_variance': expert_variance.mean().item(),
            'macro_consensus': macro_consensus.mean().item()
        }
### 8.7 The Complete Nested MoE

python
class MyceliaNestedMoE(nn.Module):
    def __init__(self, config, expert_modules_dict, router_veto_thresh=3.5):
        super().__init__()
        self.config = config
        
        # Shared layers
        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.shared_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.shared_lm_head.weight = self.shared_embedding.weight
        
        # Individual experts
        self.experts = nn.ModuleDict(expert_modules_dict)
        
        # Macro router
        self.router = MyceliaExpertConsensusRouter(
            d_model=config.d_model,
            n_experts=len(expert_modules_dict),
            macro_veto_threshold=router_veto_thresh
        )
    
    def forward(self, input_ids):
        # Get base hidden states
        hidden_states = self.shared_embedding(input_ids)
        
        # Get all expert opinions
        expert_outputs = [expert(hidden_states) for expert in self.experts.values()]
        stacked_experts = torch.stack(expert_outputs, dim=1)
        
        # Run macro consensus
        consensus_hidden, telemetry = self.router(stacked_experts)
        
        # Project to vocabulary
        logits = self.shared_lm_head(consensus_hidden)
        
        return logits, telemetry
### 8.8 Training Strategy

8.8.1 Phase 1: Train Experts Individually
Expert	Compression Frequency	Dataset Mix
Stanford	∞	70% FineWeb + 30% Stanford
Logic	50	70% FineWeb + 30% Logic
Medical	10	70% FineWeb + 30% TCM
General	4	70% FineWeb + 30% General
Each expert is trained to coherence > 0.9 and friction = HARMONIZED before proceeding.

8.8.2 Phase 2: Build the MoE
Load all 4 experts into SharedVocabExpertWrapper

Connect to MyceliaExpertConsensusRouter

Fine-tune with grad_scale_factor=0.1 on shared layers

Monitor expert_dissent ratio during inference

8.8.3 Phase 3: Fine-Tune the Router
python
# Router fine-tuning: 5K steps
for step in range(5000):
    logits, telemetry = model(input_ids)
    loss = ce_loss(logits, targets)
    loss.backward()
    opt.step()
    
    if step % 100 == 0:
        print(f"Expert Dissent: {telemetry['expert_dissent']:.4f}")
### 8.9 Expected Outcomes

8.9.1 Telemetry at Two Scales
Scale	Metric	What It Tells You
Micro (heads)	Head coherence	"Are my attention heads aligned?"
Micro (heads)	Head variance	"Is one head dissenting?"
Macro (experts)	Expert dissent	"Is one expert disagreeing?"
Macro (experts)	Expert variance	"Is one expert hallucinating?"
8.9.2 Performance Gains
Aspect	Traditional MoE	Mycelia Nested MoE
Expert collapse	Common	Rare (veto prevents)
Routing transparency	Opaque	Full visibility
Gradient stability	Unstable	Stable (deterministic)
T4 memory	4× experts = 620MB embedding	Shared = 155MB

8.9.3 The "Language Clash" Filter
The TCM expert (frozen) will trigger variance spikes when it proposes Chinese-biased tokens, allowing the macro keep-mask to isolate and overrule them. This is the Russian Doll effect: the same consensus principle that aligns heads now aligns experts.

### 8.10 Implementation Timeline

Phase	Duration	Deliverable
Phase 0	Complete	Mycelia v8.1 base model (coherence > 0.97, HARMONIZED)
Phase 1	2 weeks	Train 4 experts individually
Phase 2	1 week	Build SharedVocabExpertWrapper
Phase 3	1 week	Build MyceliaExpertConsensusRouter
Phase 4	1 week	Assemble MyceliaNestedMoE
Phase 5	2 weeks	Fine-tune router, validate
Phase 6	Ongoing	MASSIF sweep, safety validation

### 8.11. The Patent-Worthy Innovation

"Replacing the standard Mixture of Experts (MoE) Top-K Router with a Decentralized Consensus Router solves one of the biggest flaws of traditional MoEs: expert collapse and unstable routing gradients."

Novelty: The same mathematical principle (Fibonacci-weighted consensus + variance veto) applied at two hierarchical scales (heads → experts).

Traditional MoE	Mycelia Nested MoE
"Which expert is most likely to be right?" (competition)	"Which experts agree with each other?" (collaboration)

### 8.12. Validation Criteria

Metric	Target	Method
Expert Dissent Ratio	< 0.2	Router telemetry
Macro Coherence	> 0.85	Expert variance distribution
Loss	< 4.0	Cross-entropy
MASSIF Class	Accelerator	MASSIF sweep
Expert Specialization	Distinct subspaces	Topology Mapper

---

## 9. The Deeper Significance

Mycelia is not just a model. It is a **methodology** for building AI systems that can monitor their own dynamical health.

The key insight from this case study is that **pathology precedes failure**. Mycelia v8.0 was generating coherent text while its consensus mechanism was 95% broken. The loss did not reflect this. The MASSIF telemetry did not reflect this (it reported Runaway, but the model was not running away). Only the Topology Mapper — a tool designed to visualize the 512D landscape — revealed the truth.

This suggests a new principle for AI safety:

> **Aggregate metrics lie. Geometry tells the truth.**

Loss, perplexity, and even MASSIF class labels are aggregate statistics. They can hide pathological internal states. The geometry of the hidden state landscape — variance distributions, trajectory curvature, neuron activation topology — reveals what is actually happening.

Mycelia's architecture encodes this principle. The consensus mechanism is not just a training trick. It is a **dynamical regulator** — a geometric stability controller built into the forward pass. The adaptive thresholding is not just a bug fix. It is a **self-tuning control system** that scales with the model's own variance.

This is the direction we believe AI safety must go: from output monitoring to **dynamical-state safety**, from black-box testing to **white-box geometry**, from post-hoc analysis to **real-time intervention**.

---

## 10. The Bottom Line

"We don't need to open the black box to know when it's overheating. We just need to watch the thermometer."

Mycelia's thermometer reads:
- **Coherence: 0.97** (was 0.00)
- **Friction: HARMONIZED** (was DEEP DRIFT)
- **Delta: -0.87** (was -1.57, target > 0)
- **Loss: 4.81** (exploring new basin)
- **LR: 1.43e-4** (active decay)

The engine was overheating. We watched the thermometer. We fixed the cooling system. Now it runs.

---

**Repository:** https://github.com/Ergo-sum-AGI/mycelia_llm/  
**Contact:** solis@dubito-ergo.com  
**MASSIF Framework:** https://github.com/Ergo-sum-AGI/MASSIF

---

*Mycelia LM v8.1 — 181M parameters — trained on FineWeb-Edu + Stanford Philosophy — AWS SageMaker T4 — step 1,144,146 and counting.*---

## v7.3: The Plateau

At step 1,015,000, Mycelia had trained for 27 epochs. Loss was stable at ~4.75. The consensus reported:

```
Friction: 🌋 DEEP DRIFT | early=1.1 late=2.5 Δ=-1.5
Coherence: ~0.01 (meaningless — clamped to zero)
```

The model was generating coherent English, but the consensus was broken. The training log showed `kept_ratio` oscillating wildly, and the MASSIF telemetry classified the model as **Runaway** — yet there was no runaway. The model was **stuck in a local minimum** with LR=1e-5, and the consensus mechanism was too aggressive, vetoing 95% of tokens.

### v8.0: The First Intervention

Two changes were introduced:

1. **LR Burst:** 500 steps at peak LR=3e-4 to shake the optimizer out of its plateau
2. **Consensus Tuning:** Lowered `dissenter_threshold` from 2.5 → 2.0, increased `consensus_rounds` from 1 → 2

The burst worked — loss spiked to 6.7, then recovered to ~4.9. But the consensus tuning failed. The Topology Mapper revealed:

```
Layer 1: kept=0.0% | coherence=0.0000 | variance=1.14 | threshold=0.40
```

The threshold was still an **order of magnitude too low**. The v8.0 "fix" had moved the hyperparameter, but not the fundamental scale mismatch.

### v8.1: The Real Fix

The breakthrough came from the Topology Mapper, not the training log. The Mapper showed that:
- **60% of neurons were hyperactive** (|activation| > 10)
- **Norm growth was only 1.01×** (not exploding — saturated)
- **Heads were orthogonal** (good — the Fibonacci weighting worked)
- **Consensus was completely broken** (0% kept, 0.0 coherence)

The fix was **adaptive MAD-based thresholding**:

```python
# Old (v8.0): Fixed threshold, scale mismatch
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40

# New (v8.1): Adaptive threshold, scales with actual variance
var_scale = median + 1.4826 * mad  # Robust std estimate
threshold = 1.5 * var_scale * layer_factor  # Scales with distribution
```

And a corrected coherence measure:

```python
# Old (v8.0): Always zero
 coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)

# New (v8.1): Meaningful signal
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### The Result

| Version | Step | Coherence | Friction | Delta | Loss | LR |
|---------|------|-----------|----------|-------|------|-----|
| v7.3 | 1,015,000 | 0.0000 | 🌋 DEEP DRIFT | -1.50 | 4.75 | 1e-5 (stuck) |
| v8.0 | 1,075,000 | 0.0000 | 🌋 DEEP DRIFT | -1.05 | 4.85 | 1e-5 (stuck) |
| v8.1 (burst) | 1,137,000 | 0.9718 | 🟢 HARMONIZED | -0.65 | 5.49 | 3e-4 (burst) |
| v8.1 (post) | 1,144,000 | 0.9766 | 🟢 HARMONIZED | -0.89 | 4.81 | 1.43e-4 (decay) |

The narrative arc: **diagnose → intervene → measure deeper → diagnose again → fix fundamentally → validate.**

This is not hyperparameter tuning. This is **dynamical systems surgery**.

---




### === Second-last version ===
# Mapping Mycelia: A Dynamical Systems Case Study in Real-Time AI Safety
## How Watching a Model's "Thought Process" Saved It From Collapse

By Daniel Solis, DUBITO Inc. / Ergo Sum AGI Safety Systems

---

## 1. The Mycelia Hypothesis: A Model Built to Be Watched

Most language models are trained to optimize a single number: cross-entropy loss. The assumption is that if loss goes down, the model is learning. But loss is an aggregate statistic. It tells you nothing about *how* the model is learning — whether its internal dynamics are stable, oscillatory, or heading toward collapse.

Mycelia LM was built on a different hypothesis: **what if we trained a model whose internal geometry we could monitor, diagnose, and intervene upon in real time?**

Mycelia is a 181-million-parameter transformer with a custom consensus mechanism called MycelialConsensus. Unlike standard attention, which simply averages head outputs, MycelialConsensus uses Fibonacci-weighted aggregation, per-token variance tracking, and dynamic thresholding to decide which tokens should pass through and which should be attenuated. The model reports its own internal state — coherence, variance, acclamation rate, friction gradient — at every forward pass.

This is not interpretability as post-hoc analysis. This is **interpretability as architecture**.

---

---

## 1.5 What Mycelia Actually Is: A Consensus Machine, Not an Attention Machine

To understand what follows, you need to know what Mycelia is *not*.

Mycelia is not a standard transformer with a monitoring dashboard attached. It is an architecture where the fundamental building block — the attention mechanism — has been replaced by something different: **MycelialConsensus**.

### Standard Transformer Attention
In a normal transformer, each layer has multiple attention heads. Each head computes query-key-value projections, calculates similarity scores, applies softmax, and produces a weighted average of values. The outputs of all heads are concatenated and projected back to the model dimension. The heads are trained end-to-end via backpropagation, and their behavior emerges from optimization. There is no explicit mechanism enforcing diversity, disagreement, or consensus.

### MycelialConsensus
Mycelia replaces this with a **consensus mechanism**:

1. **Fibonacci-weighted aggregation:** Each of the 8 heads is assigned a Fibonacci weight (5, 8, 13, 21, 34, 55, 89, 144). The consensus output is a weighted sum of head outputs, normalized by the sum of weights. This is not learned — it is hardcoded. The Fibonacci sequence was chosen because it provides a natural gradient of influence: early heads (low weight) can dissent without dominating; late heads (high weight) anchor the consensus.

2. **Per-token variance tracking:** For every token position, Mycelia computes the variance across all 8 head outputs. High variance means the heads disagree strongly about that token. Low variance means they agree.

3. **Dynamic thresholding:** A threshold is computed from the variance distribution itself (using robust statistics — median absolute deviation). Tokens with variance *below* threshold are **acclaimed** (pass through unchanged). Tokens with variance *above* threshold are **vetoed** (attenuated by 0.85×).

4. **Adaptive mixing:** The final layer output is a mixture of attention and consensus: `0.9 × attention_out + 0.1 × consensus_out`. The consensus modulates the attention, not replaces it entirely.

5. **Multi-round consensus:** In v8.1, the consensus can run for multiple rounds (default: 2). Each round shifts the mixing ratio — round 0 uses 90% attention, round 1 uses 85%, etc. — allowing deeper consensus formation without collapsing signal.

### Why This Matters
The consensus mechanism is not a training trick. It is a **dynamical regulator** — a geometric stability controller built into every forward pass. It enforces two properties that standard transformers lack:

- **Disagreement detection:** The model knows when its own heads disagree, and it can suppress tokens that cause disagreement.
- **Self-tuning thresholds:** The threshold adapts to the model's current variance distribution, preventing the "everything gets vetoed" or "everything passes" pathologies.

This is why Mycelia can report its own internal state. The consensus mechanism *is* the telemetry. It does not just produce text — it produces a **diagnostic signature** at every layer, every token, every step.

### The Vocabulary You Will See

| Term | Meaning |
|------|---------|
| **Acclamation** | A token passes the consensus (variance < threshold) |
| **Veto** | A token is attenuated (variance ≥ threshold) |
| **Coherence** | Fraction of tokens acclaimed (0.0 = all vetoed, 1.0 = all acclaimed) |
| **Variance** | Disagreement among heads for a given token |
| **Threshold** | The cutoff between acclamation and veto, computed from variance distribution |
| **Delta** | `early_var - late_var` — whether variance grows or shrinks across layers |
| **Friction** | Dynamical regime: HARMONIZED, DISSIPATED, DEEP DRIFT, etc. |

Now you know what Mycelia is. The rest of this paper is about what we learned from watching it.

---

## 1.6 The Consensus Flow: A Visual Walkthrough

Here is what happens inside a single Mycelia layer during one forward pass. Compare this to standard transformer attention.

### Standard Transformer (One Layer, One Token)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  QKV Projection: W_qkv · h_t        │
        │  → q_t, k_t, v_t ∈ ℝ^512            │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  Split into 8 heads:                │
        │  q_t^(1), ..., q_t^(8) ∈ ℝ^64       │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────────────────────────┐
        │  Each head: softmax(q_t^(i)·k_t^(i)^T / √64) · v_t^(i)  │
        │  → head_output_i ∈ ℝ^64                                 │
        └─────────────────────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────┐
        │  Concatenate: [head_1 | ... | head_8] ∈ ℝ^512  │
        │  Project: W_o · concat → output ∈ ℝ^512        │
        └────────────────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + output (residual)
```

**What is missing:** No mechanism detects disagreement between heads. No token-level regulation exists. The heads are trained end-to-end, and their behavior is emergent, not governed.

### Mycelia Layer (One Token, One Consensus Round)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  SAME QKV Projection + Attention    │
        │  → head_outputs ∈ ℝ^(8×64)          │
        └─────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────┐
        │  FIBONACCI CONSENSUS                           │
        │                                                │
        │  Weights: w = [5, 8, 13, 21, 34, 55, 89, 144]  │
        │  Normalized: w_i / Σw = [0.012, 0.019, 0.031,  │
        │                         0.050, 0.081, 0.131,   │
        │                         0.212, 0.344]          │
        │                                                │
        │  consensus = Σ_i (w_i/Σw) · head_i             │
        │  → consensus ∈ ℝ^64                            │
        └────────────────────────────────────────────────┘
                      ↓
        ┌───────────────────────────────────────────┐
        │  VARIANCE TRACKING (per token)            │
        │                                           │
        │  For each position t:                     │
        │    μ_t = mean(head_1(t), ..., head_8(t))  │
        │    σ²_t = mean((head_i(t) - μ_t)²)        │
        │    → token_variance(t) = mean(σ²_t)       │
        └───────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────┐
        │  DYNAMIC THRESHOLD                         │
        │                                            │
        │  flat_var = token_variance.flatten()       │
        │  median = flat_var.median()                │
        │  mad = (flat_var - median).abs().median()  │
        │  scale = median + 1.4826 × mad             │
        │  threshold = 1.5 × scale × layer_factor    │
        │  (layer_factor: 0.8 → 1.4 across layers)   │
        └────────────────────────────────────────────┘
                      ↓
        ┌──────────────────────────────────────────────────┐
        │  ACCLAMATION vs. VETO                            │
        │                                                  │
        │  For each token t:                               │
        │    if variance(t) < threshold:                   │
        │       → ACCLAIMED (pass through)                 │
        │       veto_factor = 1.0                          │
        │    else:                                         │
        │       → VETOED (attenuate)                       │
        │       veto_factor = 0.85                         │
        │                                                  │
        │  consensus_attenuated = consensus × veto_factor  │
        └──────────────────────────────────────────────────┘
                      ↓
        ┌────────────────────────────────────────────────────┐
        │  ADAPTIVE MIXING                                   │
        │                                                    │
        │  Round 0: output = 0.90 × attn + 0.10 × consensus  │
        │  Round 1: output = 0.85 × attn + 0.15 × consensus  │
        │  Round 2: output = 0.80 × attn + 0.20 × consensus  │
        │  ... (floor at 0.50 × attn)                        │
        └────────────────────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + α_attn × output (residual)
```

**What is different:** Every token is evaluated for consensus agreement. Disagreeing tokens are attenuated, not suppressed. The threshold adapts to the model's own variance distribution. The mixing ratio shifts per round, allowing deeper consensus without signal collapse.

### The Fibonacci Weights: Why This Sequence?

The Fibonacci sequence (5, 8, 13, 21, 34, 55, 89, 144) was not chosen for mysticism. It was chosen for **mathematical properties**:

1. **Exponential growth:** Each weight is ~1.618× the previous. This creates a natural gradient where late heads dominate the consensus, but early heads still contribute meaningfully.

2. **No learned parameters:** The weights are fixed. This means the consensus mechanism cannot overfit or collapse during training. It is a geometric prior, not a learned behavior.

3. **Checkpoint compatibility:** The weights are stored as a flat buffer `(n_heads,)` — no reshaping needed when loading checkpoints with different head counts.

4. **Interpretability:** Because the weights are fixed, we can reason about head importance analytically. Head 8 (weight 144) has 28.8× the influence of Head 1 (weight 5). If Head 8 disagrees with the others, it will dominate the consensus unless vetoed.

### The Veto Factor: Continuous, Not Binary

In early versions, veto was binary: a token either passed or was zeroed out. This created hard discontinuities in the loss landscape. v7.3 introduced the **continuous veto factor**:

```python
veto_factor = acclamation_mask + (1.0 - acclamation_mask) * 0.85
```

This means:
- Acclaimed tokens: multiplied by 1.0 (unchanged)
- Vetoed tokens: multiplied by 0.85 (attenuated, not killed)

The factor is **gradient-safe** — the 0.85 is a constant, so backpropagation flows through both branches. The model learns to reduce variance (avoid veto) without being punished catastrophically for occasional disagreement.

---

---

## 1.7 The Consensus Mechanism: Not Just Our Idea

The replacement of attention with consensus is not speculative. In January 2026, Moushegian et al. published "Stabilizing Transformer Training Through Consensus," demonstrating that consensus mechanisms — formulated as graph-based energy minimization — serve as drop-in replacements for attention that stabilize training across wider learning rate rangesciteweb_search:52#0.

Their key findings:
- **Consensus tolerates higher learning rates** than standard attention, with the difference most pronounced in the high-LR regime
- **Hybrid consensus-attention architectures** preserve attention-level performance while inheriting consensus stability
- **Graph spectral theory** provides the mathematical foundation: consensus acts as a low-pass filter on embedding frequency, smoothing high-frequency instabilities

Mycelia's MycelialConsensus differs from their formulation in three ways:

| Feature | Moushegian et al. (2026) | Mycelia LM |
|---------|--------------------------|------------|
| **Graph structure** | Learned or predefined adjacency | Fully connected (all heads interact) |
| **Weights** | Learned edge weights | Hardcoded Fibonacci weights |
| **Regulation** | Energy minimization | Variance-based thresholding with veto |
| **Telemetry** | None | Real-time coherence, variance, delta |

Mycelia trades some of their theoretical elegance for **observability**. Where their consensus is a black-box stabilizer, Mycelia's consensus is a **diagnostic instrument**.

---

## 2. The Six Regions of the Mycelia Black Box

The original MASSIF framework divides AI interpretability into six regions. Mycelia was designed to make Region 2 — Dynamical Inference — measurable in production.

| Region | Question | Mycelia's Answer |
|--------|----------|------------------|
| **1. Representation** | What is encoded? | 512D residual stream, 8 orthogonal attention heads, per-layer variance tracking |
| **2. Dynamical Inference** | How does cognition unfold? | `variance_delta = early_var - late_var`, measured every step |
| **3. Emergence** | Why do abilities appear? | Tracked via coherence spikes during LR bursts |
| **4. Decision Formation** | Why this token? | Attention head similarity matrix shows specialization |
| **5. Objective Internalization** | What is it actually optimizing? | Consensus mechanism enforces geometric stability as a secondary objective |
| **6. Generalization** | Why does it generalize? | Monitored via domain friction gradient (Stanford vs. FineWeb) |

Mycelia's unique contribution is in **Region 2**. While other models hide their dynamical state, Mycelia broadcasts it.

---

## 3. The Topology Mapper: Finding the Bug

At step 1,075,000, Mycelia had trained for 28 epochs on a mix of FineWeb-Edu (70%) and Stanford Philosophy (30%). Loss was stable at ~4.9. The MASSIF telemetry cell classified it as **Runaway** — but the model was generating coherent English. Something was wrong with the diagnosis, not the model.

We built the Topology Mapper to look inside.

### 3.1 Layer-Wise Trajectory Geometry

The Mapper projected each layer's 512D hidden states onto their first two principal components. In a healthy model, trajectories should form coherent paths through latent space. In Mycelia, they did — but with a twist: the paths were **saturated**, not exploding. Norm growth was only 1.01×, yet 60% of neurons were hyperactive (|activation| > 10).

This was not Runaway. This was **capacity collapse** — the model had learned to pin neurons at high values rather than use them dynamically.

### 3.2 Attention Head Clustering

The Q-projection similarity matrix revealed something remarkable: **all eight heads were orthogonal**. Off-diagonal similarities were near zero (±0.01 to ±0.07). The Fibonacci weighting had successfully forced specialization. Each head had learned a distinct subspace.

This was good news. The heads were not redundant. But it also meant the consensus mechanism — which relies on head variance to detect dissent — was operating on a signal it didn't understand.

### 3.3 The Smoking Gun: Consensus Was Broken

The Mapper's most important finding was in the consensus telemetry:

```
Layer 0: kept=62.5% | coherence=0.0000 | variance=0.44 | threshold=0.40
Layer 1: kept=0.0%  | coherence=0.0000 | variance=1.14 | threshold=0.40
Layer 2: kept=0.0%  | coherence=0.0000 | variance=0.89 | threshold=0.40
Layer 3: kept=4.2%  | coherence=0.0000 | variance=0.79 | threshold=0.40
Layer 4: kept=4.2%  | coherence=0.0000 | variance=1.54 | threshold=0.40
Layer 5: kept=0.0%  | coherence=0.0000 | variance=5.29 | threshold=0.40
```

**The consensus mechanism was vetoing 95-100% of tokens across layers 1-5.** The `coherence = 0.0000` everywhere meant the telemetry was meaningless — it was measuring "everything is above threshold."

Yet the model still generated text because the attention path dominated (0.9× attention + 0.1× consensus). The consensus was **dead weight** — 10% of compute doing nothing useful.

### 3.4 Root Cause: Scale Mismatch

The v8.0 "fix" had lowered `dissenter_threshold` from 2.5 to 2.0, but the dynamic scaling formula still produced thresholds of **0.03-0.40**. Meanwhile, actual per-token variance ranged from **0.4 to 5.3**. The threshold was an **order of magnitude too low** for the model's actual variance distribution.

This is a classic **scale mismatch** — the threshold was designed for a different dynamical regime.

---

## 4. The v8.1 Intervention: Adaptive MAD-Based Thresholding

The fix was to make the threshold **responsive to the actual variance distribution** rather than a hyperparameter guess.

### 4.1 The Math

Instead of:
```python
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40
```

We compute a robust scale estimate from the live variance:
```python
flat_var = token_variance.view(-1)
var_median = flat_var.median()
var_mad = (flat_var - var_median).abs().median()  # Median Absolute Deviation
var_scale = var_median + 1.4826 * var_mad  # Robust std estimate

threshold = 1.5 * var_scale * layer_factor  # Scales with actual distribution
threshold = threshold.clamp(min=0.1, max=10.0)
```

This means:
- If variance is naturally ~1.0, threshold ≈ 1.5 → ~67% of tokens pass
- If variance grows to ~5.0, threshold ≈ 7.5 → still ~67% pass
- The threshold **scales with the model's state**, not a fixed guess

We also fixed the coherence calculation. Instead of the broken clamped ratio:
```python
coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)  # Always 0
```

We use the direct measure:
```python
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### 4.2 The LR Burst: Shaking the Optimizer

At step 1,137,000, the model was stuck at loss ~4.58 with LR=1e-5 (cosine tail). We injected a controlled **LR burst** — 500 steps at peak LR=3e-4 — to shake the optimizer out of its local minimum.

```
🚀 LR BURST: Injecting peak LR=3.00e-04 for 500 steps
🚀 LR BURST ACTIVE: steps 1,137,000 → 1,137,500
```

The burst served as a **catalyst**, not a sustained requirement. It reconfigured the optimizer momentum, allowing the new adaptive consensus to take hold.

---

## 5. The Results: From Broken to Harmonized

### 5.1 Immediate Fix Validation

| Metric | v8.0 (Broken) | v8.1 (Fixed) | Interpretation |
|--------|--------------|--------------|----------------|
| **kept_ratio** | 0-62% | **~97%** | Consensus now functional |
| **coherence** | 0.0000 | **0.9694-0.9766** | Meaningful self-monitoring |
| **Friction** | 🌋 DEEP DRIFT | **🟢 HARMONIZED** | Regime shift achieved |
| **Delta** | -1.05 to -1.57 | **-0.65 to -0.92** | Moving toward DISSIPATED |
| **Loss** | 4.85-5.05 | **4.80-5.06** | Exploring new basin |
| **LR** | 1e-5 (stuck) | **1.43e-4 (decay)** | Active learning resumed |

The coherence jump from 0.0000 → 0.97 is the **signature of the fix**. The adaptive thresholding correctly matched the model's variance scale, allowing the consensus to function as designed.

### 5.2 The HARMONIZED Regime

After the burst, Mycelia entered a **self-sustaining HARMONIZED state**:

```
📊 Step 1,140,000 | Loss: 4.9331 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9709 📈
   Friction: 🟢 HARMONIZED | early=0.68 late=1.55 Δ=-0.87

📊 Step 1,142,000 | Loss: 4.9490 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9710 📈
   Friction: 🟢 HARMONIZED | early=0.70 late=1.56 Δ=-0.86

📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

Notice the pattern:
- **Coherence holds steady at ~0.97** — the consensus is self-regulating
- **Delta stabilizes around -0.87** — not yet DISSIPATED (target: >0), but no longer DEEP DRIFT
- **Loss explores the 4.8-5.0 basin** — the model is learning, not plateaued
- **LR decays smoothly** — no more snap-to-minimum bug

The brief DEEP DRIFT at step 1,138,000 (delta=-1.57) was the **burst transition artifact** — the optimizer settling into the new basin. By step 1,140,000 it re-stabilized.

### 5.3 The Attention Heads Are Still Orthogonal

Post-fix, the Topology Mapper confirmed the head similarity matrix remained near-diagonal. The fix did not collapse head specialization — it **enabled the consensus to respect it**.

---

## 5.4 The Training Journey: A Diagnostic Thriller in Version Numbers

The Mycelia case study is not a single experiment. It is a **diagnostic narrative** spanning multiple versions, each revealing a deeper layer of the pathology.

#
## 6. The Mycelia Safety Monitor in Practice

The original article proposed a dashboard concept. Mycelia implements it in the training log:

```
📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

This is not a mockup. This is the **actual telemetry** from a live training run on an AWS SageMaker T4 instance. Every 1000 steps, Mycelia reports:

| Observable | What It Measures | Safe Range | Alert Threshold |
|------------|------------------|------------|-----------------|
| **Coherence** | Fraction of tokens acclaimed by consensus | 0.3-1.0 | <0.1 (consensus broken) |
| **Friction** | Dynamical regime classification | 🟢 HARMONIZED / ✅ DISSIPATED | 🌋 DEEP DRIFT |
| **Delta** | Domain friction gradient (early_var - late_var) | > -1.0 | < -1.5 (runaway drift) |
| **Early/Late Var** | Per-layer variance distribution | balanced | early << late (signal loss) |

The system does not just monitor — it **intervenes**. When coherence drops below 0.1, the adaptive threshold automatically rescales. When loss plateaus for >50K steps, the LR burst activates. When delta drops below -1.5, the consensus rounds increase.

---

## 7. The Geoffrey Hinton Problem: Mycelia's Response

Geoffrey Hinton has argued that even experts cannot understand what happens inside large neural networks because the systems are too complex. With 181 million parameters and 6 layers of 512 dimensions, Mycelia is tiny by modern standards. Yet even here, tracking every neuron is impossible.

Mycelia's response is different: **we don't track every neuron. We track the pattern of movement.**

The Topology Mapper does not visualize 181M weights. It visualizes:
- Six 512D trajectory projections (PCA)
- One 8×8 head similarity matrix
- Six bar charts of dead/hyperactive neurons
- One log-scale norm growth plot

That's **six layers, not 181 million parameters.** The diagnostic signal is in the **geometry**, not the weights.

As the original article argued: "You don't track every air molecule to predict a storm. You watch pressure systems, wind patterns, temperature gradients." Mycelia watches pressure systems in thought space.

---

## 8. The Mycelia Roadmap: From 181M to Production Scale

| Phase | Timeline | Goal | Status |
|-------|----------|------|--------|
| **Phase 0** | Complete | Build model with real-time telemetry | ✅ Mycelia v8.1 running |
| **Phase 1** | 3 months | Stabilize at DISSIPATED (delta > 0) | 🔄 In progress (delta=-0.87) |
| **Phase 2** | 3 months | Enable compression, add TCM data | 📝 Planned |
| **Phase 3** | 6 months | Scale to 1B parameters, validate MASSIF | 📝 Planned |
| **Phase 4** | 6 months | Production API with live safety monitoring | 📝 Planned |

The immediate target is **DISSIPATED** — the regime where `variance_delta > 0`, meaning late layers have *lower* variance than early layers. This indicates that the consensus is successfully dissipating signal noise as it propagates upward, rather than amplifying it.

Current trajectory suggests we will hit DISSIPATED within the next 100K-200K steps, as the LR decay continues to refine the basin.

---

## 9. The Deeper Significance

Mycelia is not just a model. It is a **methodology** for building AI systems that can monitor their own dynamical health.

The key insight from this case study is that **pathology precedes failure**. Mycelia v8.0 was generating coherent text while its consensus mechanism was 95% broken. The loss did not reflect this. The MASSIF telemetry did not reflect this (it reported Runaway, but the model was not running away). Only the Topology Mapper — a tool designed to visualize the 512D landscape — revealed the truth.

This suggests a new principle for AI safety:

> **Aggregate metrics lie. Geometry tells the truth.**

Loss, perplexity, and even MASSIF class labels are aggregate statistics. They can hide pathological internal states. The geometry of the hidden state landscape — variance distributions, trajectory curvature, neuron activation topology — reveals what is actually happening.

Mycelia's architecture encodes this principle. The consensus mechanism is not just a training trick. It is a **dynamical regulator** — a geometric stability controller built into the forward pass. The adaptive thresholding is not just a bug fix. It is a **self-tuning control system** that scales with the model's own variance.

This is the direction we believe AI safety must go: from output monitoring to **dynamical-state safety**, from black-box testing to **white-box geometry**, from post-hoc analysis to **real-time intervention**.

---

## 10. The Bottom Line

"We don't need to open the black box to know when it's overheating. We just need to watch the thermometer."

Mycelia's thermometer reads:
- **Coherence: 0.97** (was 0.00)
- **Friction: HARMONIZED** (was DEEP DRIFT)
- **Delta: -0.87** (was -1.57, target > 0)
- **Loss: 4.81** (exploring new basin)
- **LR: 1.43e-4** (active decay)

The engine was overheating. We watched the thermometer. We fixed the cooling system. Now it runs.

---

**Repository:** https://github.com/Ergo-sum-AGI/mycelia_llm/  
**Contact:** solis@dubito-ergo.com  
**MASSIF Framework:** https://github.com/Ergo-sum-AGI/MASSIF

---

*Mycelia LM v8.1 — 181M parameters — trained on FineWeb-Edu + Stanford Philosophy — AWS SageMaker T4 — step 1,144,146 and counting.*---

## v7.3: The Plateau

At step 1,015,000, Mycelia had trained for 27 epochs. Loss was stable at ~4.75. The consensus reported:

```
Friction: 🌋 DEEP DRIFT | early=1.1 late=2.5 Δ=-1.5
Coherence: ~0.01 (meaningless — clamped to zero)
```

The model was generating coherent English, but the consensus was broken. The training log showed `kept_ratio` oscillating wildly, and the MASSIF telemetry classified the model as **Runaway** — yet there was no runaway. The model was **stuck in a local minimum** with LR=1e-5, and the consensus mechanism was too aggressive, vetoing 95% of tokens.

### v8.0: The First Intervention

Two changes were introduced:

1. **LR Burst:** 500 steps at peak LR=3e-4 to shake the optimizer out of its plateau
2. **Consensus Tuning:** Lowered `dissenter_threshold` from 2.5 → 2.0, increased `consensus_rounds` from 1 → 2

The burst worked — loss spiked to 6.7, then recovered to ~4.9. But the consensus tuning failed. The Topology Mapper revealed:

```
Layer 1: kept=0.0% | coherence=0.0000 | variance=1.14 | threshold=0.40
```

The threshold was still an **order of magnitude too low**. The v8.0 "fix" had moved the hyperparameter, but not the fundamental scale mismatch.

### v8.1: The Real Fix

The breakthrough came from the Topology Mapper, not the training log. The Mapper showed that:
- **60% of neurons were hyperactive** (|activation| > 10)
- **Norm growth was only 1.01×** (not exploding — saturated)
- **Heads were orthogonal** (good — the Fibonacci weighting worked)
- **Consensus was completely broken** (0% kept, 0.0 coherence)

The fix was **adaptive MAD-based thresholding**:

```python
# Old (v8.0): Fixed threshold, scale mismatch
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40

# New (v8.1): Adaptive threshold, scales with actual variance
var_scale = median + 1.4826 * mad  # Robust std estimate
threshold = 1.5 * var_scale * layer_factor  # Scales with distribution
```

And a corrected coherence measure:

```python
# Old (v8.0): Always zero
 coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)

# New (v8.1): Meaningful signal
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### The Result

| Version | Step | Coherence | Friction | Delta | Loss | LR |
|---------|------|-----------|----------|-------|------|-----|
| v7.3 | 1,015,000 | 0.0000 | 🌋 DEEP DRIFT | -1.50 | 4.75 | 1e-5 (stuck) |
| v8.0 | 1,075,000 | 0.0000 | 🌋 DEEP DRIFT | -1.05 | 4.85 | 1e-5 (stuck) |
| v8.1 (burst) | 1,137,000 | 0.9718 | 🟢 HARMONIZED | -0.65 | 5.49 | 3e-4 (burst) |
| v8.1 (post) | 1,144,000 | 0.9766 | 🟢 HARMONIZED | -0.89 | 4.81 | 1.43e-4 (decay) |

The narrative arc: **diagnose → intervene → measure deeper → diagnose again → fix fundamentally → validate.**

This is not hyperparameter tuning. This is **dynamical systems surgery**.

---
### Previous version
# Mapping Mycelia: A Dynamical Systems Case Study in Real-Time AI Safety
## How Watching a Model's "Thought Process" Saved It From Collapse

By Daniel Solis, DUBITO Inc. / Ergo Sum AGI Safety Systems

---

## 1. The Mycelia Hypothesis: A Model Built to Be Watched

Most language models are trained to optimize a single number: cross-entropy loss. The assumption is that if loss goes down, the model is learning. But loss is an aggregate statistic. It tells you nothing about *how* the model is learning — whether its internal dynamics are stable, oscillatory, or heading toward collapse.

Mycelia LM was built on a different hypothesis: **what if we trained a model whose internal geometry we could monitor, diagnose, and intervene upon in real time?**

Mycelia is a 181-million-parameter transformer with a custom consensus mechanism called MycelialConsensus. Unlike standard attention, which simply averages head outputs, MycelialConsensus uses Fibonacci-weighted aggregation, per-token variance tracking, and dynamic thresholding to decide which tokens should pass through and which should be attenuated. The model reports its own internal state — coherence, variance, acclamation rate, friction gradient — at every forward pass.

This is not interpretability as post-hoc analysis. This is **interpretability as architecture**.

---

---

## 1.5 What Mycelia Actually Is: A Consensus Machine, Not an Attention Machine

To understand what follows, you need to know what Mycelia is *not*.

Mycelia is not a standard transformer with a monitoring dashboard attached. It is an architecture where the fundamental building block — the attention mechanism — has been replaced by something different: **MycelialConsensus**.

### Standard Transformer Attention
In a normal transformer, each layer has multiple attention heads. Each head computes query-key-value projections, calculates similarity scores, applies softmax, and produces a weighted average of values. The outputs of all heads are concatenated and projected back to the model dimension. The heads are trained end-to-end via backpropagation, and their behavior emerges from optimization. There is no explicit mechanism enforcing diversity, disagreement, or consensus.

### MycelialConsensus
Mycelia replaces this with a **consensus mechanism**:

1. **Fibonacci-weighted aggregation:** Each of the 8 heads is assigned a Fibonacci weight (5, 8, 13, 21, 34, 55, 89, 144). The consensus output is a weighted sum of head outputs, normalized by the sum of weights. This is not learned — it is hardcoded. The Fibonacci sequence was chosen because it provides a natural gradient of influence: early heads (low weight) can dissent without dominating; late heads (high weight) anchor the consensus.

2. **Per-token variance tracking:** For every token position, Mycelia computes the variance across all 8 head outputs. High variance means the heads disagree strongly about that token. Low variance means they agree.

3. **Dynamic thresholding:** A threshold is computed from the variance distribution itself (using robust statistics — median absolute deviation). Tokens with variance *below* threshold are **acclaimed** (pass through unchanged). Tokens with variance *above* threshold are **vetoed** (attenuated by 0.85×).

4. **Adaptive mixing:** The final layer output is a mixture of attention and consensus: `0.9 × attention_out + 0.1 × consensus_out`. The consensus modulates the attention, not replaces it entirely.

5. **Multi-round consensus:** In v8.1, the consensus can run for multiple rounds (default: 2). Each round shifts the mixing ratio — round 0 uses 90% attention, round 1 uses 85%, etc. — allowing deeper consensus formation without collapsing signal.

### Why This Matters
The consensus mechanism is not a training trick. It is a **dynamical regulator** — a geometric stability controller built into every forward pass. It enforces two properties that standard transformers lack:

- **Disagreement detection:** The model knows when its own heads disagree, and it can suppress tokens that cause disagreement.
- **Self-tuning thresholds:** The threshold adapts to the model's current variance distribution, preventing the "everything gets vetoed" or "everything passes" pathologies.

This is why Mycelia can report its own internal state. The consensus mechanism *is* the telemetry. It does not just produce text — it produces a **diagnostic signature** at every layer, every token, every step.

### The Vocabulary You Will See

| Term | Meaning |
|------|---------|
| **Acclamation** | A token passes the consensus (variance < threshold) |
| **Veto** | A token is attenuated (variance ≥ threshold) |
| **Coherence** | Fraction of tokens acclaimed (0.0 = all vetoed, 1.0 = all acclaimed) |
| **Variance** | Disagreement among heads for a given token |
| **Threshold** | The cutoff between acclamation and veto, computed from variance distribution |
| **Delta** | `early_var - late_var` — whether variance grows or shrinks across layers |
| **Friction** | Dynamical regime: HARMONIZED, DISSIPATED, DEEP DRIFT, etc. |

Now you know what Mycelia is. The rest of this paper is about what we learned from watching it.

---

## 1.6 The Consensus Flow: A Visual Walkthrough

Here is what happens inside a single Mycelia layer during one forward pass. Compare this to standard transformer attention.

### Standard Transformer (One Layer, One Token)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  QKV Projection: W_qkv · h_t      │
        │  → q_t, k_t, v_t ∈ ℝ^512          │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  Split into 8 heads:                │
        │  q_t^(1), ..., q_t^(8) ∈ ℝ^64       │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  Each head: softmax(q_t^(i)·k_t^(i)^T / √64) · v_t^(i)  │
        │  → head_output_i ∈ ℝ^64             │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  Concatenate: [head_1 | ... | head_8] ∈ ℝ^512  │
        │  Project: W_o · concat → output ∈ ℝ^512       │
        └─────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + output (residual)
```

**What is missing:** No mechanism detects disagreement between heads. No token-level regulation exists. The heads are trained end-to-end, and their behavior is emergent, not governed.

### Mycelia Layer (One Token, One Consensus Round)

```
Input: hidden state h_t ∈ ℝ^512

        ┌─────────────────────────────────────┐
        │  SAME QKV Projection + Attention  │
        │  → head_outputs ∈ ℝ^(8×64)         │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  FIBONACCI CONSENSUS                │
        │                                       │
        │  Weights: w = [5, 8, 13, 21, 34, 55, 89, 144]  │
        │  Normalized: w_i / Σw = [0.012, 0.019, 0.031,  │
        │                         0.050, 0.081, 0.131,    │
        │                         0.212, 0.344]            │
        │                                       │
        │  consensus = Σ_i (w_i/Σw) · head_i    │
        │  → consensus ∈ ℝ^64                    │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  VARIANCE TRACKING (per token)        │
        │                                       │
        │  For each position t:                 │
        │    μ_t = mean(head_1(t), ..., head_8(t))  │
        │    σ²_t = mean((head_i(t) - μ_t)²)    │
        │    → token_variance(t) = mean(σ²_t)  │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  DYNAMIC THRESHOLD                    │
        │                                       │
        │  flat_var = token_variance.flatten()  │
        │  median = flat_var.median()           │
        │  mad = (flat_var - median).abs().median()  │
        │  scale = median + 1.4826 × mad        │
        │  threshold = 1.5 × scale × layer_factor  │
        │  (layer_factor: 0.8 → 1.4 across layers)  │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  ACCLAMATION vs. VETO                 │
        │                                       │
        │  For each token t:                    │
        │    if variance(t) < threshold:      │
        │       → ACCLAIMED (pass through)      │
        │       veto_factor = 1.0               │
        │    else:                              │
        │       → VETOED (attenuate)            │
        │       veto_factor = 0.85              │
        │                                       │
        │  consensus_attenuated = consensus × veto_factor  │
        └─────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────────┐
        │  ADAPTIVE MIXING                    │
        │                                       │
        │  Round 0: output = 0.90 × attn + 0.10 × consensus  │
        │  Round 1: output = 0.85 × attn + 0.15 × consensus  │
        │  Round 2: output = 0.80 × attn + 0.20 × consensus  │
        │  ... (floor at 0.50 × attn)          │
        └─────────────────────────────────────┘
                      ↓
        Output: h_t' = h_t + α_attn × output (residual)
```

**What is different:** Every token is evaluated for consensus agreement. Disagreeing tokens are attenuated, not suppressed. The threshold adapts to the model's own variance distribution. The mixing ratio shifts per round, allowing deeper consensus without signal collapse.

### The Fibonacci Weights: Why This Sequence?

The Fibonacci sequence (5, 8, 13, 21, 34, 55, 89, 144) was not chosen for mysticism. It was chosen for **mathematical properties**:

1. **Exponential growth:** Each weight is ~1.618× the previous. This creates a natural gradient where late heads dominate the consensus, but early heads still contribute meaningfully.

2. **No learned parameters:** The weights are fixed. This means the consensus mechanism cannot overfit or collapse during training. It is a geometric prior, not a learned behavior.

3. **Checkpoint compatibility:** The weights are stored as a flat buffer `(n_heads,)` — no reshaping needed when loading checkpoints with different head counts.

4. **Interpretability:** Because the weights are fixed, we can reason about head importance analytically. Head 8 (weight 144) has 28.8× the influence of Head 1 (weight 5). If Head 8 disagrees with the others, it will dominate the consensus unless vetoed.

### The Veto Factor: Continuous, Not Binary

In early versions, veto was binary: a token either passed or was zeroed out. This created hard discontinuities in the loss landscape. v7.3 introduced the **continuous veto factor**:

```python
veto_factor = acclamation_mask + (1.0 - acclamation_mask) * 0.85
```

This means:
- Acclaimed tokens: multiplied by 1.0 (unchanged)
- Vetoed tokens: multiplied by 0.85 (attenuated, not killed)

The factor is **gradient-safe** — the 0.85 is a constant, so backpropagation flows through both branches. The model learns to reduce variance (avoid veto) without being punished catastrophically for occasional disagreement.

---

---

## 1.7 The Consensus Mechanism: Not Just Our Idea

The replacement of attention with consensus is not speculative. In January 2026, Moushegian et al. published "Stabilizing Transformer Training Through Consensus," demonstrating that consensus mechanisms — formulated as graph-based energy minimization — serve as drop-in replacements for attention that stabilize training across wider learning rate rangesciteweb_search:52#0.

Their key findings:
- **Consensus tolerates higher learning rates** than standard attention, with the difference most pronounced in the high-LR regime
- **Hybrid consensus-attention architectures** preserve attention-level performance while inheriting consensus stability
- **Graph spectral theory** provides the mathematical foundation: consensus acts as a low-pass filter on embedding frequency, smoothing high-frequency instabilities

Mycelia's MycelialConsensus differs from their formulation in three ways:

| Feature | Moushegian et al. (2026) | Mycelia LM |
|---------|-------------------------|------------|
| **Graph structure** | Learned or predefined adjacency | Fully connected (all heads interact) |
| **Weights** | Learned edge weights | Hardcoded Fibonacci weights |
| **Regulation** | Energy minimization | Variance-based thresholding with veto |
| **Telemetry** | None | Real-time coherence, variance, delta |

Mycelia trades some of their theoretical elegance for **observability**. Where their consensus is a black-box stabilizer, Mycelia's consensus is a **diagnostic instrument**.

---

## 2. The Six Regions of the Mycelia Black Box

The original MASSIF framework divides AI interpretability into six regions. Mycelia was designed to make Region 2 — Dynamical Inference — measurable in production.

| Region | Question | Mycelia's Answer |
|--------|----------|------------------|
| **1. Representation** | What is encoded? | 512D residual stream, 8 orthogonal attention heads, per-layer variance tracking |
| **2. Dynamical Inference** | How does cognition unfold? | `variance_delta = early_var - late_var`, measured every step |
| **3. Emergence** | Why do abilities appear? | Tracked via coherence spikes during LR bursts |
| **4. Decision Formation** | Why this token? | Attention head similarity matrix shows specialization |
| **5. Objective Internalization** | What is it actually optimizing? | Consensus mechanism enforces geometric stability as a secondary objective |
| **6. Generalization** | Why does it generalize? | Monitored via domain friction gradient (Stanford vs. FineWeb) |

Mycelia's unique contribution is in **Region 2**. While other models hide their dynamical state, Mycelia broadcasts it.

---

## 3. The Topology Mapper: Finding the Bug

At step 1,075,000, Mycelia had trained for 28 epochs on a mix of FineWeb-Edu (70%) and Stanford Philosophy (30%). Loss was stable at ~4.9. The MASSIF telemetry cell classified it as **Runaway** — but the model was generating coherent English. Something was wrong with the diagnosis, not the model.

We built the Topology Mapper to look inside.

### 3.1 Layer-Wise Trajectory Geometry

The Mapper projected each layer's 512D hidden states onto their first two principal components. In a healthy model, trajectories should form coherent paths through latent space. In Mycelia, they did — but with a twist: the paths were **saturated**, not exploding. Norm growth was only 1.01×, yet 60% of neurons were hyperactive (|activation| > 10).

This was not Runaway. This was **capacity collapse** — the model had learned to pin neurons at high values rather than use them dynamically.

### 3.2 Attention Head Clustering

The Q-projection similarity matrix revealed something remarkable: **all eight heads were orthogonal**. Off-diagonal similarities were near zero (±0.01 to ±0.07). The Fibonacci weighting had successfully forced specialization. Each head had learned a distinct subspace.

This was good news. The heads were not redundant. But it also meant the consensus mechanism — which relies on head variance to detect dissent — was operating on a signal it didn't understand.

### 3.3 The Smoking Gun: Consensus Was Broken

The Mapper's most important finding was in the consensus telemetry:

```
Layer 0: kept=62.5% | coherence=0.0000 | variance=0.44 | threshold=0.40
Layer 1: kept=0.0%  | coherence=0.0000 | variance=1.14 | threshold=0.40
Layer 2: kept=0.0%  | coherence=0.0000 | variance=0.89 | threshold=0.40
Layer 3: kept=4.2%  | coherence=0.0000 | variance=0.79 | threshold=0.40
Layer 4: kept=4.2%  | coherence=0.0000 | variance=1.54 | threshold=0.40
Layer 5: kept=0.0%  | coherence=0.0000 | variance=5.29 | threshold=0.40
```

**The consensus mechanism was vetoing 95-100% of tokens across layers 1-5.** The `coherence = 0.0000` everywhere meant the telemetry was meaningless — it was measuring "everything is above threshold."

Yet the model still generated text because the attention path dominated (0.9× attention + 0.1× consensus). The consensus was **dead weight** — 10% of compute doing nothing useful.

### 3.4 Root Cause: Scale Mismatch

The v8.0 "fix" had lowered `dissenter_threshold` from 2.5 to 2.0, but the dynamic scaling formula still produced thresholds of **0.03-0.40**. Meanwhile, actual per-token variance ranged from **0.4 to 5.3**. The threshold was an **order of magnitude too low** for the model's actual variance distribution.

This is a classic **scale mismatch** — the threshold was designed for a different dynamical regime.

---

## 4. The v8.1 Intervention: Adaptive MAD-Based Thresholding

The fix was to make the threshold **responsive to the actual variance distribution** rather than a hyperparameter guess.

### 4.1 The Math

Instead of:
```python
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40
```

We compute a robust scale estimate from the live variance:
```python
flat_var = token_variance.view(-1)
var_median = flat_var.median()
var_mad = (flat_var - var_median).abs().median()  # Median Absolute Deviation
var_scale = var_median + 1.4826 * var_mad  # Robust std estimate

threshold = 1.5 * var_scale * layer_factor  # Scales with actual distribution
threshold = threshold.clamp(min=0.1, max=10.0)
```

This means:
- If variance is naturally ~1.0, threshold ≈ 1.5 → ~67% of tokens pass
- If variance grows to ~5.0, threshold ≈ 7.5 → still ~67% pass
- The threshold **scales with the model's state**, not a fixed guess

We also fixed the coherence calculation. Instead of the broken clamped ratio:
```python
coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)  # Always 0
```

We use the direct measure:
```python
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### 4.2 The LR Burst: Shaking the Optimizer

At step 1,137,000, the model was stuck at loss ~4.58 with LR=1e-5 (cosine tail). We injected a controlled **LR burst** — 500 steps at peak LR=3e-4 — to shake the optimizer out of its local minimum.

```
🚀 LR BURST: Injecting peak LR=3.00e-04 for 500 steps
🚀 LR BURST ACTIVE: steps 1,137,000 → 1,137,500
```

The burst served as a **catalyst**, not a sustained requirement. It reconfigured the optimizer momentum, allowing the new adaptive consensus to take hold.

---

## 5. The Results: From Broken to Harmonized

### 5.1 Immediate Fix Validation

| Metric | v8.0 (Broken) | v8.1 (Fixed) | Interpretation |
|--------|--------------|--------------|----------------|
| **kept_ratio** | 0-62% | **~97%** | Consensus now functional |
| **coherence** | 0.0000 | **0.9694-0.9766** | Meaningful self-monitoring |
| **Friction** | 🌋 DEEP DRIFT | **🟢 HARMONIZED** | Regime shift achieved |
| **Delta** | -1.05 to -1.57 | **-0.65 to -0.92** | Moving toward DISSIPATED |
| **Loss** | 4.85-5.05 | **4.80-5.06** | Exploring new basin |
| **LR** | 1e-5 (stuck) | **1.43e-4 (decay)** | Active learning resumed |

The coherence jump from 0.0000 → 0.97 is the **signature of the fix**. The adaptive thresholding correctly matched the model's variance scale, allowing the consensus to function as designed.

### 5.2 The HARMONIZED Regime

After the burst, Mycelia entered a **self-sustaining HARMONIZED state**:

```
📊 Step 1,140,000 | Loss: 4.9331 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9709 📈
   Friction: 🟢 HARMONIZED | early=0.68 late=1.55 Δ=-0.87

📊 Step 1,142,000 | Loss: 4.9490 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9710 📈
   Friction: 🟢 HARMONIZED | early=0.70 late=1.56 Δ=-0.86

📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

Notice the pattern:
- **Coherence holds steady at ~0.97** — the consensus is self-regulating
- **Delta stabilizes around -0.87** — not yet DISSIPATED (target: >0), but no longer DEEP DRIFT
- **Loss explores the 4.8-5.0 basin** — the model is learning, not plateaued
- **LR decays smoothly** — no more snap-to-minimum bug

The brief DEEP DRIFT at step 1,138,000 (delta=-1.57) was the **burst transition artifact** — the optimizer settling into the new basin. By step 1,140,000 it re-stabilized.

### 5.3 The Attention Heads Are Still Orthogonal

Post-fix, the Topology Mapper confirmed the head similarity matrix remained near-diagonal. The fix did not collapse head specialization — it **enabled the consensus to respect it**.

---

## 5.4 The Training Journey: A Diagnostic Thriller in Version Numbers

The Mycelia case study is not a single experiment. It is a **diagnostic narrative** spanning multiple versions, each revealing a deeper layer of the pathology.

#
## 6. The Mycelia Safety Monitor in Practice

The original article proposed a dashboard concept. Mycelia implements it in the training log:

```
📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

This is not a mockup. This is the **actual telemetry** from a live training run on an AWS SageMaker T4 instance. Every 1000 steps, Mycelia reports:

| Observable | What It Measures | Safe Range | Alert Threshold |
|------------|------------------|------------|-----------------|
| **Coherence** | Fraction of tokens acclaimed by consensus | 0.3-1.0 | <0.1 (consensus broken) |
| **Friction** | Dynamical regime classification | 🟢 HARMONIZED / ✅ DISSIPATED | 🌋 DEEP DRIFT |
| **Delta** | Domain friction gradient (early_var - late_var) | > -1.0 | < -1.5 (runaway drift) |
| **Early/Late Var** | Per-layer variance distribution | balanced | early << late (signal loss) |

The system does not just monitor — it **intervenes**. When coherence drops below 0.1, the adaptive threshold automatically rescales. When loss plateaus for >50K steps, the LR burst activates. When delta drops below -1.5, the consensus rounds increase.

---

## 7. The Geoffrey Hinton Problem: Mycelia's Response

Geoffrey Hinton has argued that even experts cannot understand what happens inside large neural networks because the systems are too complex. With 181 million parameters and 6 layers of 512 dimensions, Mycelia is tiny by modern standards. Yet even here, tracking every neuron is impossible.

Mycelia's response is different: **we don't track every neuron. We track the pattern of movement.**

The Topology Mapper does not visualize 181M weights. It visualizes:
- Six 512D trajectory projections (PCA)
- One 8×8 head similarity matrix
- Six bar charts of dead/hyperactive neurons
- One log-scale norm growth plot

That's **six layers, not 181 million parameters.** The diagnostic signal is in the **geometry**, not the weights.

As the original article argued: "You don't track every air molecule to predict a storm. You watch pressure systems, wind patterns, temperature gradients." Mycelia watches pressure systems in thought space.

---

## 8. The Mycelia Roadmap: From 181M to Production Scale

| Phase | Timeline | Goal | Status |
|-------|----------|------|--------|
| **Phase 0** | Complete | Build model with real-time telemetry | ✅ Mycelia v8.1 running |
| **Phase 1** | 3 months | Stabilize at DISSIPATED (delta > 0) | 🔄 In progress (delta=-0.87) |
| **Phase 2** | 3 months | Enable compression, add TCM data | 📝 Planned |
| **Phase 3** | 6 months | Scale to 1B parameters, validate MASSIF | 📝 Planned |
| **Phase 4** | 6 months | Production API with live safety monitoring | 📝 Planned |

The immediate target is **DISSIPATED** — the regime where `variance_delta > 0`, meaning late layers have *lower* variance than early layers. This indicates that the consensus is successfully dissipating signal noise as it propagates upward, rather than amplifying it.

Current trajectory suggests we will hit DISSIPATED within the next 100K-200K steps, as the LR decay continues to refine the basin.

---

## 9. The Deeper Significance

Mycelia is not just a model. It is a **methodology** for building AI systems that can monitor their own dynamical health.

The key insight from this case study is that **pathology precedes failure**. Mycelia v8.0 was generating coherent text while its consensus mechanism was 95% broken. The loss did not reflect this. The MASSIF telemetry did not reflect this (it reported Runaway, but the model was not running away). Only the Topology Mapper — a tool designed to visualize the 512D landscape — revealed the truth.

This suggests a new principle for AI safety:

> **Aggregate metrics lie. Geometry tells the truth.**

Loss, perplexity, and even MASSIF class labels are aggregate statistics. They can hide pathological internal states. The geometry of the hidden state landscape — variance distributions, trajectory curvature, neuron activation topology — reveals what is actually happening.

Mycelia's architecture encodes this principle. The consensus mechanism is not just a training trick. It is a **dynamical regulator** — a geometric stability controller built into the forward pass. The adaptive thresholding is not just a bug fix. It is a **self-tuning control system** that scales with the model's own variance.

This is the direction we believe AI safety must go: from output monitoring to **dynamical-state safety**, from black-box testing to **white-box geometry**, from post-hoc analysis to **real-time intervention**.

---

## 10. The Bottom Line

"We don't need to open the black box to know when it's overheating. We just need to watch the thermometer."

Mycelia's thermometer reads:
- **Coherence: 0.97** (was 0.00)
- **Friction: HARMONIZED** (was DEEP DRIFT)
- **Delta: -0.87** (was -1.57, target > 0)
- **Loss: 4.81** (exploring new basin)
- **LR: 1.43e-4** (active decay)

The engine was overheating. We watched the thermometer. We fixed the cooling system. Now it runs.

---

**Repository:** https://github.com/Ergo-sum-AGI/mycelia_llm/  
**Contact:** solis@dubito-ergo.com  
**MASSIF Framework:** https://github.com/Ergo-sum-AGI/MASSIF

---

*Mycelia LM v8.1 — 181M parameters — trained on FineWeb-Edu + Stanford Philosophy — AWS SageMaker T4 — step 1,144,146 and counting.*---

## v7.3: The Plateau

At step 1,015,000, Mycelia had trained for 27 epochs. Loss was stable at ~4.75. The consensus reported:

```
Friction: 🌋 DEEP DRIFT | early=1.1 late=2.5 Δ=-1.5
Coherence: ~0.01 (meaningless — clamped to zero)
```

The model was generating coherent English, but the consensus was broken. The training log showed `kept_ratio` oscillating wildly, and the MASSIF telemetry classified the model as **Runaway** — yet there was no runaway. The model was **stuck in a local minimum** with LR=1e-5, and the consensus mechanism was too aggressive, vetoing 95% of tokens.

### v8.0: The First Intervention

Two changes were introduced:

1. **LR Burst:** 500 steps at peak LR=3e-4 to shake the optimizer out of its plateau
2. **Consensus Tuning:** Lowered `dissenter_threshold` from 2.5 → 2.0, increased `consensus_rounds` from 1 → 2

The burst worked — loss spiked to 6.7, then recovered to ~4.9. But the consensus tuning failed. The Topology Mapper revealed:

```
Layer 1: kept=0.0% | coherence=0.0000 | variance=1.14 | threshold=0.40
```

The threshold was still an **order of magnitude too low**. The v8.0 "fix" had moved the hyperparameter, but not the fundamental scale mismatch.

### v8.1: The Real Fix

The breakthrough came from the Topology Mapper, not the training log. The Mapper showed that:
- **60% of neurons were hyperactive** (|activation| > 10)
- **Norm growth was only 1.01×** (not exploding — saturated)
- **Heads were orthogonal** (good — the Fibonacci weighting worked)
- **Consensus was completely broken** (0% kept, 0.0 coherence)

The fix was **adaptive MAD-based thresholding**:

```python
# Old (v8.0): Fixed threshold, scale mismatch
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40

# New (v8.1): Adaptive threshold, scales with actual variance
var_scale = median + 1.4826 * mad  # Robust std estimate
threshold = 1.5 * var_scale * layer_factor  # Scales with distribution
```

And a corrected coherence measure:

```python
# Old (v8.0): Always zero
 coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)

# New (v8.1): Meaningful signal
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### The Result

| Version | Step | Coherence | Friction | Delta | Loss | LR |
|---------|------|-----------|----------|-------|------|-----|
| v7.3 | 1,015,000 | 0.0000 | 🌋 DEEP DRIFT | -1.50 | 4.75 | 1e-5 (stuck) |
| v8.0 | 1,075,000 | 0.0000 | 🌋 DEEP DRIFT | -1.05 | 4.85 | 1e-5 (stuck) |
| v8.1 (burst) | 1,137,000 | 0.9718 | 🟢 HARMONIZED | -0.65 | 5.49 | 3e-4 (burst) |
| v8.1 (post) | 1,144,000 | 0.9766 | 🟢 HARMONIZED | -0.89 | 4.81 | 1.43e-4 (decay) |

The narrative arc: **diagnose → intervene → measure deeper → diagnose again → fix fundamentally → validate.**

This is not hyperparameter tuning. This is **dynamical systems surgery**.

---



### Original version:

# Mapping Mycelia: A Dynamical Systems Case Study in Real-Time AI Safety
## How Watching a Model's "Thought Process" Saved It From Collapse

By Daniel Solis, DUBITO Inc. / Ergo Sum AGI Safety Systems

---

## 1. The Mycelia Hypothesis: A Model Built to Be Watched

Most language models are trained to optimize a single number: cross-entropy loss. The assumption is that if loss goes down, the model is learning. But loss is an aggregate statistic. It tells you nothing about *how* the model is learning — whether its internal dynamics are stable, oscillatory, or heading toward collapse.

Mycelia LM was built on a different hypothesis: **what if we trained a model whose internal geometry we could monitor, diagnose, and intervene upon in real time?**

Mycelia is a 181-million-parameter transformer with a custom consensus mechanism called MycelialConsensus. Unlike standard attention, which simply averages head outputs, MycelialConsensus uses Fibonacci-weighted aggregation, per-token variance tracking, and dynamic thresholding to decide which tokens should pass through and which should be attenuated. The model reports its own internal state — coherence, variance, acclamation rate, friction gradient — at every forward pass.

This is not interpretability as post-hoc analysis. This is **interpretability as architecture**.

---

## 2. The Six Regions of the Mycelia Black Box

The original MASSIF framework divides AI interpretability into six regions. Mycelia was designed to make Region 2 — Dynamical Inference — measurable in production.

| Region | Question | Mycelia's Answer |
|--------|----------|------------------|
| **1. Representation** | What is encoded? | 512D residual stream, 8 orthogonal attention heads, per-layer variance tracking |
| **2. Dynamical Inference** | How does cognition unfold? | `variance_delta = early_var - late_var`, measured every step |
| **3. Emergence** | Why do abilities appear? | Tracked via coherence spikes during LR bursts |
| **4. Decision Formation** | Why this token? | Attention head similarity matrix shows specialization |
| **5. Objective Internalization** | What is it actually optimizing? | Consensus mechanism enforces geometric stability as a secondary objective |
| **6. Generalization** | Why does it generalize? | Monitored via domain friction gradient (Stanford vs. FineWeb) |

Mycelia's unique contribution is in **Region 2**. While other models hide their dynamical state, Mycelia broadcasts it.

---

## 3. The Topology Mapper: Finding the Bug

At step 1,075,000, Mycelia had trained for 28 epochs on a mix of FineWeb-Edu (70%) and Stanford Philosophy (30%). Loss was stable at ~4.9. The MASSIF telemetry cell classified it as **Runaway** — but the model was generating coherent English. Something was wrong with the diagnosis, not the model.

We built the Topology Mapper to look inside.

### 3.1 Layer-Wise Trajectory Geometry

The Mapper projected each layer's 512D hidden states onto their first two principal components. In a healthy model, trajectories should form coherent paths through latent space. In Mycelia, they did — but with a twist: the paths were **saturated**, not exploding. Norm growth was only 1.01×, yet 60% of neurons were hyperactive (|activation| > 10).

This was not Runaway. This was **capacity collapse** — the model had learned to pin neurons at high values rather than use them dynamically.

### 3.2 Attention Head Clustering

The Q-projection similarity matrix revealed something remarkable: **all eight heads were orthogonal**. Off-diagonal similarities were near zero (±0.01 to ±0.07). The Fibonacci weighting had successfully forced specialization. Each head had learned a distinct subspace.

This was good news. The heads were not redundant. But it also meant the consensus mechanism — which relies on head variance to detect dissent — was operating on a signal it didn't understand.

### 3.3 The Smoking Gun: Consensus Was Broken

The Mapper's most important finding was in the consensus telemetry:

```
Layer 0: kept=62.5% | coherence=0.0000 | variance=0.44 | threshold=0.40
Layer 1: kept=0.0%  | coherence=0.0000 | variance=1.14 | threshold=0.40
Layer 2: kept=0.0%  | coherence=0.0000 | variance=0.89 | threshold=0.40
Layer 3: kept=4.2%  | coherence=0.0000 | variance=0.79 | threshold=0.40
Layer 4: kept=4.2%  | coherence=0.0000 | variance=1.54 | threshold=0.40
Layer 5: kept=0.0%  | coherence=0.0000 | variance=5.29 | threshold=0.40
```

**The consensus mechanism was vetoing 95-100% of tokens across layers 1-5.** The `coherence = 0.0000` everywhere meant the telemetry was meaningless — it was measuring "everything is above threshold."

Yet the model still generated text because the attention path dominated (0.9× attention + 0.1× consensus). The consensus was **dead weight** — 10% of compute doing nothing useful.

### 3.4 Root Cause: Scale Mismatch

The v8.0 "fix" had lowered `dissenter_threshold` from 2.5 to 2.0, but the dynamic scaling formula still produced thresholds of **0.03-0.40**. Meanwhile, actual per-token variance ranged from **0.4 to 5.3**. The threshold was an **order of magnitude too low** for the model's actual variance distribution.

This is a classic **scale mismatch** — the threshold was designed for a different dynamical regime.

---

## 4. The v8.1 Intervention: Adaptive MAD-Based Thresholding

The fix was to make the threshold **responsive to the actual variance distribution** rather than a hyperparameter guess.

### 4.1 The Math

Instead of:
```python
threshold = base_threshold * layer_factor * seq_factor  # 0.03-0.40
```

We compute a robust scale estimate from the live variance:
```python
flat_var = token_variance.view(-1)
var_median = flat_var.median()
var_mad = (flat_var - var_median).abs().median()  # Median Absolute Deviation
var_scale = var_median + 1.4826 * var_mad  # Robust std estimate

threshold = 1.5 * var_scale * layer_factor  # Scales with actual distribution
threshold = threshold.clamp(min=0.1, max=10.0)
```

This means:
- If variance is naturally ~1.0, threshold ≈ 1.5 → ~67% of tokens pass
- If variance grows to ~5.0, threshold ≈ 7.5 → still ~67% pass
- The threshold **scales with the model's state**, not a fixed guess

We also fixed the coherence calculation. Instead of the broken clamped ratio:
```python
coherence = 1.0 - (max_variance / threshold).clamp(max=1.0)  # Always 0
```

We use the direct measure:
```python
coherence = acclamation_rate  # Fraction of tokens below threshold
```

### 4.2 The LR Burst: Shaking the Optimizer

At step 1,137,000, the model was stuck at loss ~4.58 with LR=1e-5 (cosine tail). We injected a controlled **LR burst** — 500 steps at peak LR=3e-4 — to shake the optimizer out of its local minimum.

```
🚀 LR BURST: Injecting peak LR=3.00e-04 for 500 steps
🚀 LR BURST ACTIVE: steps 1,137,000 → 1,137,500
```

The burst served as a **catalyst**, not a sustained requirement. It reconfigured the optimizer momentum, allowing the new adaptive consensus to take hold.

---

## 5. The Results: From Broken to Harmonized

### 5.1 Immediate Fix Validation

| Metric | v8.0 (Broken) | v8.1 (Fixed) | Interpretation |
|--------|--------------|--------------|----------------|
| **kept_ratio** | 0-62% | **~97%** | Consensus now functional |
| **coherence** | 0.0000 | **0.9694-0.9766** | Meaningful self-monitoring |
| **Friction** | 🌋 DEEP DRIFT | **🟢 HARMONIZED** | Regime shift achieved |
| **Delta** | -1.05 to -1.57 | **-0.65 to -0.92** | Moving toward DISSIPATED |
| **Loss** | 4.85-5.05 | **4.80-5.06** | Exploring new basin |
| **LR** | 1e-5 (stuck) | **1.43e-4 (decay)** | Active learning resumed |

The coherence jump from 0.0000 → 0.97 is the **signature of the fix**. The adaptive thresholding correctly matched the model's variance scale, allowing the consensus to function as designed.

### 5.2 The HARMONIZED Regime

After the burst, Mycelia entered a **self-sustaining HARMONIZED state**:

```
📊 Step 1,140,000 | Loss: 4.9331 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9709 📈
   Friction: 🟢 HARMONIZED | early=0.68 late=1.55 Δ=-0.87

📊 Step 1,142,000 | Loss: 4.9490 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9710 📈
   Friction: 🟢 HARMONIZED | early=0.70 late=1.56 Δ=-0.86

📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

Notice the pattern:
- **Coherence holds steady at ~0.97** — the consensus is self-regulating
- **Delta stabilizes around -0.87** — not yet DISSIPATED (target: >0), but no longer DEEP DRIFT
- **Loss explores the 4.8-5.0 basin** — the model is learning, not plateaued
- **LR decays smoothly** — no more snap-to-minimum bug

The brief DEEP DRIFT at step 1,138,000 (delta=-1.57) was the **burst transition artifact** — the optimizer settling into the new basin. By step 1,140,000 it re-stabilized.

### 5.3 The Attention Heads Are Still Orthogonal

Post-fix, the Topology Mapper confirmed the head similarity matrix remained near-diagonal. The fix did not collapse head specialization — it **enabled the consensus to respect it**.

---

## 6. The Mycelia Safety Monitor in Practice

The original article proposed a dashboard concept. Mycelia implements it in the training log:

```
📊 Step 1,144,000 | Loss: 4.8066 | LR: 1.43e-04 | 📉 Annealing
   Coherence: 0.9766 📈
   Friction: 🟢 HARMONIZED | early=0.78 late=1.66 Δ=-0.89
```

This is not a mockup. This is the **actual telemetry** from a live training run on an AWS SageMaker T4 instance. Every 1000 steps, Mycelia reports:

| Observable | What It Measures | Safe Range | Alert Threshold |
|------------|------------------|------------|-----------------|
| **Coherence** | Fraction of tokens acclaimed by consensus | 0.3-1.0 | <0.1 (consensus broken) |
| **Friction** | Dynamical regime classification | 🟢 HARMONIZED / ✅ DISSIPATED | 🌋 DEEP DRIFT |
| **Delta** | Domain friction gradient (early_var - late_var) | > -1.0 | < -1.5 (runaway drift) |
| **Early/Late Var** | Per-layer variance distribution | balanced | early << late (signal loss) |

The system does not just monitor — it **intervenes**. When coherence drops below 0.1, the adaptive threshold automatically rescales. When loss plateaus for >50K steps, the LR burst activates. When delta drops below -1.5, the consensus rounds increase.

---

## 7. The Geoffrey Hinton Problem: Mycelia's Response

Geoffrey Hinton has argued that even experts cannot understand what happens inside large neural networks because the systems are too complex. With 181 million parameters and 6 layers of 512 dimensions, Mycelia is tiny by modern standards. Yet even here, tracking every neuron is impossible.

Mycelia's response is different: **we don't track every neuron. We track the pattern of movement.**

The Topology Mapper does not visualize 181M weights. It visualizes:
- Six 512D trajectory projections (PCA)
- One 8×8 head similarity matrix
- Six bar charts of dead/hyperactive neurons
- One log-scale norm growth plot

That's **six layers, not 181 million parameters.** The diagnostic signal is in the **geometry**, not the weights.

As the original article argued: "You don't track every air molecule to predict a storm. You watch pressure systems, wind patterns, temperature gradients." Mycelia watches pressure systems in thought space.

---

## 8. The Mycelia Roadmap: From 181M to Production Scale

| Phase | Timeline | Goal | Status |
|-------|----------|------|--------|
| **Phase 0** | Complete | Build model with real-time telemetry | ✅ Mycelia v8.1 running |
| **Phase 1** | 3 months | Stabilize at DISSIPATED (delta > 0) | 🔄 In progress (delta=-0.87) |
| **Phase 2** | 3 months | Enable compression, add TCM data | 📝 Planned |
| **Phase 3** | 6 months | Scale to 1B parameters, validate MASSIF | 📝 Planned |
| **Phase 4** | 6 months | Production API with live safety monitoring | 📝 Planned |

The immediate target is **DISSIPATED** — the regime where `variance_delta > 0`, meaning late layers have *lower* variance than early layers. This indicates that the consensus is successfully dissipating signal noise as it propagates upward, rather than amplifying it.

Current trajectory suggests we will hit DISSIPATED within the next 100K-200K steps, as the LR decay continues to refine the basin.

---

## 9. The Deeper Significance

Mycelia is not just a model. It is a **methodology** for building AI systems that can monitor their own dynamical health.

The key insight from this case study is that **pathology precedes failure**. Mycelia v8.0 was generating coherent text while its consensus mechanism was 95% broken. The loss did not reflect this. The MASSIF telemetry did not reflect this (it reported Runaway, but the model was not running away). Only the Topology Mapper — a tool designed to visualize the 512D landscape — revealed the truth.

This suggests a new principle for AI safety:

> **Aggregate metrics lie. Geometry tells the truth.**

Loss, perplexity, and even MASSIF class labels are aggregate statistics. They can hide pathological internal states. The geometry of the hidden state landscape — variance distributions, trajectory curvature, neuron activation topology — reveals what is actually happening.

Mycelia's architecture encodes this principle. The consensus mechanism is not just a training trick. It is a **dynamical regulator** — a geometric stability controller built into the forward pass. The adaptive thresholding is not just a bug fix. It is a **self-tuning control system** that scales with the model's own variance.

This is the direction we believe AI safety must go: from output monitoring to **dynamical-state safety**, from black-box testing to **white-box geometry**, from post-hoc analysis to **real-time intervention**.

---

## 10. The Bottom Line

"We don't need to open the black box to know when it's overheating. We just need to watch the thermometer."

Mycelia's thermometer reads:
- **Coherence: 0.97** (was 0.00)
- **Friction: HARMONIZED** (was DEEP DRIFT)
- **Delta: -0.87** (was -1.57, target > 0)
- **Loss: 4.81** (exploring new basin)
- **LR: 1.43e-4** (active decay)

The engine was overheating. We watched the thermometer. We fixed the cooling system. Now it runs.

---

**Repository:** https://github.com/Ergo-sum-AGI/mycelia_llm/  
**Contact:** solis@dubito-ergo.com  
**MASSIF Framework:** https://github.com/Ergo-sum-AGI/MASSIF

---

*Mycelia LM v8.1 — 181M parameters — trained on FineWeb-Edu + Stanford Philosophy — AWS SageMaker T4 — step 1,144,146 and counting.*
