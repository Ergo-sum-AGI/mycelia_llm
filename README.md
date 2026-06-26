🧬 Architecture
Core Components
Component	Description
Transformer Decoder	3 layers, 4 attention heads, 128 embedding dim
Mycelial Consensus	Fibonacci-weighted agreement across attention heads (5, 8, 13, 21, 34, 55)
Golden Dropout	φ-optimal regularization (keep 61.8%, scale by φ=1.618)
Sinusoidal PE	Parameter-free positional encoding (no learned positions)
LCLM Compressor	8× context compression (128 → 16 latent tokens)
Key Design Choices
Sinusoidal positions instead of learned embeddings → reduces parameters, improves generalization

Mean pooling for compression → preserves information while reducing sequence length

Interleaved compression → every 4th step processes compressed contexts

No weight tying → avoids dimension mismatch between embedding and LM head

Mixed precision (AMP) → faster training, lower memory usage

📊 Data Pipeline
S3-Aware JSONL Streaming
The model streams training data directly from S3 without downloading entire datasets:

python
# Supports three data sources (priority order):
# 1. SageMaker channels (local mount)
# 2. S3 prefixes (s3://bucket/prefix/)
# 3. Fallback files (specific JSONL paths)
TCM Data Formatting
The TCMTextFormatter handles multiple TCM dataset schemas:

ShenNong/MedChat → question/answer fields

Generic TCM → input/output or query/response

Chat template → <|im_start|>user/assistant markers

Curriculum Mixing
Hybrid dataset mixes 90% TCM data with 10% Mycelial curriculum:

Curriculum Category	Example
Algebra	Expand: (x + 12)² = x² + 24x + 144
Recursion	observe observe observe observe
Consensus	Agent A: x=5\nAgent B: x=5\nAgent C: x=5
Contradiction	Agent A: x=3\nAgent B: x=4
Impermanence	The river changes. No moment is identical.
🔬 MASSIF Telemetry
What It Monitors
Metric	Symbol	Description
Persistence	
I
t
I 
t
​
 	Velocity autocorrelation — tracks trajectory curling
Flip Rate	-	When 
I
t
>
0
I 
t
​
 >0 (loss of corrective ability)
Hidden Norm	
∥
∥
h
t
∥
∥
∥∥h 
t
​
 ∥∥	Tracks runaway expansion
Coherence	-	Consensus among attention heads
Dubito	-	Paradox/uncertainty score
Alignment	R_t
​
End-to-end pre-training ✅ (current)

Supervised fine-tuning (planned)

SageMaker Channels
When running on SageMaker:

SM_CHANNEL_TRAIN → mounted training data

SM_CHANNEL_VALIDATION → mounted validation data

SM_MODEL_DIR → checkpoint save location

🚀 Usage
Quick Start (SageMaker)
bash
# Install dependencies
pip install boto3 transformers datasets torch tqdm

# Run training
python mycelia_v7_1.py
Quick Start (Colab)

# Replace S3 config with local path
config.s3_bucket = "my-bucket"
config.s3_prefix_train = "massif-llm/train"
config.s3_prefix_val = "massif-llm/val"

# Or use fallback files
config.s3_fallback_files = ["local_file.jsonl"]
Resume Training
The checkpoint manager automatically detects mycelia_latest.pt and resumes from the last saved global step.

🧪 Sample Output
Prompt: "患者发热恶寒，头痛身疼，舌苔薄白，脉浮紧"

Output: The model generates TCM-style reasoning, drawing from ShenNong corpus patterns.

Prompt: "The mycelial network reaches consensus when"

Output: The model responds in a blend of Mycelial philosophy and TCM reasoning.

📈 Training Progress (Example)
text
📊 Step 0x01a2b | Loss: 2.8456 | Dubito: 0.51 | Coh: 0.863
   VRAM: 1.43GB
💾 Checkpoint saved at global step 6699 (0x01a2b)
🔮 Future Work
Multi-stage training (adapter warmup → encoder → full model → SFT)

Auxiliary reconstruction task (improves compression quality)

TCM-specific validation (accuracy on Chinese medical QA)

Agentic expansion (expand compressed segments on demand)

Victorian → TCM transfer learning (using PG19 as bridge)

🙏 Acknowledgments
Qwen tokenizer for Chinese/English support

ShenNong TCM corpus for domain data

MASSIF framework for telemetry design

AWS SageMaker for training infrastructure

📝 License
Apache 2.0

Built with 🍄 by Daniel Solis — AGI Safety Systems, Dubito Inc.
S3 TCM pipeline active. ShenNong corpus streaming.

# ADVANCEMENTS

🍄 Mycelia Architecture: Dynamic Compression & LWES
Executive Summary
The Mycelia architecture implements a self-optimizing training system that dynamically adjusts compression frequency based on training progress, while simultaneously tracking a Loss-Weighted Efficiency Score (LWES) to find the optimal balance between VRAM savings and model accuracy.

1. Dynamic Compression
The Problem
Approach	Issue
Always ON	Saves VRAM but hurts accuracy (28% loss penalty)
Always OFF	Good accuracy but wastes VRAM
Fixed schedule	Misses the "sweet spot"
The Solution
Compression probability decays from 80% → 5% over the course of training:

text
Step 0:     80% chance of compression  ← Save VRAM early
Step 50,000: 5% chance of compression   ← Prioritize accuracy later
The Formula
python
def get_compression_probability(step, total_steps=50000, start_prob=0.8, end_prob=0.05):
    decay = min(1.0, step / total_steps)
    prob = start_prob - (start_prob - end_prob) * decay
    return max(end_prob, prob)
Visualization
text
Compression Probability
  0.80 ┤ ████████████████░░░░░░░░  ← Early: 80%
  0.60 ┤ ██████████████░░░░░░░░░░
  0.40 ┤ ██████████░░░░░░░░░░░░░░
  0.20 ┤ ██████░░░░░░░░░░░░░░░░░░
  0.05 ┤ ██░░░░░░░░░░░░░░░░░░░░░░  ← Late: 5%
        └─────────────────────────
        0        25,000      50,000
                   Steps
2. LWES (Loss-Weighted Efficiency Score)
The Trade-off
text
VRAM Savings (Good)  ←───────→  Loss Increase (Bad)
The Formula
text
LWES = (Loss_Uncompressed / Loss_Compressed)^γ × (Cumulative_GB / β + 1)
Components
Component	What It Measures	When It's High
Loss Fidelity Ratio	How much compression hurts accuracy	Compression loss close to uncompressed
VRAM Utility Bonus	How much memory was saved	Lots of VRAM saved
γ (Gamma=3.0)	Sensitivity to loss increases	Penalizes loss degradation
β (Beta=10.0)	Normalizes VRAM savings	10GB saved = 2× utility bonus
Score Interpretation
LWES Score	Meaning	Action
< 0.5	❌ Compression penalty too high	Reduce compression
0.5 - 1.0	⚠️ Suboptimal equilibrium	Tune parameters
1.0 - 1.5	✅ Sweet spot achieved	Continue training
> 1.5	🌟 Optimal performance	Lock checkpoint
3. The Self-Optimizing System
Architecture Overview
text
┌─────────────────────────────────────────────────────────────────┐
│                    MYCELIA ARCHITECTURE                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LEVEL 2: EXPERT CONSENSUS (Macro)                     │   │
│  │  └── Four experts with different compression freqs    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LEVEL 1: HEAD CONSENSUS (Micro)                       │   │
│  │  └── 8 heads with Fibonacci weights                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DYNAMIC COMPRESSION                                    │   │
│  │  └── Probability decays from 80% → 5%                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LWES TRACKER                                           │   │
│  │  └── Score climbs as model recovers                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ARCHITECT CONTROLLER                                   │   │
│  │  └── Auto-saves best checkpoints                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
The Controller
python
class MyceliaArchitectController:
    def __init__(self, checkpoint_dir, patience_steps=1500, min_score_target=0.35):
        self.best_sweet_spot_score = -float('inf')
        self.best_step = 0
        self.steps_without_improvement = 0
        self.early_stop_triggered = False
    
    def evaluate_and_action(self, current_step, model, optimizer, sweet_spot_score, ...):
        # Saves checkpoint when new sweet spot found
        # Triggers early stopping if score drops below threshold
        # Tracks patience for stagnation
4. Results from Training
LWES Score Progression
Step	LWES	Milestone
400,954	0.4494	First sweet spot
403,525	0.5812	Climbing
404,735	0.6738	Breaking 0.6
404,855	0.7169	Breaking 0.7
404,885	0.8108	Breaking 0.8
404,905	0.8577	🏆 New best!
Loss Recovery
Step	Loss	Note
401,235	5.7955	Resume training
403,000	5.7480	Dropping
405,000	4.8925	Huge drop
405,065	4.3474	Fantastic recovery
Compression Report
Metric	Value
Compression ON steps	82
Compression OFF steps	3,749
Compression ON loss	7.4516
Compression OFF loss	5.9338
Loss Ratio	1.2558 (25.6% penalty)
5. Key Takeaways
What We've Learned
Dynamic Compression balances VRAM savings with accuracy

LWES provides a single metric for the compression trade-off

The score climbs as the model recovers from architecture changes

The system self-optimizes without manual intervention

Why This Matters
✅ Runs on a single T4 GPU (no expensive clusters)

✅ Transparent metrics show exactly why decisions are made

✅ Self-optimizing — no hyperparameter tuning needed

✅ Early stopping prevents wasting compute on degraded models

The Big Picture
text
Traditional MoE          →  Mycelia Approach
─────────────────────────────────────────────────
Experts compete          →  Experts collaborate
Opaque routing           →  Transparent metrics
Black-box decisions      →  We know why decisions are made
Static compression       →  Dynamic, self-optimizing
No efficiency tracking   →  LWES finds the sweet spot
6. Code Snippets
Dynamic Compression Scheduler
python
def get_compression_probability(step, total_steps=50000, start_prob=0.8, end_prob=0.01):
    decay = min(1.0, step / total_steps)
    prob = start_prob - (start_prob - end_prob) * decay
    return max(end_prob, prob)
LWES Calculator
python
def calculate_lwes_sweet_spot(loss_compressed, loss_uncompressed, cumulative_gb, 
                               gamma=3.0, beta=10.0):
    if loss_compressed == 0 or loss_uncompressed == 0:
        return 0.0
    loss_fidelity_ratio = loss_uncompressed / loss_compressed
    penalty_component = loss_fidelity_ratio ** gamma
    vram_utility_bonus = (cumulative_gb / beta) + 1.0
    return penalty_component * vram_utility_bonus
Controller Integration
python
# Initialize controller
controller = MyceliaArchitectController(
    checkpoint_dir=CKPT_DIR,
    patience_steps=1500,
    min_score_target=0.35
)

# During training, evaluate sweet spot
if len(comp_on_losses) >= 30 and len(comp_off_losses) >= 30 and sweet_spot_score > 0:
    stop_training, _ = controller.evaluate_and_action(
        current_step=global_step,
        model=model,
        optimizer=opt,
        sweet_spot_score=sweet_spot_score,
        loss=avg,
        savings_gb=cumulative_gb
    )
7. Quick Reference
Concept	Formula / Definition
Dynamic Compression	P(compression) = 0.8 → 0.01 over training
LWES	(Loss_off / Loss_on)^3 × (GB_saved/10 + 1)
Optimal LWES	> 1.5
Compression Penalty	(Loss_on / Loss_off - 1) × 100%
Early Stop	LWES < 0.20 for > 1500 steps
8. Summary Statement
"Dynamic compression saves VRAM when it matters, LWES tells us when we've found the perfect balance, and the score climbs as the model recovers its accuracy."

🍄 Mycelia Architecture v7.2 — Self-Optimizing Training System
