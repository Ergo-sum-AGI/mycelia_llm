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

