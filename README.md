# Cross-Batch Interactive Generation

A PyTorch implementation of cross-batch interaction for LLM inference. During generation, different samples in a batch can share information by mixing their hidden states, enabling collaborative inference.

## Core Idea

Traditional batch inference processes each sample independently. This project enables **cross-batch interaction** where:
- The latest token's hidden state from each sample is shared with other samples
- A learnable attention/mixing module combines information from different samples
- This allows samples to benefit from each other's context during generation

## Project Structure

```
batch-generation/
├── src/
│   ├── __init__.py                 # Package exports
│   ├── cross_batch_attention.py    # Cross-batch attention mechanisms
│   ├── cross_batch_generator.py    # Modified generation loop
│   ├── squad_eval.py               # SQuAD evaluation utilities
│   └── trainer.py                  # Fine-tuning trainer
├── main.py                         # Evaluation entry point
├── train.py                        # Training entry point
├── run_demo.py                     # Quick demo script
├── requirements.txt                # Dependencies
├── checkpoints/                    # Saved model checkpoints
└── outputs/                        # Evaluation results
```

## Core Components

### 1. CrossBatchAttention
Multi-head attention mechanism for cross-batch interaction:
- Each sample attends to other samples (not itself) in the batch
- Learnable Q, K, V projections
- Controlled mixing via learnable `alpha` parameter

### 2. CrossBatchEmbeddingMixer
Simpler similarity-based mixing:
- Computes cosine similarity between sample embeddings
- Weighted average of other samples based on similarity
- Lighter weight alternative to full attention

### 3. CrossBatchGenerator
Custom generation loop with cross-batch support:
- Extracts hidden states from specified layer
- Applies cross-batch module to mix information
- Projects mixed hidden states back to logits
- Supports both greedy and sampling decoding

### 4. CrossBatchTrainer
Fine-tuning trainer for the cross-batch module:
- Freezes base model, only trains cross-batch parameters
- Uses next-token prediction loss
- Tracks improvement over baseline (no cross-batch)

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.14.0
- tqdm >= 4.65.0

## Usage

### Quick Demo

```bash
python run_demo.py
```

### Evaluation on SQuAD

```bash
# Compare cross-batch vs standard generation
python main.py --run_comparison --max_samples 100 --batch_size 4

# Single evaluation with cross-batch
python main.py --max_samples 100 --batch_size 8
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | gpt2 | HuggingFace model name |
| `--mix_method` | attention | "attention" or "mixer" |
| `--mix_layer` | -1 | Layer to extract hidden states (-1 = last) |
| `--batch_size` | 4 | Batch size for evaluation |
| `--max_samples` | 100 | Maximum samples to evaluate |
| `--max_new_tokens` | 32 | Maximum tokens to generate |
| `--run_comparison` | False | Compare cross-batch vs standard |

### Training

```bash
# Train with default settings
python train.py

# Custom training
python train.py \
    --model_name gpt2-medium \
    --mix_method attention \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_samples 10000
```

**Training Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | gpt2 | HuggingFace model name |
| `--mix_method` | attention | "attention" or "mixer" |
| `--num_epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_samples` | 5000 | Maximum training samples |
| `--save_dir` | checkpoints | Checkpoint directory |

### Using Trained Checkpoints

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import CrossBatchGenerator, CrossBatchAttention

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create cross-batch module and load checkpoint
cross_batch_module = CrossBatchAttention(hidden_size=model.config.hidden_size)
checkpoint = torch.load("checkpoints/best_model.pt")
cross_batch_module.load_state_dict(checkpoint["cross_batch_module"])

# Create generator
generator = CrossBatchGenerator(
    model=model,
    tokenizer=tokenizer,
    cross_batch_module=cross_batch_module,
)

# Generate
prompts = ["Question 1...", "Question 2...", "Question 3..."]
outputs = generator.generate_text(prompts, max_new_tokens=50)
```

## How It Works

### Generation Loop

```
For each generation step:
1. Forward pass through frozen LLM
2. Extract hidden states from layer N
3. Get last token's hidden state for each sample: [batch, hidden]
4. Apply cross-batch module:
   - Compute attention weights between all sample pairs
   - Exclude self-attention (diagonal)
   - Mix hidden states based on attention weights
5. Project mixed hidden states to logits
6. Sample/argmax next token
7. Repeat
```

### Cross-Batch Attention

```
Input: H = [h1, h2, ..., hB]  # Hidden states from B samples

Q = W_q @ H  # Query projection
K = W_k @ H  # Key projection
V = W_v @ H  # Value projection

# Attention with self-exclusion
A = softmax(Q @ K.T / sqrt(d), dim=-1)
A[i,i] = 0  # No self-attention

# Mix and combine
H_cross = A @ V
H_out = (1 - alpha) * H + alpha * W_o @ H_cross
```

## Example Results

### Evaluation on SQuAD (50 samples, batch_size=4)

| Method | Exact Match | F1 Score |
|--------|-------------|----------|
| Standard | 4.00 | 16.75 |
| Cross-Batch (untrained) | 4.00 | 15.09 |

### Training Results (100 samples, 2 epochs)

| Epoch | Loss | Baseline Loss | Improvement |
|-------|------|---------------|-------------|
| 1 | 5.93 | 6.60 | +0.67 |
| 2 | 5.16 | 6.60 | +1.44 |

The cross-batch module learns to reduce the next-token prediction loss compared to the baseline (no cross-batch), demonstrating that information sharing between samples can be beneficial.

## Citation

If you use this code, please cite:

```bibtex
@software{cross_batch_generation,
  title = {Cross-Batch Interactive Generation},
  year = {2024},
  url = {https://github.com/your-repo/batch-generation}
}
```

## License

MIT License
