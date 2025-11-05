# LLM: Iterative Refinement Research

Exploring recursive refinement for language model generation. Testing whether predicting multiple tokens at once and refining them together can improve over standard autoregressive generation.

## Key Results

Testing Tiny Recursive Model (TRM) on character-level Shakespeare:

- **TRM**: Validation perplexity 1.01 (nearly perfect)
- **Baseline**: Validation perplexity 55.08 (significant overfitting)
- **Same model size**: ~108K parameters
- **Key finding**: TRM achieves perfect training (loss → 0.0) while maintaining excellent validation (perplexity → 1.01)

This is a small-scale proof of concept. Whether it scales to real models and tasks is unknown.

## Concept

**Standard autoregressive generation:**
- Predict one token at a time
- Commit immediately
- Can't revise based on future tokens

**TRM approach:**
- Predict multiple tokens simultaneously
- Refine them together in embedding space
- Only convert to discrete tokens at the end

The hypothesis: Models can learn to iteratively improve predictions rather than just directly predict them.

## Project Structure

### Main Experiments

- `chunk_trm_2.py` - Working TRM implementation with 2-layer architecture
- `chunk_trm.py` - Original TRM experiments
- `trm.py` - Core TRM model definitions

### Baseline Comparisons

- `tiny_shakespeare_recursive_*.py` - Various recursive model experiments
- `iterative_refinement_experiment*.py` - Scaling experiments with different layer counts

### Other Experiments

- `recursive_wikitext2.py` - Testing on WikiText-2 dataset
- `sweet.py` - Sweet spot analysis for hyperparameters
- `gpu_speed_test.py` - Performance benchmarking

### Results Directories

- `outputs/` - Training outputs and logs
- `*_results/` - Experimental results and analysis

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib

# Download Tiny Shakespeare dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tiny_shakespeare.txt
```

## Usage

### Run TRM Experiment

```bash
python chunk_trm_2.py
```

Expected output:
- TRM validation perplexity: ~1.01
- Baseline validation perplexity: ~55.08
- Training time: ~5-10 minutes on CPU

### Key Parameters

In the code, you can adjust:
- `n_refinements`: Number of refinement rounds (default: 3)
- `n_recursions`: Recursions per refinement (default: 6)
- `chunk_size`: Tokens to predict simultaneously (default: 2)
- `context_size`: Context window (default: 64)

## How It Works

### Architecture

**Two levels of recursion:**
1. **Inner (Recursion)**: Updates reasoning latent `z`
   - 6 passes building up "chain of thought"
2. **Outer (Refinement)**: Updates answer draft `y`
   - 3 passes progressively improving the prediction

**Total computation**: 21 forward passes per example
- 3 refinements × 6 recursions = 18 reasoning updates
- 3 refinements × 1 answer update = 3 answer updates

### Training: Deep Supervision

Key innovation: Calculate loss multiple times per example using the same ground truth.

```python
for supervision_step in range(4):
    draft, reasoning = model(context, draft, reasoning)
    loss = compute_loss(draft, ground_truth)
    loss.backward()
    optimizer.step()
    draft, reasoning = draft.detach(), reasoning.detach()
```

This teaches the model to refine rather than memorize.

### No Causal Masking Within Chunks

Unlike standard transformers, tokens within a chunk can see each other bidirectionally during refinement. This enables mutual correction.

## Results Analysis

### Why No Overfitting?

TRM achieves:
- Training loss: 0.0000 (perfect memorization)
- Validation perplexity: 1.01 (excellent generalization)

Hypothesis: The model learns a **refinement skill** rather than memorizing sequences. This appears to generalize differently than direct prediction.

### Limitations

This is exploratory research on a toy problem:
- Tiny dataset (1MB Shakespeare)
- Tiny model (108K parameters)
- Tiny chunks (2 characters)
- Character-level only
- Single task

Whether this scales to real applications is unknown and requires further research.

## Publications

📝 Medium article: [Testing TRM on Tiny Shakespeare](https://medium.com/@mbonsign)

## Future Work

Open questions:
1. Does this scale to larger chunks (8, 16, 32 tokens)?
2. Does this scale to large models (millions/billions of parameters)?
3. Does this help with reasoning, math, or code generation?
4. Can we learn adaptive stopping (when to stop refining)?
5. Does this work with modern architectures (rotary embeddings, etc.)?

## Citation

If you use this work, please reference:

```
@misc{bonsignore2025trm,
  title={Testing TRM on Tiny Shakespeare: Recursive Refinement for Language Models},
  author={Bonsignore, Michael},
  year={2025},
  url={https://github.com/MikeyBeez/llm}
}
```

Built on ideas from "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

## License

MIT License - see LICENSE file for details

## Contact

- GitHub: [@MikeyBeez](https://github.com/MikeyBeez)
- Medium: [@mbonsign](https://medium.com/@mbonsign)

---

**Status**: Experimental research - proof of concept stage
