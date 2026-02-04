# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRLM (Tiny Recursive Language Model) is a PyTorch research codebase exploring iterative refinement techniques for parameter-efficient language modeling. The key innovation is a two-level recursion system where inner loops update a reasoning latent state and outer loops refine prediction drafts.

**Key Result**: 19.3M parameter TRM achieves 40.41 perplexity on WikiText-103 (GPT-2 with 117M params achieves 37.5 - only 7.7% worse with 83.5% fewer parameters).

## Commands

```bash
# Install dependencies
pixi install

# Main experiment (WikiText-103 scaling - flagship work)
pixi run train

# Continue training from checkpoint
pixi run continue-train

# Proof of concept (Tiny Shakespeare, fast validation)
pixi run poc

# Run any script directly
pixi run python <script_name>.py

# CPU-only environment (no CUDA)
pixi run -e cpu python bigger.py
```

## Architecture

**Two-Level Recursion System:**
```
Input → Base Transformer (6 layers) → Initial Prediction
    ↓
For each refinement (2):
    For each recursion (3):
        Refinement Layer → Updates reasoning state
    Answer Update Layer
    ↓
Final Prediction (total 8 passes: 2 + 2×3)
```

**Key Design Decisions:**
- No causal masking within chunks during refinement (bidirectional attention)
- Deep supervision: loss calculated multiple times per example with detached states
- Dual recursion: inner updates reasoning latent, outer updates prediction draft

## Key Files

| File | Purpose |
|------|---------|
| `bigger.py` | Main WikiText-103 experiment (19.3M params, 6-layer TRM) |
| `continue_bigger.py` | Resume training from checkpoint |
| `chunk_trm_2.py` | Tiny Shakespeare proof of concept (2-layer TRM) |
| `trm.py` | Core model definitions (TransformerBlock, config dataclass) |

## Critical Hyperparameters

**WikiText-103 (bigger.py):**
- Learning rate: `1e-4` (marked as CRITICAL - do not change)
- Batch size: 8, Sequence length: 512
- Model: 256 embed_dim, 1024 hidden_dim, 6 layers, 8 heads
- TRM: 2 refinements, 3 recursion depth

**Tiny Shakespeare (chunk_trm_2.py):**
- Learning rate: `1e-3`
- Block size: 64, Batch size: 128
- 2 TRM layers (vs 4 baseline layers)

## Output Structure

```
outputs/
├── checkpoints/     # Model checkpoints (every 1000 batches)
├── logs/           # Training logs
└── results.json    # Epoch metrics (train_loss, val_loss, val_perplexity)
```

## Dependencies

Managed via Pixi (`pixi.toml`). Core packages: `pytorch`, `numpy`, `datasets`, `transformers`, `tqdm`

Datasets are cached via Hugging Face (`datasets.load_dataset('wikitext', 'wikitext-103-v1')`).
