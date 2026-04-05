# Multi-Shape NCA: One Rule, Many Shapes

## Overview
This experiment tests whether a single NCA with one set of weights can grow multiple
distinct shapes from different learned seed embeddings. Previous work (Mordvintsev 2020)
trains one NCA per shape. Here, one NCA learns to grow 5 different shapes — circle,
square, plus, triangle, line — conditioned only on a 4-channel seed vector placed in
the center cell. Two architectures are compared: Mordvintsev's Sobel + 1x1 conv approach
and a general-purpose stacked conv architecture used across all other experiments in
this repo.

## Key Question

**Can one NCA local rule, conditioned on a learned seed embedding, grow multiple distinct shapes? And can a general-purpose stacked conv architecture match specialized morphogenesis architecture on this task?**

## Architecture

### Shared Setup (both architectures)
- **Grid**: `(1, 25, 40, 40)` — 1 batch, 25 channels, 40x40 cells
  - Channel 0: visible output (compared against target)
  - Channels 1-20: hidden state for growth computation
  - Channels 21-24: seed embedding (never updated during forward pass)
- **Seed**: `nn.Embedding(5, 4)` — learned lookup table, 5 shapes, 4 values each
  - Placed at center cell (20, 20) at step 0
  - Backprop learns seed vectors alongside NCA weights
- **Steps**: Random 64-96 per training example (forces temporal stability)
- **Stochastic mask**: 50% of cells randomly skip each step
- **Update scaling**: multiply by 0.1 to prevent explosion over 80+ steps
- **Loss**: BCEWithLogitsLoss on channel 0 vs target shape
- **Gradient clipping**: max norm 1.0

### Architecture A: Mordvintsev (Sobel + 1x1 conv)
- **Perception**: 3 fixed filters (identity, sobel_x, sobel_y) applied to all 25
  channels = 75 perception channels. Not learned — hardcoded edge detection.
- **MLP**: Two 1x1 convolutions (75→128 with ReLU, 128→21 zero-initialized)
- **Optimizer**: Adam, lr=0.002
- **Training**: 100,000 iterations

### Architecture B: Stacked Conv (general purpose)
- **Conv1**: 25→128 channels, 3x3 kernel, ReLU (learned perception + thinking)
- **Conv2**: 128→128 channels, 1x1 kernel, ReLU (deeper per-cell processing)
- **Conv3**: 128→21 channels, 1x1 kernel, zero-initialized (decide update)
- **Optimizer**: AdamW, lr=0.0005, weight_decay=0.01
- **Training**: 100,000 iterations

### Key Architectural Difference
Mordvintsev hardcodes perception with Sobel filters optimized for spatial gradients.
Stacked conv learns its own perception from scratch. Mordvintsev is specialized for
morphogenesis. Stacked conv is the same architecture used for binary addition, heat
diffusion, Maxwell's equations, Game of Life, and sorting throughout this repo.

## Files

### Training Scripts

**`multi_shape_nca.py`**
- Mordvintsev architecture: Sobel perception + 1x1 conv MLP
- Trains on 5 shapes with learned seed embeddings
- 100,000 iterations, Adam lr=0.002
- Outputs: nca_weights.pth

**`stacked_conv_nca.py`**
- Stacked conv architecture: Conv3x3 + Conv1x1 + Conv1x1
- Trains on 5 shapes with learned seed embeddings
- 100,000 iterations, AdamW lr=0.0005
- Outputs: stacked_conv_weights.pth

### Testing Scripts

**`test_multi_shape.py`**
- Loads nca_weights.pth, grows all 5 shapes, reports pixel accuracy
- Generates comparison visualization (target vs grown)

**`test_stacked_conv.py`**
- Loads stacked_conv_weights.pth, grows all 5 shapes, reports pixel accuracy
- Generates comparison visualization (target vs grown)

## Experiments & Results

### Experiment 1: Mordvintsev Architecture
**Setup**: Sobel + 1x1 conv, 100k iterations, tested at 80 growth steps

| Shape | Pixel Accuracy |
|-------|---------------|
| Circle | 1.0000 |
| Square | 1.0000 |
| Plus | 1.0000 |
| Triangle | 1.0000 |
| Line | 1.0000 |

**Conclusion**: Perfect accuracy. One rule grows 5 shapes from different seeds.

### Experiment 2: Stacked Conv Architecture
**Setup**: Conv3x3 + Conv1x1 + Conv1x1, 100k iterations, tested at 80 growth steps

| Shape | Pixel Accuracy |
|-------|---------------|
| Circle | 1.0000 |
| Square | 1.0000 |
| Plus | 1.0000 |
| Triangle | 1.0000 |
| Line | 1.0000 |

**Conclusion**: Perfect accuracy. General-purpose architecture matches specialized
Mordvintsev architecture on same task.

### Architecture Comparison

| | Mordvintsev | Stacked Conv |
|---|---|---|
| Final accuracy | 1.0000 | 1.0000 |
| Perception | Hardcoded Sobel | Learned |
| Convergence speed | Faster (0.001 loss by 5k) | Slower (0.001 loss by 35k) |
| Stability | Stable at lr=0.002 | Needed AdamW + lr=0.0005 |
| Generality | Morphogenesis only | Works on addition, physics, GoL, sorting |
| Training time | ~1.5 hours | ~2.5 hours |

## Why This Result Matters

### What Mordvintsev (2020) proved
One NCA can grow one shape from a seed and self-repair. Train a lizard NCA, get a
lizard. Want a butterfly? Train a separate NCA.

### What this experiment proves
One NCA can grow multiple shapes from different seeds. Same weights interpret different
seed vectors and produce different shapes. The growth program is general, not memorized.

### Stacked conv vs Mordvintsev
Both architectures achieve perfect accuracy, but the stacked conv result is arguably
more significant because the same architecture already works across every other task
in this repo. Mordvintsev's Sobel perception is optimized for spatial gradient
detection — perfect for morphogenesis but not transferable. The stacked conv learns
whatever perception the task requires, making it a universal NCA architecture for
both computation and morphogenesis.

## Training Notes

### Mordvintsev training
- Converged fast, stable throughout
- No hyperparameter tuning needed beyond standard settings
- Reached perfect accuracy by 100k iterations

### Stacked conv training
- Initial attempts at lr=0.002 with Adam exploded at ~45k iterations
- lr=0.0001 was too slow (loss still at 0.1 by 20k)
- AdamW with lr=0.0005 and weight_decay=0.01 converged and stayed stable
- Gradient clipping at 1.0 applied to all conv layers
- Loss reached 0.000000 on multiple shapes by 50k iterations
- Perfect accuracy by 100k iterations