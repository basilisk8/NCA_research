# Game of Life with Neural Cellular Automata

## Overview
This experiment explores whether an NCA can learn the rules of Conway's Game of Life
when given only input states and expected outputs. Unlike previous experiments (heat
diffusion, Maxwell's equations) which involved continuous linear PDEs, Game of Life
is discrete and nonlinear. A single wrong cell cascades into completely divergent
states. By training on 16x16 grids and testing on grids up to 16x larger and 100
time steps deep, we can see if the learned local rules are exact — not approximate.

## Key Question

**Can an NCA learn discrete nonlinear rules from data alone, and does it generalize
perfectly across time and space despite cascading error sensitivity?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Single channel: binary cell state (0.0 = dead, 1.0 = alive)
- **Conv1**: 2D Conv (1→16 channels, 3×3 kernel, padding=1), ReLU activation
- **Conv2**: 2D Conv (16→1 channels, 1×1 kernel), no activation (outputs logits)
- **Prediction**: sigmoid(logits) > 0.5 thresholded to binary
- **Steps**: 1 forward pass per Game of Life frame
- **Loss**: BCEWithLogitsLoss (binary cross entropy on logits)
- **No residual connection**: output replaces input entirely (cells can die)
- **Target Generation**: Ground truth computed using Game of Life rules:
  - Live cell + 2 or 3 neighbors → survives
  - Dead cell + exactly 3 neighbors → born
  - All other cells → die

### Why 16 Hidden Channels
1 channel fails because Game of Life requires non-monotonic response to neighbor
count. Dead cell + 2 neighbors = stay dead, + 3 = born, + 4 = stay dead. One
linear function + one activation cannot separate these cases. 16 hidden channels
each learn a different threshold, second layer combines them.

## Files

### Training Scripts

**`train_GoL.py`**
- Main training script for Game of Life
- Generates random binary 16x16 grids as training data
- Target is one Game of Life step applied to input
- Single NCA forward pass predicts next state
- 50,000 training iterations
- Outputs: GoL_weights.pth

### Testing Scripts

**`test_GoL.py`**
- Tests spatial generalization from 16x16 to 256x256
- Tests temporal generalization up to 100 autoregressive steps
- Tests combined time + space on large grids
- Compares NCA predictions against true Game of Life rules
- **Core experiment for proving NCA learns exact discrete rules**

## Experiments & Results

### Experiment 1: Spatial Generalization
**Setup**: Train on 16x16, test on 16 to 256 (16x training size)

**Results**:

| Grid Size | Perfect Grids |
|-----------|---------------|
| 16x16 (seen) | 20/20 |
| 32x32 (unseen) | 20/20 |
| 64x64 (unseen) | 20/20 |
| 128x128 (unseen) | 20/20 |
| 256x256 (unseen) | 20/20 |

**Conclusion**: ✓ Perfect spatial generalization at all scales.

### Experiment 2: Time Generalization (Autoregressive)
**Setup**: Train on 1-step prediction, test autoregressive rollout on 32x32

**Results**:

| Time Steps | Avg Final Accuracy |
|------------|-------------------|
| 10 steps | 100.0% |
| 20 steps | 100.0% |
| 50 steps | 100.0% |
| 100 steps | 100.0% |

**Conclusion**: ✓ Perfect temporal generalization. Zero error accumulation.

### Experiment 3: Time + Space Combined
**Setup**: Large grids run for 50 autoregressive steps

**Results**:

| Grid Size | Accuracy After 50 Steps |
|-----------|------------------------|
| 64x64 | 100.0% |
| 128x128 | 100.0% |

**Conclusion**: ✓ Perfect generalization across both time and space simultaneously.

### Why This Result Is Stronger Than Heat Diffusion
**Heat diffusion**: Continuous values, linear rule, small errors stay small.
Accuracy was 99.97% — excellent but approximate.

**Game of Life**: Binary values, nonlinear rule, one wrong cell flips neighbor
counts which flips birth/death decisions which cascades across the grid.
After 100 steps, a single error in step 1 would corrupt the entire grid.
Accuracy was 100.0% — the NCA learned the rule exactly, not approximately.
