# Time taken with grid size scaling

## Overview
This experiment tests if training time for a grid 11 wide and 30 wide takes about the same time. If yes then NCA's will have an edge over NN because they can generalize and scale without needing constant re-training

## Key Question

**Does it take about the same time to train an NCA that is 11 wide compared to an NCA 30 wide**

## Files

### Training Scripts

**`grid_scaling.py`**
- Main training script for trainng an NCA with width 11 and width 30
- Grid: 2D (1×16×2×grid_size)
- Training range: Configurable (0-99)

## Experiments & Results

### Experiment 1: Finding time difference
**Setup**: Train an NCA that ius 11 wide  and train an NCA 30 wide. time the training time.
**Result**: ~117 seconds to train NCA 11 wide and ~110 seconds for training an NCA 30 wide
**Conclusion**: NCA cell updates though conv is Parallel and making the grid wider in a reasonable size doesn't change training time

**Conclusion**: ✓ The NCA scales efficiently with bigger grids doe to conv natural parallel structure in NCA
