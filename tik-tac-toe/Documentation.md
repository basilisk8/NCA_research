# NCA Learning Tic Tac Toe: Experiments & Results

## Overview
This experiment explores weather an NCA can learn the mechanism of tic tac toe, aby being trained by minimax, and then be tested against the minimax algorithm. The NCA is trained to play tic tac toe by playing against a minimax opponent, and then evaluated on its win/draw/loss rate against the same minimax algorithm.
## Key Question

**Can an NCA learn to play tic tac toe at a strong level by being trained against a minimax opponent?**

## Architecture

### Base Architecture (All Experiments)
- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Input layer (Raw board state, never modified)
  - Channel 1: Output layer (NCA output move probabilities)
  - Channels 2+: Hidden state (comparison/routing computation)
- **Convolution**: 
2D Conv (3×3 kernel, padding=1) input channel : 32; output channel 256
2D Conv (1×1 kernel, padding=0) input channel : 256; output channel 256
1D Conv (1×1 kernel, padding=0) input channel : 256; output channel 256
1D Conv (1×1 kernel, padding=0) input channel : 256; output channel 31
- **Activation**: tanh (bounds updates to [-1, 1])

## Files

### Training Scripts

**`train_nca.py`**
- Training Script for NCA with minimax giving the training data
- Grid: `(1, 32, 3, 3)`
- Training: 500k iterations, 
- Outputs: `tictactoe_nca.pth`

### Testing Scripts

**`nca_vs_minimax.py`**
- NCA plays 10 games against minimax. 9 games minimax is first and 1 game NCA is first.
- Metrics: win/draw/loss rate for NCA
- Outputs: Print results to console, including win/draw/loss counts and percentages, and overall assessment of NCA's performance.
## Experiments & Results


### Experiment 1: Train NCA to play tic tac toe against minimax
- **File**: `train_nca.py`
**Setup**: 32 channels, 30 steps, 500k iterations
**Result**: Success : 100% draw rate against minimax, no wins or losses. NCA learns to play perfectly defensively, but never wins.
**Conclusion**: NCA learns to play optimally, not letting minimax (a perfect player) win. 

**Conclusion**: 
- The tic taac toe grid being the exact size of the convolution kernel (3x3) allows the NCA to learn local patterns that correspond to winning/drawing moves.
- The NCA learns to play perfectly defensively, achieving a 100% draw rate against
- The situation alligns so NCA can see hte global board state. 
- It is not completely perfect, as when I was playing against NCA, there is 1 specific set of moves that creates a double fork NCA doesn't block allowing me to win. So it is not perfect, but more training might fix this.