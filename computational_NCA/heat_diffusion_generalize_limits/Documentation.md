# Heat Diffusion with Neural Cellular Automata

## Overview
This experiment explores whether an NCA can learn the local rules of heat diffusion
and generalize through space and time. Instead of hand-coding the heat
equation, the NCA discovers the underlying physics purely from data. By training on
1 step heat equation with minimal channels and parameters, NCA can learn the equation
that generalizes to bigger grids and more steps.

## Key Question

**Can an NCA learn the local rules of heat diffusion from data alone and generalize through space and time?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Temperature field (input state, modified during evolution)
- **Convolution**: 2D Conv (1→1 channels, 1×3 kernel, padding=1)
- **Activation**: None
- **Steps**: 1 forward pass every 1 physics frame
- **Loss**: MSELoss (mean squared error between predicted and target temperature)
- **Target Generation**: Ground truth computed using discrete heat equation:
  `T_new[i][j] = T[i][j] + α * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - 4*T[i][j])`

## Files

### Training Scripts

**`variable_steps.py`**
***Failed***
- Tried the same logic of the original heat diffusion model, except with variable steps
- Generates random initial temperature fields as training data
- Target is calculated by applying discrete heat equation for variable steps
- Training : grid sizes: 8-16, steps: 5, 20

**single_step_nca.py**
- Minimalistic architecture with 1 channel
- 1 forward pass, and compared to 1 physics frame
- Also test generalization to 16x width and 1000x steps with perfect accuracy

### Testing Scripts

**`test_generalization.py`**
***Failed***
- Tests if NCA generalizes beyond training grid sizes and step count
- Requires: Weights trained on 8-16 grids
- Tests: 16, 32, 64, 128 width grids (up to 8x training size)
- Tests: 5, 10, 50, 100 steps
- Compares NCA output against ground truth heat equation

## Experiments & Results

### Experiment 1: Adding variable step count
**Setup**: Train on grid sizes 8-16, step count from 5 - 20, test on grid size upto 128 wide and 100 steps
**Result**: Accuracy : 0% 
**Conclusion**: It learnt the pattern for the specific range, not generalized

### Experiment 2: Generalization (Core Result)
**Setup**: Train on grid sizes 8-16, test on 16, 32, 64, 128

**Results**:
0.00000000 MSE loss when expanded upto 128 wide and 1000 steps

**Conclusion**: ✓ The NCA **learned heat diffusion**, not just memorized patterns.
Accuracy actually improves on larger grids, demonstrating that the NCA discovered
genuine local diffusion rules that compose correctly at any scale.

### Why Accuracy Improves With Scale
**Observation**: MSE decreases as grid size increases
**Explanation**: Boundary effects have proportionally less influence on larger grids.
The interior cells follow pure local rules which the NCA learned correctly. More
interior cells relative to boundary cells means higher overall accuracy.