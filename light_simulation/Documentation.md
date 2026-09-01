# Maxwell's Equations with Neural Cellular Automata

## Overview
This experiment explores whether an NCA can learn Maxwell's equations (electromagnetic wave propagation) from data alone and generalize through space and time. Instead of hand-coding FDTD (Finite-Difference Time-Domain) solvers, the NCA discovers the underlying wave physics purely from examples. By training on single-step Maxwell updates with minimal parameters (81 weights), the NCA learns the exact finite-difference stencils that generalize to larger grids and longer time horizons.

## Key Question

**Can an NCA learn the local rules of electromagnetic wave propagation from data alone, and does it discover actual Maxwell's equations or problem-specific shortcuts?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Ez (electric field, z-component)
  - Channel 1: Hx (magnetic field, x-component)
  - Channel 2: Hy (magnetic field, y-component)
- **Convolution**: 2D Conv (3→3 channels, 3×3 kernel, padding=1, bias=False)
- **Parameters**: 81 (3×3×3×3)
- **Activation**: None (linear - Maxwell's equations are linear)
- **Steps**: 1 forward pass per physics timestep
- **Loss**: MSE on all three field components (Ez excludes PEC solid cells)
- **Target Generation**: Ground truth computed using Maxwell's FDTD update:
  - `Hx_new = Hx - c*(dEz/dy)`
  - `Hy_new = Hy + c*(dEz/dx)`
  - `Ez_new = Ez + c*(dHy/dx - dHx/dy)`
  - Courant number c = 0.5 (CFL stability condition)
  - Mur ABC (absorbing boundary conditions)
  - PEC boundary: Ez = 0 inside solid

## Files

### Training Scripts

**`train_maxwell.py`**
- Minimalistic architecture with 3 channels, no bias, no activation
- Trains on random grid sizes (16-32 × 16-32)
- Random PEC solid obstacles (2×2 to 8×8 rectangles)
- Random light source position (single pulse: Ez=1 at t=0)
- Random physics timestep input (t=1-5)
- 1 forward pass predicts t→t+1 transition
- Training time: ~10 minutes for 50k iterations on GPU

### Testing Scripts

**`test_maxwell_weights.py`**
- Tests time generalization: trained on t=1-5, tested on t=200
- Tests space generalization: trained on 16-32, tested on 128×128
- Compares NCA output against ground truth Maxwell solver

## Experiments & Results

### Experiment 1: Generalization Across Time
**Setup**: Train on physics timesteps t=1-5, test on t=200 (40× longer)

**Results**:
- Final error at t=200: 0.000051
- Max error over 200 steps: 0.000674

**Conclusion**: ✓ The NCA **learned Maxwell's time evolution**, but hasn't truely recovered the exact stencil, and the error grows slowly over time. The NCA generalizes well to longer time horizons, but small numerical errors accumulate.
### Experiment 2: Generalization Across Space
**Setup**: Train on 16-32 grid sizes, test on 128×128 (16× larger)

**Results**:
- Final error at 128×128: 0.000032
- Max error over 50 steps: 0.000042

**Conclusion**: ✓ The NCA **failed to learn local wave physics** as the NCA doesn't know law of energy preservation, and 

## Key Findings

1. **NCA learns actual physics laws**: The learned weights are black-box approximations.

2. **Minimal parameters**: 81 weights (324 bytes) encode full 2D Maxwell solver extimator.

3. **Training efficiency**: 10 minutes to learn electromagnetic wave physics from scratch.