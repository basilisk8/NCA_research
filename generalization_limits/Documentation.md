# Testing Generalization with big numbers

## Overview
This experiment explores whether locally learnt addition rules for numbers from 0 - 99 provides accurate responses when tested with numbers from 100 - 999. This experiment tried to do this by eliminating most factors that affect output expect training data. By training on small numbers and applying same weights to completely unseen data we can see if NCA's local rule generalize beyond training data

## Key Question

**Can an NCA learn the local rules of binary addition (bit-wise sum + carry propagation) and generalize to numbers it has never seen?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Input layer (two numbers in binary, never modified)
  - Channel 1: Output layer (sum appears here after evolution)
  - Channels 2-15: Hidden state (for carry propagation and computation)
- **Convolution**: 2D Conv (16→15 channels, 3×3 kernel, padding=1)
- **Activation**: tanh (bounds updates to [-1, 1])
- **Steps**: 20 forward passes per example
- **Loss**: BCEWithLogitsLoss (binary cross-entropy with built-in sigmoid)


## Files

### Training Scripts

**`2_digit_training.py`**
- Main training script for generalization experiments
- Grid: 2D (1×16×2×11)
- Training range: Configurable (0-99)
- Outputs: `my_weights.pth`

### Testing Scripts

**`3_digit_generalize_test.py`**
- Tests if NCA generalizes beyond training data
- Requires: Weights trained on 0-99
- Tests: 0-99 (seen) vs 100 - 999 (unseen)
- **Core experiment for proving learning vs memorization**

## Experiments & Results

### Experiment 1: Training Data Performance
**Setup**: Train on 0-99, test on 0-99
**Result**: 500/500 correct
**Conclusion**: NCA learnt 2 digit addition with 100% accuracy

### Experiment 2: Generalization (Core Result)
**Setup**: Train on 0-99, test on 100- 999
**Results**:
- Training data (0-99): **500/500 (100%)**
- Unseen data (100-999): **9911/10000 (99%)**

**Conclusion**: ✓ The NCA **learned addition**, not just memorized. 99% accuracy on completely unseen data demonstrates genuine generalization.