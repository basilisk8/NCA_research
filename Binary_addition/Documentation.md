# Binary Addition with Neural Cellular Automata

## Overview

This experiment explores whether Neural Cellular Automata (NCAs) can **learn** binary addition as an algorithm, rather than simply memorizing input-output pairs. By training on small numbers and testing on larger unseen numbers, we investigate the generalization capabilities of NCAs.

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

**`memorization_addition_nca.py`**
- Proof-of-concept: Memorizes single example (3 + 5 = 8)
- Grid: 1D, width=8
- Demonstrates basic NCA mechanics
- **Not for generalization testing**

**`train_2d_addition_nca.py`**
- Main training script for generalization experiments
- Grid: 2D (1×16×2×4)
- Training range: Configurable (0-7, 0-6, or 0-5)
- Outputs: `my_weights.pth`, `example.txt`
- **Key parameter**: `random.randint(0, X)` - adjust for generalization tests

### Testing Scripts

**`test_generalization.py`**
- Tests if NCA generalizes beyond training data
- Requires: Weights trained on 0-5
- Tests: 0-5 (seen) vs 6-7 (unseen)
- **Core experiment for proving learning vs memorization**

**`test_2digit_addition.py`**
- Tests if learned patterns transfer to wider grids
- Uses weights from 4-bit training on 8-bit problems
- Tests: Random 2-digit additions (10-99)
- **Explores spatial generalization**

## Experiments & Results

### Experiment 1: Memorization Baseline
**Setup**: Train on single example (3 + 5)
**Result**: Perfect memorization (loss → 0)
**Conclusion**: NCA can fit data, but this proves nothing about learning

### Experiment 2: Training Data Performance
**Setup**: Train on 0-7, test on 0-7
**Result**: 64/64 correct (100%)
**Conclusion**: NCA can learn all 64 combinations

### Experiment 3: Generalization (Core Result)
**Setup**: Train on 0-5, test on 6-7
**Results**:
- Training data (0-5): **36/36 (100%)**
- Unseen data (6-7): **24/28 (86%)**

**Failures**:
3+7 ✗
5+7 ✗
7+0 ✗
7+5 ✗

**Conclusion**: ✓ The NCA **learned addition**, not just memorized. 86% accuracy on completely unseen data demonstrates genuine generalization.

### Experiment 4: Grid Size Transfer
**Setup**: Train on 4-bit (0-7), test on 8-bit (10-99)
**Result**: **64/100 (64%)**
**Conclusion**: Partial transfer of learned patterns. Model trained on width=4 shows some success on width=8, suggesting local rules were learned.

## Key Findings

1. **NCAs can learn algorithms** - Not just lookup tables
2. **Generalization works** - 86% on unseen 6-7 after training on 0-5
3. **Partial spatial transfer** - 64% on 2-digit when trained on 1-digit
4. **Learning rate matters** - lr=0.01 causes instability; lr=0.0001 recommended
5. **Hidden channels crucial** - 16 channels needed for carry/state management

## Training Tips

### Stable Training
```python
lr = 0.0001  # Not 0.01 (causes wild logit swings)
steps = 20   # Sufficient for 4-bit carry propagation
```

### Generalization Test
```python
# In train_2d_addition_nca.py, change:
a = random.randint(0, 5)  # Instead of (0, 7)
b = random.randint(0, 5)  # Test on 6-7 later
```

### GPU Training (Google Colab)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Add .to(device) to conv and tensors
```

## Limitations

1. **Grid width mismatch** - Training on width=4, testing on width=8 reduces accuracy
2. **Rare failures** - Even on trained data, occasional errors (4/28 on unseen)
3. **Compute cost** - 20 forward passes per example vs 1 for direct NN
4. **Training time** - 100k iterations needed for convergence

## Next Steps

### Hypothesis: Width=11 Training
**Goal**: Train on width=11 grid with 1-digit numbers, test on 3-digit numbers

**Logic**: If NCA learns local addition rules (bit + bit + carry), the same weights should work across any grid width (translationally invariant algorithm).

**Prediction**: If successful (>80% on 3-digit), this proves NCA learned the **algorithm** of binary addition, not patterns specific to small numbers.

**Significance**: This would demonstrate true algorithmic learning in NCAs - a relatively unexplored capability.

## References

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Original NCA paper (Mordvintsev et al., 2020)
- Binary addition requires local carry propagation (sequential dependency)
- NCAs excel at local, parallel, translationally-invariant tasks

## Conclusion

Neural Cellular Automata can learn binary addition as a generalizable algorithm, not just memorize training examples. With 86% accuracy on completely unseen numbers and 64% transfer to different grid sizes, this work demonstrates that NCAs can capture computational patterns beyond simple pattern matching.

**The key insight**: NCAs may be well-suited for tasks where the algorithm is local and translationally invariant - a class of problems that includes many fundamental computations.
