# NCA Sorting Experiments

## Overview
This experiment explores whether Neural Cellular Automata can learn to sort arrays of integers using only local rules. Unlike addition where values transform in place, sorting requires **routing** — moving values from one position to another while preserving them. This led to a multi-day investigation through 8 different architectural approaches, each revealing distinct failure modes and ultimately proving what NCA can and cannot do.

## Key Question

**Can an NCA learn to sort an array by discovering local comparison and swap rules, similar to how bubble sort works?**

## Architecture

### Base Architecture (All Experiments)
- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 0: Input layer (values to sort, never modified)
  - Channel 1: Output layer (sorted result)
  - Channels 2+: Hidden state (comparison/routing computation)
- **Convolution**: 2D Conv (3×3 kernel, padding=1)
- **Activation**: tanh (bounds updates to [-1, 1])
- **Normalization**: `(value - 128) / 64` → input range [-2, 2]

### 5-Phase NCA Architecture
- **5 separate Conv networks** with independent weights
- **32 channels** for expanded hidden state
- **20 steps per phase** (100 total steps)
- **Hungarian matching loss** to prevent interpolation

### Gated residuals Architecture
 - **1 conv network** learning sorting
 = **Gated residuals** for activation function, to seperate value preservation from logic
 - **64 channels** for compute, 32 channels for value and 32 for gates
 - **60 steps** for info prop and value preservation
 - **Hungarian matching loss** to prevent interpolation

## Files

### Training Scripts

**`swap_and_preserve.py`**
- Single NCA with swap + preservation loss
- Grid: `(1, 16, 1, 2)`
- Training: 50k iterations, proves basic comparison works
- Outputs: `swap_weights.pth`

**`train_compare_and_swap.py`**
- Single NCA with expanded capacity (32 channels, 30 steps)
- Grid: `(1, 32, 1, 2)`
- Training: 500k iterations, tests if capacity solves precision
- Outputs: `sort_big_weights.pth`

**`4_elements_sort.py`**
- 5-phase NCA with Hungarian matching loss
- Grid: `(1, 32, 1, 4)`
- Training: 700k iterations on width-4 arrays
- Outputs: `4_elements_sort.pth`

**`4_elements_rank.py`**
- Single NCA with cross-entropy ranking approach
- Grid: `(1, 16, 1, 4)`
- Classification approach (4 rank classes)
- Outputs: `sort4_rank_weights.pth`

**gated_residual_sort.py**
 - Single NCA with gated residuals as activation function
 - Grid; `(1, 64, 7, 1)`
 - Training : 500k itterations
 - Direct value output
 - Output: `gated_residual_sort.pth`

### Testing Scripts

**`test_4_elements_sort.py`**
- Evaluates 5-phase sorting model
- Tests: width-4 (trained), close values, width-5/6 (unseen)
- Metrics: exact match and ±5 tolerance

**gated_residual_test.py**
- Evaluate the NCA weights
- Tests: Width 7 (trained), close values, width-7/10 (unseen)
- Metrics: exact match and ±5 tolerance

## Experiments & Results


### Experiment 1: 2-Element Swap
**File**: `swap_and_preserve.py`, `train_compare_and_swap.py`
**Setup**: 32 channels, 30 steps, 500k iterations, simplest case
**Result**: SUCCESS — Works with ±1-2 precision
```
INPUT           | OUTPUT (Raw)     | EXPECTED
[182, 241]      | [182.8, 241.1]   | [182, 241]  ✓
[112, 99]       | [100.3, 110.9]   | [99, 112]   ✓ (Off by 2)
[236, 113]      | [113.1, 235.4]   | [113, 236]  ✓ (Off by 1)
```
**Conclusion**: NCA CAN compare and swap 2 values. Precision limited by tanh's continuous domain.

---

### Experiment 2: Direct Value Routing
**Setup**: 16 channels, 20 steps, MSE loss on sorted output
**Result**: FAILED — Interpolation trap
```
Input:    [Random values]
Output:   [50, 100, 150, 200]  (Fixed ascending ramp regardless of input)
```
**Conclusion**: MSE rewards outputting statistical average of sorted arrays, not actual sorting.

---

### Experiment 3: MSE + Preservation Loss
**Setup**: Added `preserve_loss = MSE(sort(output), sort(input))`
**Result**: FAILED — Identity trap
```
Input:    [195, 40, 10, 80]
Output:   [195, 40, 10, 80]  (No sorting attempted)
```
**Conclusion**: Identity scores 0 on preservation. Moving values corrupts them temporarily, so gradient descent refuses to move.

---

### Experiment 4: Binary Encoding
**Setup**: Encode each number as 8 binary channels
**Result**: FAILED — Bit desynchronization
```
Bit 0 swaps correctly, Bit 7 stays in place → Number corrupts
```
**Conclusion**: NCA treats channels independently. No concept that channels 0-7 are "one number."

---

### Experiment 5: Inversion Loss
**Setup**: `loss = relu(left_neighbor - right_neighbor)` — only penalize local inversions
**Result**: FAILED — Oscillation
```
Fixing pair [0,1] breaks pair [1,2], fixing [1,2] breaks [0,1]... forever
```
**Conclusion**: Local fixes cascade into global instability without mechanism to lock solved pairs.

---

### Experiment 6: Cross-Entropy Ranking
**File**: `4_elements_rank.py`
**Setup**: Each cell classifies its rank (0 to N-1), N output channels
**Result**: SUCCESS — Works for fixed width
```
Input:      [200, 50, 150, 100]
Pred ranks: [3, 0, 2, 1]  ✓
True ranks: [3, 0, 2, 1]
```
**Limitation**: Cannot generalize to unseen widths. Width 4 needs 4 channels, width 8 needs 8.
**Conclusion**: Proves NCA CAN learn local comparison rule. Limitation is encoding, not NCA.

---

### Experiment 7: Single-Channel MSE Ranking
**Setup**: Output rank as continuous value in one channel, MSE loss
**Result**: FAILED — Position-based shortcut
```
Input:      [150, 152, 148]
Pred ranks: [0.4, 1.5, 1.1]  (Position bias dominates)
True ranks: [1, 2, 0]
```
**Conclusion**: Network learns `rank ≈ position + f(value)` instead of actual comparison.

---

### Experiment 8: 5-Phase NCA with Hungarian Loss
**File**: `4_elements_sort.py`, `test_4_elements_sort.py`
**Setup**: 
- 5 independent Conv networks run sequentially
- Hungarian matching loss forces one-to-one value assignment
- 32 channels, 20 steps per phase (100 total)
- 700k training iterations

**Result**: PARTIAL SUCCESS — 60% accuracy within ±5 tolerance
```
Iter 415000:
  Expected: [37, 65, 176, 235]
  Got:      [38, 64, 176, 235]
  Sort Loss: 0.0001 | Match Loss: 0.0304
```


**Test Results**:
| Test | Accuracy |
|------|----------|
| Width 4 (exact match) | 0/100 |
| Width 4 (±5 and ordered) | 60/100 |
| Close values (spacing 1-5) | 6/9 (67%) |
| Width 5 (never trained) | 0/20 |
| Width 6 (never trained) | 0/20 |


### Experiment 9: Gated residuals with hungarian loss
**File**: `gated_residual_test.py`, `gated_residual_test.py`
**Setup**
 - 64 channels, split into value and logic channels, with gate channel next to it's correcponding value channel
 - Hungarian matching loss
 - 60 steps
 - 500k itterations

**Results**: 99% accuracy on seen data, 100% on unseen data, 85% on unseen data of different width
| Test | Accuracy |
|------|----------|
| Width 7 (exact match) | 7/100 |
| Width 7 (±5 and ordered) | 99/100 |
| Width 5 (never trained) | 20/20 |
| Width 10 (never trained) | 17/20 |

**Conclusion**: 
- ✓ NCA learns value routing — order is correct, values are approximately correct
- ✗ Precision limited by 100 steps of tanh accumulation
- ✗ Zero generalization to unseen widths
- Ranking approach achieves ~95% accuracy but doesn't move values
- Value routing achieves 60% accuracy but actually moves values