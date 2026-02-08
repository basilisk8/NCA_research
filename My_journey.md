## Journey

### Day 1

Started knowing nothing about NCAs. Built binary addition from scratch.

**What I built:**
- 1D NCA: 8 wide, 2 channels (input + output)
- Conv1d, 3 kernel, tanh activation
- Trained on 3+5=8, then generalized to random pairs

**Problems hit:**
- MSE loss → model settled at 0.5 (hedging)
- Fixed with BCE loss → forces commitment to 0 or 1
- In-place operations broke gradients → used .clone()

**Result:** 
- Trained on 0-5, tested on 0-7
- ~84% accuracy on unseen numbers
- 14 parameters total

**Step-skipping attempts:**
- Tried polynomial approximation of tanh → error compounds
- Tried derivative analysis(with AI help in coding) → system chaotic before convergence
- Tried finding stable layer → doesn't exist for this architecture
- Conclusion: can't skip steps mathematically

---

### Day 2

Tried to push NCAs further. Mostly failed.

**ASCII addition:**
- 32 channels, 24 wide, 2 rows
- Encode digits as ASCII binary
- Failed: ~39% accuracy, hallucinating symbols like ":" and ";"
- Why: ASCII encoding is arbitrary. 8 bits = 1 char requires non-local understanding. NCAs only see 3 neighbors.

**Meta-NCA:**
- Train one model to predict ANY NCA's output given weights
- Failed: loss stuck at 80+, never learned
- Why: random weights produce wildly different behaviors, too diverse

**Distillation:**
- Train NN to mimic ONE trained NCA
- Worked: 6x faster inference, same accuracy
- But: training cost went UP (train NCA + train NN)

---

### Day 3

Expanded day 1 binary addition code to a bigger grid for more clear generalization patterns

**3 digit generalization based on 2 digit trained model**
 - Expanded grid to 11 wide
 - changed loss function to look at entire row, because looking at parts of it means the ignored part show random values, since they aren't part of loss function
 - Success : The model behaved as expected, with 100% accuracy on seen data and 99% (9911/10000) accuracy on unseen data. The model behaves as expected and works because carry rules are always the same regardless of size of the number

---

### Day 4

Tried to train NCA using hebbian learing

**Hebbian learning**
 - Per cell weights for the NCA so the weights are stored in the grid itself
 - Expected result was cells specializing liek brain does
 - Actual result was it didn't learn 
 
### Day 5

Measure how much time it takes to train an NCA for a grid 11 wide and 30 wide

**Grid time scaling**
 - Takes about the same time to train an NCA that is 11 wide and 30 wide
 - 30 wide grid can store a lot more information than the 11 wide grid but take the same training time

### What I learned

NCAs work when:
- Local information is meaningful
- Same rule applies everywhere
- Spatial structure matches problem structure
- Need generalization beyond training data, where the local rules are always the same

NCAs fail when:
- Need global context (ASCII)
- Encoding is arbitrary
- Problem requires grouping (8 bits = 1 char)
- Try to train using Hebbian learning

---

### Open questions
- Per-cell weights instead of global? (Tried and failed)
- New learning algorithm without backprop? (Backprop is the most efficient and accurate learning algorithm fro NCA's)
- NCAs as equation solvers?
- Running NCA on esp32?