# NCA Research

Exploring Neural Cellular Automata for computation.

## What's here

Each folder = one experiment/direction.

| Folder | What | Status |
|--------|------|--------|
| `Binary_addition/` | NCA learns to add binary numbers | Works |
| `generalization_limits/` | NCA learns to generalize to add numbers 100 - 999 when trained on data from 0 - 99 | Works |
| `hebbian_learning_nca/` | NCA trained on Hebbian Learning | Failed | 

## Quick start
### Binary addition
```bash
cd Binary_addition
python train_2d_addition_nca.py
cd testing_weights
python generalize_test.py
```

### Generalization Limits
``` bash
cd generalization_limits
python 2_digit_training.py
cd testing_generalized_weights
python 3_digit_generalize_test.py
```

## Files in each folder
### Binary_addition
- `memorize_addition_nca.py` - code to test if nca can 'remember'
- `train_2d_addition_nca.py` - training code to train nca 1 digit addition in a 16 channel, 2D array
- `testing_weights/generalize_test.py` - Check how well nca generalized on problems seen in training, and never before seen problems
- `2_digit_generalization.py` - testing code to check accuracy of nca trained on 1 digit number addition on 2 digit number addition
- `Documentation.md` - domumentation of everything I tried, results and conclusion

### generalization_limits
- `2_digit_training.py` - code to train nca to learn addition from numbers 0 - 99
- `testing_generalized_weights/3_digit_generalize.py` - Check how accurate NCA learned on 2 digit addition is when tested on 3 digit numbers
- `raw_notes.md` - My raw notes before, during and after the experiment
- `Documentation.md` - cleaned up raw notes with details on experiments and results

### hebbian_learning_nca
 - `hebbian_learing_nca.py` - code that tried to train NCA on hebbian learning
 - `Documentation.md` - Documentation of experiment and my thoughts on why it failed
 - `raw_notes.md` - My raw notes describing my thinking during and after experiment
 
## Docs

- `journey.md` - full timeline of what I did
- `ideas_and_brain_dump.md` - raw braindump of ideas to explore

## Key findings

- Binary addition generalizes (train 0-5, test 0-7 → 84%)
- ASCII fails (locality mismatch)
- Can't skip steps (nested tanh doesn't simplify)
- Distillation works but doubles training cost
- 3-digit generalization works (train 0-99, test 100-999 → 99%)