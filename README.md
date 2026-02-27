# NCA Research

Exploring Neural Cellular Automata for computation.

## What's here

Each folder = one experiment/direction.

| Folder | What | Status |
|--------|------|--------|
| `Binary_addition/` | NCA learns to add binary numbers | Works |
| `generalization_limits/` | NCA learns to generalize to add numbers 100 - 999 when trained on data from 0 - 99 | Works |
| `hebbian_learning_nca/` | NCA trained on Hebbian Learning | Failed | 
| `time_grid_scaling/` | Testing training time when grid size increases | Same training time |
| `computational_NCA/logic_gates` | NCA learns logic gates then generalizes to 16x training data | Trained and 100% accuracy on unseen data |
| `computational_NCA/heat_diffusion` | Train an NCa to figure out heat diffusion rules when only given input and target | Works and generalizes |
| `sorting` | NCA learns to sort arrays via value routing and ranking | 60% routing, 95% ranking, 0% generalization |

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

### Time Grid Scaling
``` bash
cd time_grid_scaling
python grid_scaling.py
```

### Logic Gates
``` bash
cd computational_NCA/logic_gates
python train_logic_gates.py
python test_logic_gates.py
```

### Sorting
``` bash
cd sorting
python 4_elements_sort.py
python test_4_elements_sort.py
```
``` bash
cd sorting
python gated_residual_sort.py
python gated_residual_test.py
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
 
### time_grid_scaling
 - `raw_notes.md` - Notes and results of experiment
 - `Documentation.md` - Documentation of experiment and what it reveals about NCA's parallel stucture
 - `grid_scaling.py` - code that times time to train NCA with different grid sizes

### logic_gates
 - `Documentation.md` - Experiment results and significance of experiment
 - `train_logic_gates.py` - Train NCA, uses variable grid size that adds noise to force generalization
 - `test_logic_gates.py` - Test the trained weights on 16x training data

### heat_diffusion
 - `Documentation.md` - Experiment results and significance is physics
 - `train_heat_diffusion.py` - Code to train heat diffusion. Gives NCA random input and target is calculated for diffusion after 5 steps
 - `test_heat_diffusion.py` - Test accuracy on seen and unseen data

### sorting
 - `Documentation.md` - 8 experiments, failure taxonomy, and core findings on NCA routing limits
 - `raw_notes.md` - My thinking on why each experiment fails
 - `2_element_swap.py` - Basic 2-element sort with preservation loss
 - `2_element_sort_extended.py` - 2-element with expanded capacity (32 channels, 500k iterations)
 - `4_elements_sort.py` - 5-phase NCA with Hungarian matching loss
 - `4_elements_rank.py` - Cross-entropy ranking approach
 - `test_4_elements_sort.py` - Evaluation script with ±5 tolerance metrics
 - `gated_residual_sort.py` - Train NCA to sort using gated residual activation
 - `gated_residual_test.py` - Test gated weights on seen and unseen data

## Key findings

- Binary addition generalizes (train 0-5, test 0-7 → 84%)
- ASCII fails (locality mismatch)
- Can't skip steps (nested tanh doesn't simplify)
- Distillation works but doubles training cost
- 3-digit generalization works (train 0-99, test 100-999 → 99%)
- training time isn't influenced by grid size
- NCAs are universal function generalizers for local systems
- NCA can learn local physics rules if given input and expected output
- the local weights also generalize beyond training data
- Sorting via ranking works (95% on width 4) but doesn't move values
- Sorting via value routing works (60% on width 4) but precision limited by tanh
- Neither sorting approach generalizes to unseen widths
- Multi-phase NCA enables sequential operations single NCA cannot do
- Activation function matters depending on the task 