# NCA Research

Exploring Neural Cellular Automata for computation.

## What's here

Each folder = one experiment/direction.

| Folder | What                                                                                                                                                              | Status                                                                                            |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `Binary_addition/` | NCA learns to add binary numbers                                                                                                                                  | Works                                                                                             |
| `generalization_limits/` | NCA learns to generalize to add numbers 100 - 999 when trained on data from 0 - 99                                                                                | Works                                                                                             |
| `hebbian_learning_nca/` | NCA trained on Hebbian Learning                                                                                                                                   | Failed                                                                                            | 
| `time_grid_scaling/` | Testing training time when grid size increases                                                                                                                    | Same training time                                                                                |
| `computational_NCA/logic_gates` | NCA learns logic gates then generalizes to 16x training data                                                                                                      | Trained and 100% accuracy on unseen data                                                          |
| `computational_NCA/heat_diffusion` | Train an NCA to figure out heat diffusion rules when only given input and target                                                                                  | Works and generalizes                                                                             |
| `computational_NCA/heat_diffusion_generalize_limits` | Train an NCA on 1 forward pass and test generalization across time and space                                                                                      | Generalized                                                                                       | 
| `sorting` | NCA learns to sort arrays via value routing and ranking                                                                                                           | 60% routing, 95% ranking, 0% generalization , 100% gated routing 85% gated routing generalization |
| `light_simulation/` | NCA learns Maxwell's equations (electromagnetic wave propagation); estimates the ftdt stencil and breaks down in OOD | Works partially|
| `game_of_life/` | NCA learns Game of Life rules; tests perfect generalization across space and time                                                                                 | Works                                                                                             |
| `multi_shape_morphogenesis/5_shapes/` | Train a single NCA to grow 5 different binary shapes from learned seed embeddings; compares Mordvintsev architecture to stacked conv architecture                 | Works, stacked conv is perfect                                                                    |
| `multi_shape_morphogenesis/5_emoji/` | Train a single NCA to grow 5 different RGB emojis from learned seed embeddings; tests if same architecture can handle increased complexity of natural color images | Works, but not perfectly                                                                          |
| `multi_shape_morphogenesis/25_emoji_5_classes/` | Train a NCA to grow 25 emoji from 5 classes and 5 variants per class; tests if NCA can learn to grow many complex shapes with minimal parameters                  | Works                                                                                             |
| `tik-tac-toe/` | Train an NCA to play tic tac toe against a minimax opponent; tests if NCA can learn the game mechanics and achieve a strong win/draw/loss rate against minimax    | Works, 100% draw rate against minimax                                                             |
| `language/` | Train an NCA to learn the structure of english words based on local interactions                                                                                  | Works, 76% accuracy on next letter prediction                                                     |

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

### Heat diffusion
``` bash 
cd computational_NCA/heat_diffusion
python train_heat_diffusion.py
python test_heat_diffusion.py
```

### Single Step heat diffusion
``` bash
cd computational_NCA/heat_diffusion_generalize_limits
python single_step_NCA.py
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

### Light simulation
```bash
cd light_simulation
python train_maxwell.py
python test_maxwell_weights.py
```

### Game of Life
```bash
cd game_of_life
python train_GoL.py
python test_GoL.py
```

### Multi-shape morphogenesis
#### 5 shapes
```bash
cd multi_shape_morphogenesis/5_shapes
python multi_shape_nca.py
python test_multi_shape.py
```
```bash
cd multi_shape_morphogenesis/5_shapes
python stacked_conv_nca.py
python test_stacked_conv.py
```

#### 5 emojis
```bash
cd multi_shape_morphogenesis/5_emoji
python train_stacked_conv.py
python test_stacked_conv.py
```

#### 25 emojis, 5 classes
```bash
cd multi_shape_morphogenesis/25_emoji_5_classes
python train_stacked_conv.py
python test_stacked_conv.py
```

#### Tik Tac Toe
```bash
cd tik-tac-toe
python train_nca.py
python nca_vs_minimax.py
```

#### Language
```bash
cd language
python 7_letter_word_train.py
python test_generation.py
```
## Files in each folder
### Binary_addition
- `memorization_addition_nca.py` - code to test if nca can 'remember'
- `train_2d_addition_nca.py` - training code to train nca 1 digit addition in a 16 channel, 2D array
- `testing_weights/generalize_test.py` - Check how well nca generalized on problems seen in training, and never before seen problems
- `2_digit_generalization.py` - testing code to check accuracy of nca trained on 1 digit number addition on 2 digit number addition
- `Documentation.md` - documentation of everything I tried, results and conclusion

### generalization_limits
- `2_digit_training.py` - code to train nca to learn addition from numbers 0 - 99
- `testing_generalized_weights/3_digit_generalize_test.py` - Check how accurate NCA learned on 2 digit addition is when tested on 3 digit numbers
- `raw_notes.md` - My raw notes before, during and after the experiment
- `Documentation.md` - cleaned up raw notes with details on experiments and results

### hebbian_learning_nca
 - `hebbian_learning_nca.py` - code that tried to train NCA on hebbian learning
 - `Documentation.md` - Documentation of experiment and my thoughts on why it failed
 - `raw_notes.md` - My raw notes describing my thinking during and after experiment
 
### time_grid_scaling
 - `raw_notes.md` - Notes and results of experiment
 - `Documentation.md` - Documentation of experiment and what it reveals about NCA's parallel structure
 - `grid_scaling.py` - code that times time to train NCA with different grid sizes

### logic_gates
 - `Documentation.md` - Experiment results and significance of experiment
 - `train_logic_gates.py` - Train NCA, uses variable grid size that adds noise to force generalization
 - `test_logic_gates.py` - Test the trained weights on 16x training data

### heat_diffusion
 - `Documentation.md` - Experiment results and significance is physics
 - `train_heat_diffusion.py` - Code to train heat diffusion. Gives NCA random input and target is calculated for diffusion after 5 steps
 - `test_heat_diffusion.py` - Test accuracy on seen and unseen data

### heat_diffusion_generalize_limits
 - `Documentation.md` - Experiment and results with future implications in physics simulation
 - `variable_steps.py` - Failed architecture to generalize across time
 - `single_step_NCA.py` - Train NCA on 1 forward pass and test generalization across time and space

### sorting
 - `Documentation.md` - 8 experiments, failure taxonomy, and core findings on NCA routing limits
 - `raw_notes.md` - My thinking on why each experiment fails
 - `swap_and_preserve.py` - Basic 2-element sort with preservation loss
 - `train_compare_and_swap.py` - 2-element with expanded capacity (32 channels, 500k iterations)
 - `4_elements_sort.py` - 5-phase NCA with Hungarian matching loss
 - `4_elements_rank.py` - Cross-entropy ranking approach
 - `test_4_elements_sort.py` - Evaluation script with ±5 tolerance metrics
 - `gated_residual_sort.py` - Train NCA to sort using gated residual activation
 - `gated_residual_test.py` - Test gated weights on seen and unseen data

### light_simulation
- `Documentation.md` - Experiment overview, architecture, training details, results, and key findings (Maxwell/FDTD rediscovery)
- `train_maxwell.py` - Training script for the Maxwell NCA (3 channels: Ez, Hx, Hy)
- `test_maxwell_weights.py` - Tests time and space generalization; compares NCA to FDTD ground truth
- `physics_light_sim.pth` - Example trained weights / checkpoint

### game_of_life
- `Documentation.md` - Experiment overview, architecture, training details, results, and key findings on perfect generalization of discrete nonlinear rules
- `train_GoL.py` - Training script for Game of Life NCA (16 hidden channels, binary state)
- `test_GoL.py` - Tests spatial generalization (16x to 256x) and temporal generalization (up to 100 steps autoregressive)
- `raw_notes.md` - My raw notes during and after the experiment, including training logs and detailed generalization results

### multi_shape_morphogenesis/5_shapes
- `Documentation.md` - Experiment overview, architecture details for both Mordvintsev and stacked conv approaches, training details, results, and key findings on multi-shape morphogenesis
- `multi_shape_nca.py` - Training script for Mordvintsev architecture (Sobel + 1x1 conv)
- `test_multi_shape.py` - Tests the trained Mordvintsev weights on all 5 shapes and reports pixel accuracy; generates comparison visualization
- `stacked_conv_nca.py` - Training script for stacked conv architecture (Conv3x3 + Conv1x1 + Conv1x1)
- `test_stacked_conv.py` - Tests the trained stacked conv weights on all 5 shapes and reports pixel accuracy; generates comparison visualization

### multi_shape_morphogenesis/5_emoji
- `Documentation.md` - Experiment overview, architecture details, training details, results, and key findings on multi-shape morphogenesis extended to RGB emojis, including the challenges of continuous color values and fine visual details
- `train_stacked_conv.py` - Training script for stacked conv architecture adapted for RGB emoji generation
- `test_stacked_conv.py` - Tests the trained stacked conv weights on all 5 emojis, reports MSE loss and pixel accuracy, and generates comparison visualization
- `emoji_targets.pt` - Target RGB emoji images used for training

### multi_shape_morphogenesis/25_emoji_5_classes
- `Documentation.md` - Experiment overview, architecture details, training details, results, and key findings on scaling multi-shape morphogenesis to 25 RGB emojis across 5 classes, demonstrating the NCA's capacity for learning many complex shapes with minimal parameters
- `train_25_emoji.py` - Training script for stacked conv architecture adapted for 25 RGB emoji generation
- `test_25_emoji.py` - Tests the trained stacked conv weights on all 25 emojis, reports MSE loss and pixel accuracy, and generates comparison visualization
- `emoji_targets.pt` - Target RGB emoji images used for training

### tik-tac-toe
- `Documentation.md` - Experiment overview, architecture details, training details, results, and key findings on training an NCA to play tic tac toe against a minimax opponent, achieving a 100% draw rate and learning optimal defensive play
- `train_nca.py` - Training script for the tic tac toe NCA, where the NCA learns by playing against a minimax opponent
- `nca_vs_minimax.py` - Testing script where the trained NCA plays 10 games against minimax (9 games with minimax first, 1 game with NCA first) and reports win/draw/loss rates for the NCA

### language
- `Documentation.md` - Experiment overview, architecture details, training details, results, and key findings on training an NCA to learn the structure of English words based on local interactions, achieving 76% accuracy on next letter prediction and demonstrating that global attention is not a requirement for language modeling
- `7_letter_word_train.py` - Training script for the language NCA, where the NCA learns to predict the next letter in English words based on local interactions, trained on a dataset of 2-7 letter words
- `test_generation.py` - Testing script that evaluates the word generative capability of the trained NCA by systematically loading every word and testing every possible index combination, tracking the total inputs and the number of generated words that exist in the dataset or English dictionary
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
- NCA does better with minimal parameters in physics simulations
- NCA learns discrete nonlinear rules (Game of Life) with perfect generalization across space and time, despite cascading error sensitivity
- NCA can learn structure of English words with purely local interactions, achieving 76% accuracy on next letter prediction, demonstrating that global attention is not a requirement for language modeling