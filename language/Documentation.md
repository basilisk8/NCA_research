# Language with Neural Cellular Automata

## Overview
After transformers, language has been assumed to always need a global attention mechanism to be reliable, this experiment tries to use purely local NCA to generate english words based on training data from 10000 most common english words on google

## Key Question

**Can an NCA with purely local rules be capable enough to learn the structure of english words based purely on local interactions**

## Architecture

- **Grid**: 2D tensor `(1, 64, 1, 7)` 
  - All channels are the letter vectors
- **Convolution / Layers**: Depth of 5 conv networks, with 3 computational convs, and 2 compression / expansion of input / output
- **Activation**: Leaky reLU
- **Steps**: 75
- **Loss**: CrossEntropy
- **Target Generation / Ground Truth**: The next letter of the word

## Files

### Training Scripts

**`7_letter_word_train.py`**
- Train NCA with the core idea the same as transformer next token prediction 
- Dataset is filtered for 2 - 7 letter words, and a random index is chosen. 
- Input is index 0 to randomly generated index, and the output is the predicted char at the next index
- Output : `letter_predictor.pth`

### Testing Scripts

**`test_genration.py`**
- Evaluate the word generative capability of NCA
- Systematically load every word, and test every possible index combination of the word
- Track total inputs and number of generated owrds that exist in dataset or in english dictionary

## Experiments & Results

### Experiment 1: Train on 2 - 7 letter words
**Setup**: The NCA trained on 2 - 7 letter words and train through bottlenecking and expanding info

**Results**:
- Total inputs given : 33104
- Number of real words generated : 25197
- Accuracy : 76.11%

**Conclusion**: The NCA learnt the general structure of words in english, further training and finetuning would increase accuracy


## Implications 
- **NCA as local generative models**: This experiment proves global attention is not a requirement, and completely local interactions are capable of complex things like language
