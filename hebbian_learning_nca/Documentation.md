# Testing Hebbian Learning as a learning algorithm

## Overview
This experiment tries to see if NCA's can be trained with Hebbian learning of neurons that fire together wire together. The experiment gives a binary input of 0110 and expects an binary output of 1111. 1 Big assumption behind this experiment is that NCA's purely local structure will compensate for the lack of knowledge we have about how hebbian learning actually works

## Key Question

**Can an NCA be trained to modify patterns purely based on hebbian learning?**

## Architecture

- **Grid**: 2D tensor `(batch, channels, height, width)`
  - Channel 1-9: Per cell weights for each of it's neighbor
  - Channel 10: Activation value, also output channel
- **Steps**: 20 forward passes per example
- **Loss**: Hebbian learning of reward a positive connection between cells

## Files

### Training Scripts

**`hebbian_learning_nca.py`**
- Main training script for NCA
- Grid: 2D (1×11×2×11)
- Training range: Pre defined 

## Experiments & Results

### Experiment 1: NCA learning to modify input to produce output
**Setup**: Give 0110 as input and set target to 1111
**Result**: Failed, with reward at a constant -1 
**Conclusion**: Hebbian learning has gaps that the experiment highlights and it's not computationally cheap to run either
**Reasoning**: These are the reasons I think the experiment failed 
    - Per cell weights is unstable artichture to modify because per cell weights add too many parameters
    - The experiment realied on the assumption that NCA's local structure withh fill in the knowledge gaps we have about hebbian learing
    - Computationally it is more expensive to compute per cell weights because they add a lot more parameters
**Conclusion**: The NCA **failed to learn** using hebbian learning