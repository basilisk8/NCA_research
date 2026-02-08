"""
Filename: test_2digit_addition.py

Purpose: Test if NCA trained on 4-bit (0-7) addition can extrapolate to 8-bit (2-digit) addition.
         This tests whether the NCA learned LOCAL addition rules that transfer to wider grids.

Hypothesis:
 - If NCA learned bit-wise addition + carry propagation as local rules,
   it should work on wider grids even though trained on width=4
 - Expected: Partial success (50-70%) since grid width changed

Requirements:
 - my_weights.pth (trained on 4-bit addition, 0-7 range)

Test Setup:
 - Grid expanded from width=4 to width=8
 - Numbers: 10-99 (2 digits, never seen during training)
 - Same architecture: 16 channels, 3x3 conv, 20 steps
 - 100 random test cases

Key Question:
 Can local learned patterns transfer to different grid sizes?

Expected Results:
 - ~64% accuracy observed (better than random 0%)
 - Shows partial transfer of learned patterns
 - Full success requires training on width=8
"""

import torch
import torch.nn as nn
import random

# Load weights trained on width=4 grid (0-7 addition)
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1))
conv.load_state_dict(torch.load('my_weights.pth', map_location='cpu'))

def step(tpUpdate):
    """Single NCA step: apply conv and update channels 1-15."""
    update = torch.tanh(conv(tpUpdate))
    newGrid = tpUpdate.clone()
    newGrid[0, 1:16, :, :] = tpUpdate[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid

def test(add1, add2):
    """
    Run NCA on 2-digit addition using 8-bit grid.
    
    NOTE: Grid is now width=8 (double the training width)
    """
    binary_str1 = format(add1, '08b')  # 8 bits
    binary_str2 = format(add2, '08b')  # 8 bits

    # Grid now width=8 instead of width=4
    grid = torch.zeros(1, 16, 2, 8)  # width 8
    for i in range(8):
        grid[0, 0, 0, i] = float(binary_str1[i])
    for i in range(8):
        grid[0, 0, 1, i] = float(binary_str2[i])

    # Target is also 8 bits
    target = torch.zeros(1, 1, 1, 8)
    total = add1 + add2
    sum_bin = format(total, '08b')  # 8 bits
    for i in range(8):
        target[0, 0, 0, i] = float(sum_bin[i])
    
    # Same 20 steps as training
    for i in range(20):
        grid = step(grid)
    return grid, target

def results(inpGrid, target):
    """
    Compare NCA output to expected result.
    
    Returns:
        predicted: List of 0/1 predictions
        expected: List of 0/1 targets
        is_correct: True if all bits match
    """
    grid = inpGrid[0, 1, 0, :]
    targetSlice = target[0, 0, 0, :]
    predicted = [1 if x > 0 else 0 for x in grid]
    expected = [int(t) for t in targetSlice]
    return predicted, expected, predicted == expected

if __name__ == "__main__":
    correct = 0
    total = 100
    
    # Test on 100 random 2-digit additions
    for _ in range(total):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        grid, target = test(a, b)
        pred, exp, is_correct = results(grid, target)
        
        if is_correct:
            correct += 1
        else:
            # Print failures with binary output for debugging
            print(f"{a}+{b}={a+b} | pred {''.join(map(str,pred))} exp {''.join(map(str,exp))}")
    
    print(f"\n2-digit accuracy: {correct}/{total}")