"""
Filename: 3_digit_generalize_test.py

Purpose: Test whether trained NCA generalizes to unseen data.
         Loads weights from training on 0-99, tests on 100-999 to check generalization.

Requirements:
 - my_weights.pth (from training script with random.randint(0, 99))

Test Strategy:
 - Training data (0-99): All combinations the model was trained on
 - Unseen data (100,999): 3 digit combinations involving digits never seen during training

Expected Results:
 - Training data: 100% accuracy (500/500)
 - Unseen data: ~99% accuracy (991/1000)
 - Model learned addition, and now can apply same local rules to bigger numbers

Output Format:
 - Prints accuracy on both datasets
 - Prints the answers gettone wrong in unseen data
"""

import torch
import torch.nn as nn
import random

# Load trained model (weights to CPU for local testing)
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
    Run NCA on addition problem add1 + add2.
    
    Returns:
        grid: Final NCA state after 20 steps
        target: Expected binary output
    """
    binary_str1 = format(add1, '011b')
    binary_str2 = format(add2, '011b')

    # Initialize grid with two numbers in channel 0
    grid = torch.zeros(1, 16, 2, 11)
    for i in range(11):
        grid[0, 0, 0, i] = float(binary_str1[i])
    for i in range(11):
        grid[0, 0, 1, i] = float(binary_str2[i])

    # Create target sum
    target = torch.zeros(1, 1, 1, 11)
    total = add1 + add2
    sum_bin = format(total, '011b')
    for i in range(11):
        target[0, 0, 0, i] = float(sum_bin[i])
    
    # Run NCA for 20 steps
    for i in range(20):
        grid = step(grid)
    return grid, target

def results(inpGrid, target):
    """
    Check if NCA output matches target.
    
    Returns:
        True if prediction == expected, False otherwise
    """
    grid = inpGrid[0, 1, 0, :]  # Channel 1, row 0 = output
    targetSlice = target[0, 0, 0, :]
    predicted = [1 if x > 0 else 0 for x in grid]  # Logits: >0 → 1, <0 → 0
    expected = [int(t) for t in targetSlice]
    return predicted == expected

if __name__ == "__main__":
    # Test on training data (0 - 99)
    train_correct = 0
    train_total = 500
    for i in range(train_total):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        final_grid, target = test(a, b)
        if (results(final_grid, target)):
            train_correct += 1
    print(f"Training data (0-99): {train_correct}/500")

    # Test on unseen data (100 - 999)
    unseen_correct = 0
    unseen_total = 10000
    for i in range(unseen_total):
        a = random.randint(100, 999)
        b = random.randint(100,999)
        final_grid, target = test(a, b)
        if (results(final_grid, target)):
            unseen_correct += 1
        else:
            print(f"Numbers added were {a} + {b}")
    print(f"Unseen data (0 - 999): {unseen_correct}/{unseen_total}")