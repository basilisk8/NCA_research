"""
Filename: generalize_test.py

Purpose: Test whether trained NCA generalizes to unseen data.
         Loads weights from training on 0-5, tests on 0-7 to check generalization.

Requirements:
 - my_weights.pth (from training script with random.randint(0, 5))

Test Strategy:
 - Training data (0-5): All 36 combinations the model was trained on
 - Unseen data (6,7): 28 combinations involving 6 or 7 (never seen during training)

Expected Results:
 - Training data: 100% accuracy (36/36)
 - Unseen data: ~85% accuracy (24/28) if model learned addition algorithm
 - If accuracy < 50% on unseen data → model memorized, didn't generalize

Output Format:
 - Prints accuracy on both datasets
 - Lists failed cases (e.g., "7+5 ✗")
"""

import torch
import torch.nn as nn

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
    binary_str1 = format(add1, '04b')
    binary_str2 = format(add2, '04b')

    # Initialize grid with two numbers in channel 0
    grid = torch.zeros(1, 16, 2, 4)
    for i in range(4):
        grid[0, 0, 0, i] = float(binary_str1[i])
    for i in range(4):
        grid[0, 0, 1, i] = float(binary_str2[i])

    # Create target sum
    target = torch.zeros(1, 1, 1, 4)
    total = add1 + add2
    sum_bin = format(total, '04b')
    for i in range(4):
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
    # Test on training data (0-5)
    train_correct = 0
    for i in range(6):
        for j in range(6):
            grid, target = test(i, j)
            if results(grid, target):
                train_correct += 1
    print(f"Training data (0-5): {train_correct}/36")

    # Test on unseen data (any combination with 6 or 7)
    unseen_correct = 0
    unseen_total = 0
    for i in range(8):
        for j in range(8):
            if i >= 6 or j >= 6:
                grid, target = test(i, j)
                if results(grid, target):
                    unseen_correct += 1
                else:
                    print(f"{i}+{j} ✗")
                unseen_total += 1
    print(f"Unseen data (6,7): {unseen_correct}/{unseen_total}")