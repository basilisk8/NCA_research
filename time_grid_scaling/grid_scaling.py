"""
Filename: grid_scaling.py

Purpose: Test if grid width affects NCA training speed.

Hypothesis: GPU parallelizes across cells, so width shouldn't matter.

Method: Train 10k iterations on width 11 vs width 30. Compare time.

Results:
 - Width 11: 117.9s
 - Width 30: 111.0s
 - Conclusion: Grid width doesn't affect speed.
"""

import torch
import torch.nn as nn
import random
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)

def step(grid):
    """Single NCA step: conv + tanh + add to channels 1-15."""
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:16, :, :] = grid[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid

def trainingLoop(log, add1, add2, iteration, grid_size):
    """Train on one addition problem with specified grid width."""
    binary_str1 = format(add1, '011b')
    binary_str2 = format(add2, '011b')
    sum_bin = format(add1 + add2, '011b')

    grid = torch.zeros(1, 16, 2, grid_size, device=device)
    target = torch.zeros(1, 1, 1, 11, device=device)

    for i in range(11):
        grid[0, 0, 0, i] = float(binary_str1[i])
        grid[0, 0, 1, i] = float(binary_str2[i])
        target[0, 0, 0, i] = float(sum_bin[i])

    optimizer.zero_grad()
    for _ in range(20):
        grid = step(grid)
    
    loss = nn.BCEWithLogitsLoss()(grid[0, 1, 0, :11], target[0, 0, 0, :])
    
    if log:
        print(f"Iteration {iteration} | {add1}+{add2}={add1+add2} | Loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Test grid width 11
    print("Training on grid size 11")
    t1 = time.time()
    for i in range(10000):
        a, b = random.randint(0, 99), random.randint(0, 99)
        trainingLoop(i % 1000 == 0, a, b, i, 11)
    t2 = time.time()
    print(f"Time: {t2 - t1:.1f}s\n")

    # Test grid width 30
    print("Training on grid size 30")
    t3 = time.time()
    for i in range(10000):
        a, b = random.randint(0, 99), random.randint(0, 99)
        trainingLoop(i % 1000 == 0, a, b, i, 30)
    t4 = time.time()
    print(f"Time: {t4 - t3:.1f}s")
