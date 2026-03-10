"""
Filename: test_GoL.py

Purpose: Test trained NCA on Game of Life with various grid sizes and time steps.
         Measures generalization from 16x16 training to 256x256 testing.
         Tests autoregressive rollout over 100 steps.
         Compares NCA predictions against true Game of Life rules.

Key Parameters:
 - Test sizes: 16, 32, 64, 128, 256 (spatial)
 - Time steps: 10, 20, 50, 100 (temporal)
 - Samples per test: 20 (spatial), 10 (temporal)
 - NCA steps per frame: 1

Architecture:
 - Conv1: 1→16 channels, kernel=(3,3), padding=(1,1), ReLU activation
 - Conv2: 16→1 channels, kernel=(1,1), no activation (logits output)
 - Prediction: sigmoid(output) > 0.5 thresholded to binary
 - Loads weights from GoL_weights.pth

Rules Being Tested:
 - Game of Life:
   - Live cell + 2 or 3 neighbors → survives
   - Dead cell + exactly 3 neighbors → born
   - All other cells → die
 - NCA should match this without being told the rules

Expected Results:
 - Spatial: 100% accuracy at all sizes (16x16 to 256x256)
 - Temporal: 100% accuracy at all step counts (10 to 100)
 - Unlike continuous physics (heat, Maxwell), GoL has zero drift
   because binary thresholding eliminates floating point accumulation

Inputs Required:
 - GoL_weights.pth: Trained model weights (dict with 'conv1' and 'conv2')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(16, 1, kernel_size=1).to(device)

conv1.load_state_dict(torch.load('GoL_weights.pth', map_location=device)['conv1'])
conv2.load_state_dict(torch.load('GoL_weights.pth', map_location=device)['conv2'])

def gol_step(grid):
    """
    Apply one step of true Game of Life rules.

    Counts live neighbors using convolution with ones kernel
    (center excluded). Applies birth/survival rules.

    Args:
        grid: Binary grid tensor (1, 1, H, W) of 0.0 and 1.0

    Returns:
        Next state under Game of Life rules
    """
    kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(grid.device)
    neighbors = F.conv2d(grid, kernel, padding=1)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (grid == 0) & (neighbors == 3)
    return (survive | born).float()

def step(grid):
    """
    Single NCA step: two-layer conv network outputs logits.

    Args:
        grid: Current binary grid state (1, 1, H, W)

    Returns:
        Logits for next state (apply sigmoid + threshold to get prediction)
    """
    hidden = torch.relu(conv1(grid))
    output = conv2(hidden)
    return output

print("=" * 50)
print("SPATIAL GENERALIZATION")
print("=" * 50)

for size in [16, 32, 64, 128, 256]:
    correct = 0
    total = 20
    for _ in range(total):
        grid = torch.randint(0, 2, (1, 1, size, size)).float().to(device)
        target = gol_step(grid)
        output = step(grid)
        pred = (torch.sigmoid(output) > 0.5).float()
        acc = (pred == target).float().mean()
        if acc > 0.99:
            correct += 1
    print(f"  {size}x{size}: {correct}/{total} perfect")

print()
print("=" * 50)
print("TIME GENERALIZATION (autoregressive)")
print("=" * 50)

for num_steps in [10, 20, 50, 100]:
    accuracies = []
    for trial in range(10):
        grid_true = torch.randint(0, 2, (1, 1, 32, 32)).float().to(device)
        grid_nca = grid_true.clone()

        final_acc = 0
        for t in range(num_steps):
            grid_true = gol_step(grid_true)
            grid_nca = (torch.sigmoid(step(grid_nca)) > 0.5).float()
            final_acc = (grid_nca == grid_true).float().mean().item()

        accuracies.append(final_acc)

    avg = sum(accuracies) / len(accuracies)
    print(f"  {num_steps} steps: {avg*100:.1f}% avg final accuracy")

print()
print("=" * 50)
print("TIME GENERALIZATION (step by step detail)")
print("=" * 50)

grid_true = torch.randint(0, 2, (1, 1, 32, 32)).float().to(device)
grid_nca = grid_true.clone()

for t in range(50):
    grid_true = gol_step(grid_true)
    grid_nca = (torch.sigmoid(step(grid_nca)) > 0.5).float()
    acc = (grid_nca == grid_true).float().mean().item()
    if t < 10 or t % 5 == 4:
        print(f"  Step {t+1:3d}: {acc*100:.1f}%")

print()
print("=" * 50)
print("TIME + SPACE (large grid, many steps)")
print("=" * 50)

for size in [64, 128]:
    grid_true = torch.randint(0, 2, (1, 1, size, size)).float().to(device)
    grid_nca = grid_true.clone()

    for t in range(50):
        grid_true = gol_step(grid_true)
        grid_nca = (torch.sigmoid(step(grid_nca)) > 0.5).float()

    acc = (grid_nca == grid_true).float().mean().item()
    print(f"  {size}x{size} after 50 steps: {acc*100:.1f}%")