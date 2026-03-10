"""
Filename: train_gol.py

Purpose: Train a 2D NCA to learn Game of Life rules from examples.
         NCA observes binary grid states and learns to predict
         the next state after 1 Game of Life step.
         No rules given - NCA discovers birth/survival logic from data.

Key Parameters:
 - Grid size: (1, 1, 16, 16) - 1 batch, 1 channel, 16x16 grid
 - Learning rate: 0.001
 - NCA steps per frame: 1
 - Training iterations: 50,000
 - Input: Random binary grids (0 or 1)
 - Loss: BCEWithLogitsLoss (binary classification per cell)

Architecture:
 - Conv1: 1→16 channels, kernel=(3,3), padding=(1,1), ReLU activation
 - Conv2: 16→1 channels, kernel=(1,1), no activation (logits output)
 - No residual connection - output replaces input entirely
 - 16 hidden channels provide nonlinearity needed for GoL rules
   (1 channel fails because GoL requires non-monotonic response:
   3 neighbors = born, but 2 and 4 neighbors = no birth)

Rules Being Learned:
 - Game of Life:
   - Live cell + 2 or 3 neighbors → survives
   - Dead cell + exactly 3 neighbors → born
   - All other cells → die
 - NCA discovers these from (input, target) pairs only

Training Strategy:
 - Random binary 16x16 grids each iteration
 - Target = one Game of Life step applied to input
 - Single NCA forward pass predicts next state

Expected Results:
 - Loss decreases to ~0.000003
 - NCA perfectly learns Game of Life rules
 - Generalizes to any grid size and any number of time steps

Outputs:
 - GoL_weights.pth: Dict with 'conv1' and 'conv2' state dicts
"""

import torch
import torch.nn.functional as F
import random
import torch.nn as nn

# Auto-detect GPU if available (for Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(16, 1, kernel_size=1).to(device)

optimizer = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=0.001)

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

    # alive + 2 or 3 neighbors → survive
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))

    # dead + exactly 3 neighbors → born
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

def lossFunc(finalGrid, target):
    """
    Compute BCE loss between NCA output logits and target binary state.

    Uses BCEWithLogitsLoss because NCA outputs raw logits,
    sigmoid is applied internally by the loss function.

    Args:
        finalGrid: NCA output logits (1, 1, H, W)
        target: True next Game of Life state (1, 1, H, W)

    Returns:
        BCE loss per cell averaged over grid
    """
    output = finalGrid[0, 0, :, :]
    target_slice = target[0, 0, :, :]
    loss = nn.BCEWithLogitsLoss()(output, target_slice)
    return loss

def trainingLoop(num, width, iteration):
    """
    Train on one Game of Life example.

    Args:
        num: If 1, log output; if 0, silent
        width: Unused (fixed 16x16 grid)
        iteration: Current training iteration (for logging)
    """
    # Random binary grid
    grid = torch.randint(0, 2, (1, 1, 16, 16)).float().to(device)
    target = gol_step(grid)

    # Forward pass: 1 NCA step
    optimizer.zero_grad()
    for _ in range(1):
        grid = step(grid)
    loss = lossFunc(grid, target)

    # Logging
    if num == 1:
        print(f"Iter {iteration} | loss={loss.item():.6f}")

    # Backpropagation
    loss.backward()
    optimizer.step()

if __name__ == "__main__":

    # Main training loop: 50k random examples
    for i in range(50000):
        width = random.randint(8, 16)
        if i % 5000 == 0:
            trainingLoop(1, width, i)
        else:
            trainingLoop(0, width, i)

    # Save final weights
    torch.save({
        'conv1': conv1.state_dict(),
        'conv2': conv2.state_dict()
    }, 'GoL_weights.pth')