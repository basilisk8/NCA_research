"""
Filename: multi_shape_nca.py

Purpose: Train a single NCA to grow 5 different binary shapes from learned seed embeddings.
         One set of weights, one local rule. Different seeds produce different shapes.
         This is the NCA equivalent of MNIST - proving one rule can handle multiple targets.

Key Parameters:
 - Grid size: (1, 25, 40, 40) - 1 batch, 25 channels, 40x40 grid
 - Learning rate: 0.002
 - Steps per example: random 64-96 (forces temporal stability)
 - Training iterations: 100,000
 - Shapes: circle, square, plus, triangle, line
 - Loss: BCEWithLogitsLoss

Architecture:
 - Channel 0: Visible output - checked by loss function
 - Channels 1-20: Hidden state for growth computation
 - Channels 21-24: Seed embedding (4 channels, never updated during forward pass)

 - Perception: 3 fixed Sobel filters (identity, horizontal gradient, vertical gradient)
   applied to all 25 channels = 75 perception channels per cell
   These are NOT learned. They just let each cell sense its neighbors.

 - Per-cell MLP (two 1x1 convolutions):
   - fc1: 75 to 128 channels with ReLU (each cell thinks about its perception)
   - fc2: 128 to 21 channels, zero-initialized (each cell decides what to change)
   - 1x1 conv = regular neural network layer applied independently at every cell

 - Stochastic update mask: 50% of cells randomly skip each step
   Forces robustness - no single cell is critical to the growth program

 - Update scaling: multiply by 0.1 to prevent value explosion over 80+ steps

 - Seed embedding: nn.Embedding(5, 4) - learned lookup table
   Each shape gets a 4-number vector, learned by backprop alongside NCA weights
   Placed in channels 21-24 of the center cell at step 0

Training Strategy:
 - Random shape selected each iteration
 - Seed placed at center cell (20, 20)
 - NCA runs 64-96 steps, shape grows outward from seed
 - Loss computed on channel 0 vs target shape

Expected Results:
 - Loss drops below 0.001 by ~30k iterations
 - 99%+ pixel accuracy on all 5 shapes
 - Each seed produces a visually distinct, correct shape

Outputs:
 - nca_weights.pth: Trained NCA weights + seed embeddings + optimizer state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

# Fixed Sobel filters - never change during training
# Let each cell detect horizontal and vertical gradients in its neighborhood
sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device) / 8.0
sobel_y = sobel_x.T
identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).to(device)
filters = torch.stack([identity, sobel_x, sobel_y])

# Per-cell MLP as two 1x1 convolutions
# fc1: 75 perception channels to 128 hidden (thinking)
# fc2: 128 hidden to 21 update channels (deciding what to change)
fc1 = nn.Conv2d(75, 128, 1).to(device)
fc2 = nn.Conv2d(128, 21, 1).to(device)

# Zero-initialize fc2 so NCA starts as identity (no change at step 0)
nn.init.zeros_(fc2.weight)
nn.init.zeros_(fc2.bias)

# Learned seed lookup table: 5 shapes, 4 values each
# Starts random, backprop figures out what values produce which shapes
seed_embed = nn.Embedding(5, 4).to(device)

# Train NCA weights and seed embeddings together
optimizer = torch.optim.Adam(
    list(fc1.parameters()) + list(fc2.parameters()) + list(seed_embed.parameters()),
    lr=2e-3
)

# Precompute target shapes
targets = []

# Shape 0: Circle
t = torch.zeros(40, 40, device=device)
for y in range(40):
    for x in range(40):
        if (x - 20) ** 2 + (y - 20) ** 2 <= 10 ** 2:
            t[y, x] = 1.0
targets.append(t)

# Shape 1: Square
t = torch.zeros(40, 40, device=device)
t[10:30, 10:30] = 1.0
targets.append(t)

# Shape 2: Plus sign
t = torch.zeros(40, 40, device=device)
t[15:25, 5:35] = 1.0
t[5:35, 15:25] = 1.0
targets.append(t)

# Shape 3: Triangle
t = torch.zeros(40, 40, device=device)
for y in range(20, 35):
    width = y - 20
    t[y, 20 - width:20 + width + 1] = 1.0
targets.append(t)

# Shape 4: Horizontal line
t = torch.zeros(40, 40, device=device)
t[18:22, 5:35] = 1.0
targets.append(t)


def perceive(grid):
    """
    Each cell gathers info about itself and neighbors using 3 fixed filters.
    Identity: what is my current value (25 channels)
    Sobel_x: whats changing left/right (25 channels)
    Sobel_y: whats changing up/down (25 channels)
    Returns 75 channels of perception per cell.
    """
    ch = grid.shape[1]
    perceived = []
    for f in filters:
        kernel = f.view(1, 1, 3, 3).expand(ch, -1, -1, -1)
        perceived.append(F.conv2d(grid, kernel, padding=1, groups=ch))
    return torch.cat(perceived, dim=1)


def step(grid):
    """
    Single NCA step: perceive neighbors, think, update.
    1. Sobel perception gathers 75 channels of neighbor info
    2. fc1 + ReLU: think about it (75 to 128)
    3. fc2: decide what to change (128 to 21)
    4. Stochastic mask: randomly skip 50% of cells
    5. Add scaled update to channels 0-20, seed channels 21-24 untouched
    """
    perceived = perceive(grid)
    update = fc2(torch.relu(fc1(perceived)))

    # Each cell has 50% chance of updating, forces robustness
    mask = (torch.rand(1, 1, grid.shape[2], grid.shape[3], device=device) < 0.5)

    new_grid = grid.clone()
    # Only update channels 0-20, seed channels 21-24 stay fixed
    # Multiply by 0.1 to keep values stable over 80+ steps
    new_grid[:, 0:21] = grid[:, 0:21] + update * mask * 0.1
    return new_grid


def train(shape_index, log, iteration):
    """
    Train on one shape. Place seed at center, run NCA, compute loss.
    """
    target = targets[shape_index]

    # Fresh grid: all zeros except seed at center
    grid = torch.zeros(1, 25, 40, 40, device=device)
    seed = seed_embed(torch.tensor(shape_index, device=device))
    grid[0, 21:25, 20, 20] = seed

    optimizer.zero_grad()

    # Random step count forces NCA to be stable across a range of timesteps
    steps = random.randint(64, 96)
    for s in range(steps):
        grid = step(grid)

    # BCE loss on channel 0 vs target shape
    loss = F.binary_cross_entropy_with_logits(grid[0, 0], target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(fc1.parameters()) + list(fc2.parameters()), 1.0)
    optimizer.step()

    if log:
        output = grid[0, 0].detach()
        print("Iter", iteration, "| Shape", shape_index,
              "| Loss:", round(loss.item(), 6),
              "| Out range: [", round(output.min().item(), 2), ",", round(output.max().item(), 2), "]")


if __name__ == "__main__":
    for i in range(100000):
        shape_index = random.randint(0, 4)
        if i % 1000 == 0:
            train(shape_index, True, i)
        else:
            train(shape_index, False, i)

    torch.save({
        'fc1_state_dict': fc1.state_dict(),
        'fc2_state_dict': fc2.state_dict(),
        'seed_embed_state_dict': seed_embed.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'nca_weights.pth')
    print("Weights saved to nca_weights.pth")