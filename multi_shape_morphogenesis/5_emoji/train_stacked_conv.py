"""
Filename: stacked_conv_nca.py

Purpose: Train a single NCA to grow 5 different binary shapes from learned seed embeddings.
         Uses stacked conv architecture instead of Mordvintsev's Sobel + 1x1 approach.
         One set of weights, one local rule. Different seeds produce different shapes.
         Proves general-purpose NCA architecture can match specialized morphogenesis architecture.

Key Parameters:
 - Grid size: (1, 25, 40, 40) - 1 batch, 25 channels, 40x40 grid
 - Learning rate: 0.0005
 - Optimizer: AdamW with weight_decay=0.01
 - Steps per example: random 64-96 (forces temporal stability)
 - Training iterations: 100,000
 - Shapes: circle, square, plus, triangle, line
 - Loss: BCEWithLogitsLoss

Architecture:
 - Channel 0: Visible output - checked by loss function
 - Channels 1-20: Hidden state for growth computation
 - Channels 21-36: Seed embedding (16 channels, never updated during forward pass)

 - Stacked conv approach (no fixed Sobel filters):
   - Conv1: 80 to 128 channels, 3x3 kernel (perceive neighbors + begin thinking)
   - Conv2: 128 to 128 channels, 1x1 kernel (deeper per-cell processing)
   - Conv3: 128 to 64 channels, 1x1 kernel, zero-initialized (decide update)
   - ReLU between conv1-conv2 and conv2-conv3
   - Unlike Mordvintsev, perception is learned not hardcoded

 - Stochastic update mask: 50% of cells randomly skip each step
   Forces robustness - no single cell is critical to the growth program

 - Update scaling: multiply by 0.1 to prevent value explosion over 80+ steps

 - Seed embedding: nn.Embedding(5, 16) - learned lookup table
   Each shape gets a 16-number vector, learned by backprop alongside NCA weights
   Placed in channels 21-36 of the center cell at step 0

Training Strategy:
 - Random shape selected each iteration
 - Seed placed at center cell (20, 20)
 - NCA runs 64-96 steps, shape grows outward from seed
 - Loss computed on channel 0 vs target shape
 - Gradient clipping at 1.0 to prevent instability

Expected Results:
 - Loss drops below 0.001 by ~40k iterations
 - 100% pixel accuracy on all 5 shapes
 - Each seed produces a visually distinct, correct shape
 - Matches or beats Mordvintsev Sobel+1x1 architecture on same task

Outputs:
 - stacked_conv_weights.pth: Trained conv weights + seed embeddings
"""

import torch
import torch.nn.functional as F
import random
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

# Stacked conv architecture - all perception is learned
conv1 = nn.Conv2d(80, 128, kernel_size=3, padding=1).to(device)  # perceive + think
conv2 = nn.Conv2d(128, 128, kernel_size=1).to(device)             # think deeper
conv3 = nn.Conv2d(128, 64, kernel_size=1).to(device)              # decide update

# Zero-init final layer so NCA starts as identity (no change at step 0)
nn.init.zeros_(conv3.weight)
nn.init.zeros_(conv3.bias)

# Learned seed lookup table: 5 shapes, 16 values each
seed_embed = nn.Embedding(5, 16).to(device)

# AdamW prevents weight explosion over long training runs
optimizer = torch.optim.AdamW(
    list(conv1.parameters()) + list(conv2.parameters()) + list(conv3.parameters()) + list(seed_embed.parameters()),
    lr=0.0005, weight_decay=0.01
)

targets = torch.load('emoji_targets.pt')
targets = [t.to(device) for t in targets]

def step(grid):
    """
    Single NCA step: stacked convolutions with residual update.

    1. Conv1 (3x3): perceive neighbors and begin processing
    2. Conv2 (1x1): deeper per-cell thinking
    3. Conv3 (1x1): decide what to change (zero-init means starts as no-op)
    4. Stochastic mask: randomly skip 50% of cells
    5. Add scaled update to channels 0-20, seed channels 21-24 untouched

    Args:
        grid: Current NCA state (1, 25, 40, 40)

    Returns:
        Updated grid with seed channels unchanged
    """
    x = torch.relu(conv1(grid))
    x = torch.relu(conv2(x))
    update = conv3(x)

    # Each cell has 50% chance of updating, forces robustness
    mask = (torch.rand(1, 1, 40, 40, device=device) < 0.5)

    # Only update channels 0-20, seed channels 21-36 stay fixed
    # Multiply by 0.1 to keep values stable over 80+ steps
    new_grid = grid.clone()
    new_grid[:, 0:64] = grid[:, 0:64] + update * mask * 0.1
    return new_grid

def lossFunc(finalGrid, target):
    """
    MSE loss between final grid's channel 0 and target shape. Measures how well the NCA grew the desired shape.

    Args:
        finalGrid: NCA state after growth (1, 25, 40, 40)
        target: Target binary shape (40, 40)

    Returns:
        BCE loss averaged over all pixels
    """
    output = finalGrid[0, 0:3, :, :]
    loss = nn.MSELoss()(output, target)
    return loss

def trainingLoop(log, shape_index, iteration):
    """
    Train on one shape growth example.

    Args:
        log: If True, print loss
        shape_index: Which shape to grow (0-4)
        iteration: Current training iteration for logging
    """
    # Fresh grid: all zeros except seed embedding at center cell
    grid = torch.zeros(1, 80, 40, 40, device=device)
    seed = seed_embed(torch.tensor(shape_index, device=device))
    grid[0, 64:80, 20, 20] = seed

    optimizer.zero_grad()

    # Random step count forces stability across time range
    steps = random.randint(64, 96)
    for _ in range(steps):
        grid = step(grid)

    loss = lossFunc(grid, targets[shape_index])

    if log:
        print("Iter", iteration, "| Shape", shape_index, "| loss:", round(loss.item(), 6))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(conv1.parameters()) + list(conv2.parameters()) + list(conv3.parameters()),
        1.0
    )
    optimizer.step()


if __name__ == "__main__":
    for i in range(100000):
        shape_index = random.randint(0, 4)
        if i % 5000 == 0:
            trainingLoop(True, shape_index, i)
        else:
            trainingLoop(False, shape_index, i)

    torch.save({
        'conv1': conv1.state_dict(),
        'conv2': conv2.state_dict(),
        'conv3': conv3.state_dict(),
        'seed_embed': seed_embed.state_dict()
    }, 'stacked_conv_emoji_weights.pth')
    print("Weights saved to stacked_conv_emoji_weights.pth")