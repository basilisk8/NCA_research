"""
Filename: 4_elements_sort.py

Purpose: Train a 5-phase NCA to sort 4 integers by actually moving values
         to their correct sorted positions (value routing, not ranking).
         Uses Hungarian matching loss to prevent interpolation and force
         the network to preserve original input values during sorting.

Key Parameters:
 - Grid size: (1, 32, 1, 4) - 1 batch, 32 channels, 1 row, 4 elements
 - Learning rate: 0.0001
 - Steps per phase: 20 (100 total across 5 phases)
 - Training iterations: 700,000
 - Input range: 0-255 (normalized to [-2, 2] via (v-128)/64)
 - Loss: MSE (ordering) + Hungarian matching (value preservation)

Architecture:
 - 5 separate Conv2d networks (phases), each with independent weights
 - Each phase runs 20 NCA steps before passing state to the next
 - Channel 0: Input (never modified)
 - Channel 1: Output - read after all 5 phases complete
 - Channels 2-31: Hidden state (scratch space, persists across phases)
 - Conv2d: 32→31 channels per phase, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Why 5 Phases:
 - Single NCA can't simultaneously compare, decide, and swap values
 - Multiple phases with different weights allow sequential operations
 - Similar to how deep CNNs use different layers for different functions
 - The NCA decides what each phase does (not manually assigned)

Loss Function:
 - Sort Loss (MSE): Penalizes difference from sorted target order
 - Match Loss (Hungarian): Finds optimal one-to-one assignment between
   output values and input values, penalizes total assignment cost.
   Prevents the "interpolation trap" where the NCA outputs a generic
   ascending ramp instead of the actual input values rearranged.
 - Combined: sort_loss + match_loss

Key Findings:
 - Achieves 60% accuracy within ±5 tolerance on width 4
 - Order is consistently correct, values are close but not exact
 - Precision limited by tanh accumulation over 100 steps
 - Does not generalize to width 5 or 6 (0% accuracy)
 - Ranking approach is more accurate but doesn't move values
 - This approach actually routes values, proving NCA can do it

Outputs:
 - 4_elements_sort.pth: Trained model weights for all 5 phases
"""

import torch
import torch.nn as nn
import random
from scipy.optimize import linear_sum_assignment

# Auto-detect GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

channels = 32

# 5 independent conv networks, one per phase
# Each maps 32→31 channels: channel 0 is input-only, channels 1-31 are updated
conv1 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv3 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv4 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)
conv5 = nn.Conv2d(channels, channels - 1, kernel_size=3, padding=1).to(device)

# Small initial weights for stable training
for conv in [conv1, conv2, conv3, conv4, conv5]:
    nn.init.xavier_uniform_(conv.weight, gain=0.1)
    nn.init.zeros_(conv.bias)

# All parameters trained jointly with single optimizer
all_params = list(conv1.parameters()) + \
             list(conv2.parameters()) + \
             list(conv3.parameters()) + \
             list(conv4.parameters()) + \
             list(conv5.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.0001)


def step(grid, conv):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-31.
    Channel 0 (input) is never modified.

    Args:
        grid: Current grid state (1, 32, 1, 4)
        conv: Which phase's conv to apply

    Returns:
        Updated grid with channel 0 unchanged, channels 1-31 incremented
    """
    update = torch.tanh(conv(grid))
    newGrid = grid.clone()
    newGrid[0, 1:channels, :, :] = grid[0, 1:channels, :, :] + update[0, :, :, :]
    return newGrid


def hungarian_loss(output, input_tensor):
    """
    Compute optimal matching cost between output and input values.
    Uses the Hungarian algorithm to find the one-to-one assignment
    that minimizes total distance. This forces the network to output
    values that are actual rearrangements of the input, not interpolations.

    Args:
        output: NCA output values (4,)
        input_tensor: Original input values (4,)

    Returns:
        Sum of distances under optimal matching (differentiable via cost matrix)
    """
    # Pairwise distance matrix between all output-input pairs
    cost = torch.cdist(output.unsqueeze(1), input_tensor.unsqueeze(1))
    # Find optimal assignment (non-differentiable, but cost matrix is differentiable)
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    # Return sum of matched distances (gradients flow through cost matrix)
    return cost[row_ind, col_ind].sum()


def train_step(values, iteration, log=False):
    """
    Train on one sorting problem: sort 4 integers.

    Args:
        values: List of 4 integers (0-255) to sort
        iteration: Current training iteration (for logging)
        log: If True, print progress
    """
    width = 4

    # Initialize grid: channel 0 holds normalized input values
    grid = torch.zeros(1, channels, 1, width).to(device)
    for i in range(width):
        grid[0, 0, 0, i] = (values[i] - 128.0) / 64.0  # normalize to [-2, 2]

    # Compute targets in same normalized space
    sorted_values = sorted(values)
    target = torch.tensor(
        [(v - 128.0) / 64.0 for v in sorted_values],
        device=device
    ).float()
    input_tensor = torch.tensor(
        [(v - 128.0) / 64.0 for v in values],
        device=device
    ).float()

    optimizer.zero_grad()

    # Run all 5 phases sequentially, 20 steps each
    # Hidden state persists across phases (no scratch clearing)
    for s in range(20):
        grid = step(grid, conv1)
    for s in range(20):
        grid = step(grid, conv2)
    for s in range(20):
        grid = step(grid, conv3)
    for s in range(20):
        grid = step(grid, conv4)
    for s in range(20):
        grid = step(grid, conv5)

    # Read sorted output from channel 1
    output = grid[0, 1, 0, :width]

    # Sort loss: output should be in ascending order matching sorted target
    sort_loss = nn.MSELoss()(output, target)
    # Match loss: output values should match input values (just rearranged)
    match_loss = hungarian_loss(output, input_tensor)
    # Combined loss: must be sorted AND preserve original values
    loss = sort_loss + match_loss

    if log:
        # Convert back to integer scale for readable output
        pred = [round(output[i].item() * 64.0 + 128.0) for i in range(width)]
        correct = pred == sorted_values
        print(f"Iter {iteration} | Loss: {loss.item():.6f} | {'✓' if correct else '✗'}")
        print(f"  Input:    {values}")
        print(f"  Expected: {sorted_values}")
        print(f"  Got:      {pred}")
        print(f"  Sort Loss: {sort_loss.item():.4f} | Match Loss: {match_loss.item():.4f}")
        print()

    # Backprop through all 100 steps across all 5 phases
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()


if __name__ == "__main__":
    # Main training loop: 700k random sorting problems
    for i in range(700000):
        values = [random.randint(0, 255) for _ in range(4)]
        log = (i % 5000 == 0)
        train_step(values, i, log=log)
    
    # Save all 5 phase weights
    torch.save({
        'conv1': conv1.state_dict(),
        'conv2': conv2.state_dict(),
        'conv3': conv3.state_dict(),
        'conv4': conv4.state_dict(),
        'conv5': conv5.state_dict(),
    }, '4_elements_sort.pth')
    print("Saved")