"""
Filename: 2_digit_training.py

Purpose: Train a 2D NCA to learn binary addition on 2 digit numbers (0-99)
         Uses 16 channels with 2D convolution to allow carry propagation.
         Tests whether NCA can generalize beyond training data.

Key Parameters:
 - Grid size: (1, 16, 2, 11) - 1 batch, 16 channels, 2 rows, 11 bits wide
 - Learning rate: 0.0001
 - Steps per example: 20
 - Training iterations: 100,000
 - Training range: 0-99
 - Loss: BCEWithLogitsLoss 

Architecture:
 - Channel 0: Input (never modified)
   - Row 0: First number in 11-bit binary
   - Row 1: Second number in 11-bit binary
 - Channel 1: Output - checked by loss function (row 0 only)
 - Channels 2-15: Hidden state for computation/carry
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Training Strategy:
 - Random pairs from 0-99
 - Test on 3 digit addition (100-999)

Expected Results:
 - Training on 0-99: 100% accuracy on seen data
 - Training on 100-999: ~99% accuracy on numbers for 3 digit numbers
 - Loss decreases to ~0.00003 (very confident predictions)

Outputs:
 - my_weights.pth: Trained model weights
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available (for Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 16→15 channels: channel 0 is input-only, channels 1-15 are updated
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)

def step(tpUpdate):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    
    Args:
        tpUpdate: Current grid state (1, 16, 2, 11)
    
    Returns:
        Updated grid with channel 0 unchanged, channels 1-15 incremented
    """
    update = torch.tanh(conv(tpUpdate))
    newGrid = tpUpdate.clone()
    newGrid[0, 1:16, :, :] = tpUpdate[0, 1:16, :, :] + update[0, 0:15, :, :]
    return newGrid

def lossFunc(finalGrid, target):
    """
    Compute BCE loss between output (channel 1, row 0) and target.
    Full row Comparison becasue we want carry propogation to generalize
    and splitting the target will result in NCA generating random values outside the range of targets
    Args:
        finalGrid: NCA state after 20 steps
        target: Expected output in binary
    
    Returns:
        BCE loss (expects logits, applies sigmoid internally)
    """
    output = finalGrid[0, 1, 0, :]
    target_slice = target[0, 0, 0, :]
    loss = nn.BCEWithLogitsLoss()(output, target_slice)  # Sigmoid built-in
    return loss

def trainingLoop(num, add1, add2, iteration):
    """
    Train on one addition problem: add1 + add2.
    
    Args:
        num: If 1, log output; if 0, silent
        add1, add2: Numbers to add (0-7)
        iteration: Current training iteration (for logging)
    """
    binary_str1 = format(add1, '011b')
    binary_str2 = format(add2, '011b')

    # Initialize grid: channel 0 holds inputs, others start at zero
    grid = torch.zeros(1, 16, 2, 11).to(device)
    for i in range(11):
        grid[0, 0, 0, i] = float(binary_str1[i])
    
    for i in range(11):
        grid[0, 0, 1, i] = float(binary_str2[i])

    # Create target: expected sum in binary
    target = torch.zeros(1, 1, 1, 11).to(device)
    total = add1 + add2
    sum_bin = format(total, '011b')
    for i in range(11):
        target[0,0,0, i] = float(sum_bin[i])
    
    # Forward pass: 20 NCA steps
    optimizer.zero_grad()
    for i in range(20):
        grid = step(grid)
    loss = lossFunc(grid, target)

    # Logging (every 1000 iterations)
    if num == 1:
        logits = grid[0, 1, 0, :].detach()
        output = torch.sigmoid(logits).cpu().numpy()  # Convert to probabilities for display
        print(f"Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
        print(f"Iteration {iteration} | {add1}+{add2}={total} | Loss: {loss.item():.6f}")
    # Backpropagation
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Initialize log file
    with open("example.txt", "w") as file:
        file.write("NCA Addition Training Log\n")
    
    # Main training loop: 100k random additions
    for i in range(100000):
        a = random.randint(0, 99) # has to be 6 or 5 when testing generalization of weights
        b = random.randint(0, 99)
        if i % 1000 == 0:
            trainingLoop(1, a, b, i)
        else:
            trainingLoop(0, a, b, i)
    
    # Save final weights
    print(f"\nTraining complete!")
    print(f"Weight shape: {conv.weight.shape}")
    print(f"Weights: {conv.weight.data}")
    print(f"Bias: {conv.bias.data}")
    
    with open("example.txt", "a") as file:
        file.write(f"\n\n=== FINAL WEIGHTS ===")
        file.write(f"\nWeight shape: {conv.weight.shape}")
        file.write(f"\nWeights: {conv.weight.data}")
        file.write(f"\nBias: {conv.bias.data}")
    torch.save(conv.state_dict(), 'my_weights.pth')