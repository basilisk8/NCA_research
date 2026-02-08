"""
Filename: train_2d_addition_nca.py

Purpose: Train a 2D NCA to learn binary addition on small numbers (0-7).
         Uses 16 channels with 2D convolution to allow carry propagation.
         Tests whether NCA can generalize beyond training data.

Key Parameters:
 - Grid size: (1, 16, 2, 4) - 1 batch, 16 channels, 2 rows, 4 bits wide
 - Learning rate: 0.01 (use 0.0001 for stable training)
 - Steps per example: 20
 - Training iterations: 100,000
 - Training range: 0-7 (can be reduced to 0-5 or 0-6 for generalization tests)
 - Loss: BCEWithLogitsLoss (expects raw logits, applies sigmoid internally)

Architecture:
 - Channel 0: Input (never modified)
   - Row 0: First number in 4-bit binary
   - Row 1: Second number in 4-bit binary
 - Channel 1: Output - checked by loss function (row 0 only)
 - Channels 2-15: Hidden state for computation/carry
 - Conv2d: 16→15 channels, kernel=(3,3), padding=(1,1)
 - Activation: tanh (bounds updates to [-1, 1])

Training Strategy:
 - Random pairs from 0-7
 - Change to random.randint(0, 5) or (0, 6) to test generalization
 - Logs every 1000 iterations to example.txt

Expected Results:
 - Training on 0-7: 100% accuracy on seen data
 - Training on 0-5: ~86% accuracy on 6-7 (generalization test)
 - Loss decreases to ~0.00001 (very confident predictions)

Outputs:
 - my_weights.pth: Trained model weights
 - example.txt: Training log with predictions and loss values
"""

import torch
import torch.nn as nn
import random

# Auto-detect GPU if available (for Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 16→15 channels: channel 0 is input-only, channels 1-15 are updated
conv = nn.Conv2d(16, 15, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.01)

def step(tpUpdate):
    """
    Single NCA step: apply 3x3 convolution and add update to channels 1-15.
    
    Args:
        tpUpdate: Current grid state (1, 16, 2, 4)
    
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
    
    Args:
        finalGrid: NCA state after 20 steps
        target: Expected output in binary
    
    Returns:
        BCE loss (expects logits, applies sigmoid internally)
    """
    output = finalGrid[0, 1, 0, :]
    target_slice = target[0, 0, 0, 0:4]
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
    binary_str1 = format(add1, '04b')
    binary_str2 = format(add2, '04b')

    # Initialize grid: channel 0 holds inputs, others start at zero
    grid = torch.zeros(1, 16, 2, 4).to(device)
    for i in range(4):
        grid[0, 0, 0, i] = float(binary_str1[i])
    
    for i in range(4):
        grid[0, 0, 1, i] = float(binary_str2[i])

    # Create target: expected sum in binary
    target = torch.zeros(1, 1, 1, 4).to(device)
    total = add1 + add2
    sum_bin = format(total, '04b')
    for i in range(4):
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
        rounded = []
        for x in output:
            if x > 0.5:
                rounded.append(1)
            else:
                rounded.append(0)
        print(f"Iteration {iteration} | {add1}+{add2}={total} | Loss: {loss.item():.6f}")
        with open("example.txt", "a") as file:
            file.write(f"\n--- Iteration {iteration} ---")
            file.write(f"\nProblem: {add1} + {add2} = {total}")
            file.write(f"\nInput binary: {binary_str1} + {binary_str2}")
            file.write(f"\nTarget: {target[0, 0, 0, :].tolist()}")
            file.write(f"\nOutput raw: {output}")
            file.write(f"\nOutput rounded: {rounded}")
            file.write(f"\nLoss: {loss.item():.6f}")

    # Backpropagation
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Initialize log file
    with open("example.txt", "w") as file:
        file.write("NCA Addition Training Log\n")
    
    # Main training loop: 100k random additions
    for i in range(100000):
        a = random.randint(0, 7) # has to be 6 or 5 when testing generalization of weights
        b = random.randint(0, 7)
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