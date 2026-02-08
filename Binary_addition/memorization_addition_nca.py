"""
Filename: memorization_addition_nca.py

Purpose: Proof-of-concept NCA that memorizes a single addition problem (3 + 5 = 8).
         Uses 1D convolution with continuous backprop to reduce loss on one specific example.

Key Parameters:
 - Grid size: (1, 2, 8) - 1 batch, 2 channels, 8 width
 - Learning rate: 0.01
 - Steps: 10 forward passes per training iteration
 - Loss: MSE (easier for memorization, bad for generalization)

Architecture:
 - Channel 0: Input (never modified) - holds the two numbers
 - Channel 1: Output/processing - updated by convolution
 - Conv1d: 2→1 channels, kernel=3, padding=1
 - Activation: tanh

Results:
 - Successfully memorizes 3 + 5 = 8 with near-zero loss
 - Does NOT generalize to other additions (as expected)
"""
import torch
import torch.nn as nn

# Target: 8 in binary (1000)
target = torch.zeros(1, 1, 8)
target[0, 0, 0:4] = torch.tensor([1., 0., 0., 0.])

conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.01)

def step(tpUpdate):
    """Single NCA step: apply conv, add update to channel 1"""
    update = torch.tanh(conv(tpUpdate))
    newGrid = tpUpdate.clone()
    newGrid[0, 1, :] = tpUpdate[0, 1, :] + update[0, 0, :]
    return newGrid

def lossFunc(finalGrid):
    """MSE between output and target (first 4 bits)"""
    return ((finalGrid[0, 1, 0:4] - target[0, 0, 0:4]) ** 2).mean()

def trainingLoop(verbose):
    """Run one training iteration: 10 steps + backprop"""
    grid = torch.zeros(1, 2, 8)
    grid[0, 0, 0:4] = torch.tensor([0., 0., 1., 1.])  # 3 in binary
    grid[0, 0, 4:8] = torch.tensor([0., 1., 0., 1.])  # 5 in binary
    
    optimizer.zero_grad()
    for _ in range(10):
        grid = step(grid)
    
    loss = lossFunc(grid)
    if verbose:
        print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    for i in range(100):
        trainingLoop(verbose=(i % 10 == 0))
    
    print(f"\nFinal weights:")
    print(f"  Shape: {conv.weight.shape}")
    print(f"  Weights: {conv.weight.data}")
    print(f"  Bias: {conv.bias.data}")