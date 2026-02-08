import torch
import torch.nn as nn
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

grid = torch.zeros(1, 12, 6, 6)
grid = grid.to(device)  # ← ADD THIS

all_activations = []
def initialize_weights():
    for i in range(1,10):
        for j in range(1,5):
            for k in range(1,5):
                grid[0, i, j , k] = (random.random() - 0.5) * 0.2

def clear_activation(grid):
    grid[0, 10, :, :] = 0
    return grid

def step(grid):
    new_activation = grid[0, 10, :, :].clone()
    
    for i in range(1, 5):
        for j in range(1, 5):
            # Get 9 neighbor activations
            a0 = grid[0, 10, i-1, j-1]  # top_left
            a1 = grid[0, 10, i-1, j]    # top
            a2 = grid[0, 10, i-1, j+1]  # top_right
            a3 = grid[0, 10, i,   j-1]  # left
            a4 = grid[0, 10, i,   j]    # center
            a5 = grid[0, 10, i,   j+1]  # right
            a6 = grid[0, 10, i+1, j-1]  # bottom_left
            a7 = grid[0, 10, i+1, j]    # bottom
            a8 = grid[0, 10, i+1, j+1]  # bottom_right
            
            # Get 9 weights for this cell
            w0 = grid[0, 1, i, j]
            w1 = grid[0, 2, i, j]
            w2 = grid[0, 3, i, j]
            w3 = grid[0, 4, i, j]
            w4 = grid[0, 5, i, j]
            w5 = grid[0, 6, i, j]
            w6 = grid[0, 7, i, j]
            w7 = grid[0, 8, i, j]
            w8 = grid[0, 9, i, j]
            
            # Weighted sum + tanh
            total = a0*w0 + a1*w1 + a2*w2 + a3*w3 + a4*w4 + a5*w5 + a6*w6 + a7*w7 + a8*w8
            new_activation[i, j] = torch.tanh(total)
    
    grid[0, 10, :, :] = new_activation
    all_activations.append(new_activation)
    return grid

def correct(grid, all_activations, target, lr=0.01):
    output = grid[0, 10, 4, 1:5]
    target_tensor = torch.tensor(target, device=device).float()
    
    error = torch.abs(output - target_tensor).mean()
    reward = 1.0 - 2.0 * error
    
    for step_act in all_activations:
        for i in range(1, 5):
            for j in range(1, 5):
                my_act = step_act[i, j]
                
                neighbors = [
                    step_act[i-1, j-1],
                    step_act[i-1, j],
                    step_act[i-1, j+1],
                    step_act[i, j-1],
                    step_act[i, j],
                    step_act[i, j+1],
                    step_act[i+1, j-1],
                    step_act[i+1, j],
                    step_act[i+1, j+1],
                ]
                
                for k in range(9):
                    change = lr * my_act * neighbors[k] * reward
                    grid[0, k+1, i, j] += change
    
    return reward  # just for logging

if "__main__" == __name__:
    initialize_weights()
    
    for j in range(100000):
        all_activations = []
        grid = clear_activation(grid)
        grid[0, 10, 1, 1:5] = torch.tensor([0., 1., 1., 0.], device=device)
        target = [1, 1, 1, 1]
        
        for i in range(20):
            grid = step(grid)
        
        reward = correct(grid, all_activations, target)
        
        # ← RIGHT HERE
        if j % 1000 == 0:
            output = grid[0, 10, 4, 1:5]
            print(f"Step {j}, Reward: {reward:.3f}")
            print(f"  Output: {output.cpu().numpy()}")  # See actual values
            print(f"  Target: {target}")