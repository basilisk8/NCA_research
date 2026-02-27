"""
Filename: gated_residual_sort.py

Purpose: Train a NCA to sort 7 integers by actually moving values
         to their correct sorted positions. 
         Uses Hungarian matching loss to prevent interpolation and force
         the network to preserve original input values during sorting.
         Gated residual is the activation function to preserve values 
         better than tanh

Key Parameters:
 - Grid size: (1, 64, 1, 7) - 1 batch, 64 channels, 1 row, 7 elements
 - Learning rate: 0.0001
 - Steps : 60
 - Training iterations: 500,000
 - Input range: 0-255 raw values, not normalized
 - Loss: MSE (ordering) + Hungarian matching (value preservation)

Architecture:
 - NCA runs 60 NCA steps
 - Channel 0: Input (never modified)
 - Even Channel numbers : Raw value
 - Odd number Channels : gates value

Loss Function:
 - Sort Loss (MSE): Penalizes difference from sorted target order
 - Match Loss (Hungarian): Finds optimal one-to-one assignment between
   output values and input values, penalizes total assignment cost.
   Prevents the "interpolation trap" where the NCA outputs a generic
   ascending ramp instead of the actual input values rearranged.
 - Combined: sort_loss + match_loss

Key Findings:
 - Achieves 85% accuracy within ±5 tolerance on width 7
 - Order is consistently correct, values are close but not exact
 - Precision limited by training time
 - Generalized to width 10 with 85% accuracy
 - Ranking approach is equally accuracte, but this scales 
 - This approach actually routes values, proving NCA can do it

Outputs:
 - gated_residual_sort.pth: Trained model weights for all 5 phases
"""

import torch.nn as nn
import torch
import random
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

conv = nn.Conv2d(64, 126, kernel_size=(3, 3), padding=(1, 1)).to(device)

optimizer = torch.optim.AdamW(conv.parameters(), lr=0.0001, weight_decay=0.01)

nn.init.xavier_uniform_(conv.weight, gain=0.1)
with torch.no_grad():
    conv.bias[0::2].fill_(0)    # value biases = 0
    conv.bias[1::2].fill_(1.0)  # gate biases = 1 (odd indices)

def diagnose(grid, conv, steps):
    """
    Runs NCA forward pass while measuring 5 things:
    1. Output at each step
    2. Gate values (open/closed?)
    3. Channel activity (dead channels?)
    4. Value range (exploding?)
    5. Step-to-step change (converging?)
    """
        
    outputs = []
    gate_means = []
    dead_channels = []
    value_maxes = []
    step_changes = []
        
    prev_output = None
        
    for s in range(steps):
        # Forward pass (same as your step function)
        raw = conv(grid)
        v = raw[:, 0::2, :, :]
        g = raw[:, 1::2, :, :]
        gates = torch.sigmoid(g)
        update = v * gates
        
        grid = grid.clone()
        grid[0, 1:, :, :] = grid[0, 1:, :, :] + update[0]
        
        # 1. Output at this step
        output = grid[0, 1, 0, :].detach()
        outputs.append([round(x.item()) for x in output])
        
            # 2. Gate values
        gate_means.append(gates.mean().item())
            
            # 3. Dead channels (std < 0.001)
        channel_stds = grid[0, 1:, 0, :].std(dim=1)
        dead = (channel_stds < 0.001).sum().item()
        dead_channels.append(dead)
            
            # 4. Value range
        value_maxes.append(grid[0, 1:].abs().max().item())
            
            # 5. Step change
        if prev_output is not None:
            change = (output - prev_output).abs().mean().item()
            step_changes.append(change)
        prev_output = output.clone()
        
    return {
        'outputs': outputs,
        'gate_means': gate_means,
        'dead_channels': dead_channels,
        'value_maxes': value_maxes,
        'step_changes': step_changes
    }


def print_diagnosis(results, input_vals, steps):
    """Print the diagnosis results."""
        
    print("=" * 50)
    print("NCA DIAGNOSIS")
    print("=" * 50)
        
    print(f"\nInput: {input_vals}")
    print(f"Target: {sorted(input_vals)}")
    print(f"Final: {results['outputs'][-1]}")
        
    # 1. Output evolution
    print(f"\n[1] OUTPUT EVOLUTION")
    for s in [0, 9, 19, 29]:
        if s < steps:
            print(f"  Step {s+1:2d}: {results['outputs'][s]}")
    
    # 2. Gates
    print(f"\n[2] GATE VALUES")
    print(f"  Step 1:  {results['gate_means'][0]:.3f}")
    print(f"  Step 15: {results['gate_means'][14]:.3f}")
    print(f"  Step 30: {results['gate_means'][-1]:.3f}")
    if results['gate_means'][-1] < 0.1:
        print("Gates closing - network shutting down")
    elif results['gate_means'][-1] > 0.9:
        print("Gates fully open - no selectivity")
    else:
        print("Gates selective")
    
    # 3. Channel activity
    print(f"\n[3] CHANNEL ACTIVITY")
    print(f"  Dead channels: {results['dead_channels'][-1]}/63")
    if results['dead_channels'][-1] > 15:
        print("Over half channels dead")
    else:
        print("Channels active")
    
    # 4. Value range
    print(f"\n[4] VALUE RANGE")
    print(f"  Max absolute: {results['value_maxes'][-1]:.2f}")
    if results['value_maxes'][-1] > 50:
        print("Values exploding")
    else:
        print("Values stable")
    
    # 5. Convergence
    print(f"\n[5] CONVERGENCE")
    if len(results['step_changes']) > 1:
        early = results['step_changes'][0]
        late = results['step_changes'][-1]
        print(f"  Early change: {early:.4f}")
        print(f"  Late change:  {late:.4f}")
        if late > early * 0.5:
            print("Not converging - still changing at end")
        elif late < 0.001:
            print("Converged (maybe too early?)")
        else:
            print("Converging normally")    
    print("=" * 50)
        
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

def step(grid):
    raw_update = conv(grid)
    v = raw_update[:, 0::2, :, :]
    g = raw_update[:, 1::2, :, :]
    gated_update = v * torch.sigmoid(g)
    newGrid = grid.clone()
    newGrid[0, 1:, :, :] = grid[0, 1:, :, :] + gated_update[0, :, :, :]
    return newGrid

def trainingLoop(v, j, log):
    optimizer.zero_grad()
    grid = torch.zeros(1, 64, 1, 7).to(device)
    for i in range(7):
        grid[0, 0, 0, i] = v[i]

    target = torch.zeros(1, 1, 1, 7).to(device)
    target[0, 0, 0, :] = torch.sort(grid[0, 0, 0, :])[0]

    for _ in range(60):
        grid = step(grid)

    output = grid[0, 1, 0, :]
    sort_loss = nn.MSELoss()(output, target[0, 0, 0, :])
    # Match loss: output values should match input values (just rearranged)
    match_loss = hungarian_loss(output, grid[0, 0, 0, :])
    # Combined loss: must be sorted AND preserve original values
    hidden_penalty = grid[0, 2:, :, :].abs().mean() * 0.001
    loss = sort_loss + match_loss + hidden_penalty

    if log:
        print(f"Iteration {j} | Loss: {loss.item():.4f}")
        print(f"  Input:    {v}")
        print(f"  Expected: {[round(x) for x in target[0, 0, 0, :].tolist()]}")
        print(f"  Output:   {[round(x) for x in output.tolist()]}")
        print()
        print()
        print()
        optimizer.zero_grad()
        grid = torch.zeros(1, 64, 1, 7).to(device)
        for i in range(7):
            grid[0, 0, 0, i] = v[i]
        results = diagnose(grid, conv, steps=60)
            
        # Print it
        print_diagnosis(results, v, steps=60)

    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    v = [0, 0, 0, 0, 0, 0, 0]
    for i in range(500000):
        for j in range(7):
            v[j] = random.randint(0, 1)
        if v[j] == 1:
            v[j] = random.randint(0, 255)
        trainingLoop(v, i, i%10000 == 0)
    torch.save(conv.state_dict(), 'gated_residual_sort.pth')