"""
Filename: diagnosis_function.py

Purpose: Run 1 itteration of the conv given a grid, and print detailed logs of everything
         going on in the hidden channels
"""

import torch
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
    
    # Modify this to match your step function
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
