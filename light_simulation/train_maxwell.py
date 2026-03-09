"""
Filename: physics_light_sim_documented.py

Purpose: Train a simple Neural Cellular Automaton (NCA) to emulate a
single-step Maxwell update in a 2‑D light simulation with obstacles.
The solver uses finite‑difference time‑domain (FDTD) style updates for
Ez, Hx and Hy fields and enforces Perfect Electric Conductor (PEC)
boundaries inside randomly placed rectangular solids. The NCA observes
a 3‑channel grid [Ez, Hx, Hy] and is trained to predict the next frame
of the physics engine using a small convolutional model.

Key Parameters:
  - Grid:   (1, 3, H, W) tensor – batch 1, 3 field channels, variable
            height/width. Ez=channel0, Hx=channel1, Hy=channel2.
  - Conv2d: 3→3 channels, kernel 3x3, padding 1. The model learns to
            update the three channels directly (no hidden state).
  - Optimizer: Adam, lr=1e-4
  - Training loops: 500k random examples of varying size & solid shapes
  - Physics step: explicit 1‑D Maxwell update with Courant number c=0.5
  - Boundary conditions: Mur absorbing boundaries + PEC interior blocks

Training Strategy:
  - A random light source is placed on an empty grid, and then a rectangular
    solid (with zero Ez) is placed so that the source lies outside it.
  - The ground-truth target is obtained by running the physics_frame twice
    (once to simulate the training horizon, then once to compute the
    next-step target). The solid region is zeroed in the target.
  - The NCA runs one step from the initial grid, and loss is computed on
    all three field channels. The Ez loss omits values inside the solid.

Expected Results:
  - The NCA should learn to approximate the Maxwell update over the training
    distribution of grid sizes and obstacle shapes, reducing the loss over
    time. This is mainly a demonstration of training NCAs on physics
    problems rather than achieving production‑quality solvers.

Outputs:
  - physics_light_sim.pth: saved model weights after training
"""

import torch
import torch.nn as nn
import random

# pick GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# simple 3→3 convolution; channel 0=Ez, 1=Hx, 2=Hy
conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1)).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.0001)


def physics_frame(grid, solid):
    """
    Perform a single Maxwell time step on the input grid.

    This implements a 2‑D TE‑mode update with a Courant number of 0.5. The
    input grid has shape (1, 3, H, W) representing the Ez, Hx and Hy fields.
    Periodic indexing (torch.roll) is used for derivatives; Mur absorbing
    boundary conditions are applied at the four edges, and the Ez field is
    forced to zero inside the rectangular solid (Perfect Electric Conductor).

    Args:
        grid (torch.Tensor): state tensor, shape (1, 3, H, W).
        solid (tuple): (x1, y1, width, height) specifying the PEC region.

    Returns:
        torch.Tensor: updated grid with the same shape as the input.
    """
    Ez = grid[0, 0]
    Hx = grid[0, 1]
    Hy = grid[0, 2]

    c = 0.5  # Courant number (CFL stable)

    # magnetics update
    Hx_new = Hx - c * (torch.roll(Ez, -1, dims=0) - Ez)
    Hy_new = Hy + c * (torch.roll(Ez, -1, dims=1) - Ez)

    # electric update
    Ez_new = Ez + c * (
        (Hy_new - torch.roll(Hy_new, 1, dims=1)) -
        (Hx_new - torch.roll(Hx_new, 1, dims=0))
    )

    # Mur absorbing boundaries
    Ez_new[0, :] = Ez[1, :]
    Ez_new[-1, :] = Ez[-2, :]
    Ez_new[:, 0] = Ez[:, 1]
    Ez_new[:, -1] = Ez[:, -2]

    # enforce perfectly conducting interior box (Ez=0)
    x1, y1, w, h = solid
    Ez_new[y1:y1+h, x1:x1+w] = 0.0

    # rebuild composite tensor
    new_grid = torch.stack([Ez_new, Hx_new, Hy_new]).unsqueeze(0)
    return new_grid


def step(grid):
    """
    Single NCA update step.

    The convolution is applied to all three channels and added to the input
    state. There is no activation function; the network is free to propose
    additive changes to the fields.

    Args:
        grid (torch.Tensor): current state tensor of shape (1, 3, H, W).

    Returns:
        torch.Tensor: new state after applying the learned update.
    """
    update = conv(grid)
    return grid + update


def trainingLoop(num, width, height, time, iteration):
    """
    Execute one training example and backpropagate the loss.

    A random light source and solid block are sampled; the physics engine is
    stepped `time` times to produce the starting state and once more to
    generate the target. The NCA is run for a single step, then losses are
    computed on all channels. The Ez loss ignores the solid region because
    the field is forced to zero there.

    Args:
        num (int): log flag (1 prints progress, 0 is silent).
        width (int): grid width for this example.
        height (int): grid height for this example.
        time (int): number of physics steps to run before taking the target.
        iteration (int): global iteration count (for logging).
    """
    # initialize a blank grid and place a unit impulse source
    grid = torch.zeros(1, 3, height, width).to(device)
    light_x = random.randint(0, width - 1)
    light_y = random.randint(0, height - 1)
    grid[0, 0, light_y, light_x] = 1.0

    # choose a random solid that does not contain the source
    while True:
        solid_x1 = random.randint(0, width - 8)
        solid_y1 = random.randint(0, height - 8)
        solid_width = random.randint(2, 8)
        solid_height = random.randint(2, 8)
        inside_x = solid_x1 <= light_x < solid_x1 + solid_width
        inside_y = solid_y1 <= light_y < solid_y1 + solid_height
        if not (inside_x and inside_y):
            break

    # run the physics forward to produce the starting grid
    for _ in range(time):
        grid = physics_frame(grid, (solid_x1, solid_y1, solid_width, solid_height))

    # now compute the target: one more physics step with solid enforced
    target = physics_frame(grid, (solid_x1, solid_y1, solid_width, solid_height))
    target[0, 0, solid_y1:solid_y1+solid_height, solid_x1:solid_x1+solid_width] = 0.0

    # forward pass of the NCA
    optimizer.zero_grad()
    for _ in range(1):
        grid = step(grid)
    # enforce solid on NCA output as well
    grid[0, 0, solid_y1:solid_y1+solid_height, solid_x1:solid_x1+solid_width] = 0.0

    # mask to exclude solid region from Ez loss
    solid_mask = torch.zeros(height, width, dtype=torch.bool).to(device)
    solid_mask[solid_y1:solid_y1+solid_height, solid_x1:solid_x1+solid_width] = True

    # compute losses on each channel
    loss_ez = ((grid[0, 0][~solid_mask] - target[0, 0][~solid_mask]) ** 2).mean()
    loss_hx = ((grid[0, 1] - target[0, 1]) ** 2).mean()
    loss_hy = ((grid[0, 2] - target[0, 2]) ** 2).mean()
    loss = loss_ez + loss_hx + loss_hy

    if num == 1:
        print(f"Iter {iteration} | width={width} | loss={loss.item():.6f}")

    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # run training for a fixed number of examples
    for i in range(500000):
        width = random.randint(16, 32)
        height = random.randint(16, 32)
        time = random.randint(1, 5)
        if i % 5000 == 0:
            trainingLoop(1, width, height,  time, i)
        else:
            trainingLoop(0, width, height, time, i)

    # save the learned weights
    torch.save(conv.state_dict(), 'physics_light_sim.pth')
