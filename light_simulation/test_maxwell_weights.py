"""
Filename: train_maxwell_documented.py

Purpose: Evaluate and visualize how the trained NCA generalizes beyond its
training distribution by comparing its predictions to the explicit
Maxwell solver. Runs tests across longer time horizons and larger spatial
domains, logs errors, and produces plots/animations for inspection.

This script assumes that a model weights file named
`physics_light_sim.pth` exists in the working directory (produced by
`train_maxwell.py` or its documented variant). The NCA uses a 3→3
convolution (Ez, Hx, Hy) and enforces Perfect Electric Conductor (PEC)
solids during each step.

Key Components:
  - `physics_frame`: reference physics engine step (same as training)
  - `nca_step`: apply NCA update and re-enforce PEC boundary
  - `test_generalization`: run parallel NCA and physics simulations,
    compute mean squared error, and collect history for visualization
  - Two predefined tests: 32×32 grid for long time (time generalization) and
    128×128 grid for moderate time (space generalization)
  - Uses Matplotlib to generate plots and save a summary figure

Output:
  - `generalization_test.png`: visualization of fields and error curves
  - Console logs summarizing final and maximum errors

Device selection, model loading, and plotting tools are included for
reproducible evaluation in notebooks or scripts.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# device selection
from IPython.display import HTML

# choose GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load trained convolutional model (no bias) and set to eval mode
conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1), bias=False).to(device)
conv.load_state_dict(torch.load('physics_light_sim.pth'))
conv.eval()


def physics_frame(grid, solid):
    """Compute one timestep of the ground-truth Maxwell update.

    This function is identical to the one used during training and applies
    the TE‑mode FDTD update with Mur absorbing boundaries and PEC enforcement
    inside the supplied rectangular solid. It is used as a reference to
    measure NCA accuracy.

    Args:
        grid (torch.Tensor): state tensor shape (1, 3, H, W) containing Ez,
            Hx and Hy channels.
        solid (tuple): (x, y, width, height) of perfect conductor region.

    Returns:
        torch.Tensor: new state after one physics timestep.
    """
    Ez = grid[0, 0]
    Hx = grid[0, 1]
    Hy = grid[0, 2]

    c = 0.5

    Hx_new = Hx - c * (torch.roll(Ez, -1, dims=0) - Ez)
    Hy_new = Hy + c * (torch.roll(Ez, -1, dims=1) - Ez)

    Ez_new = Ez + c * (
        (Hy_new - torch.roll(Hy_new, 1, dims=1)) -
        (Hx_new - torch.roll(Hx_new, 1, dims=0))
    )

    Ez_new[0, :] = Ez[1, :]
    Ez_new[-1, :] = Ez[-2, :]
    Ez_new[:, 0] = Ez[:, 1]
    Ez_new[:, -1] = Ez[:, -2]

    x1, y1, w, h = solid
    Ez_new[y1:y1+h, x1:x1+w] = 0.0

    new_grid = torch.stack([Ez_new, Hx_new, Hy_new]).unsqueeze(0)
    return new_grid


def nca_step(grid, solid):
    """Compute one timestep using the trained NCA and enforce PEC.

    The update is simply `grid + conv(grid)`; no activation is applied. After
    computing the new state the Ez field is zeroed inside the solid region to
    mimic the boundary conditions used during training.

    Args:
        grid (torch.Tensor): current state (1, 3, H, W).
        solid (tuple): PEC rectangle (x, y, w, h).

    Returns:
        torch.Tensor: updated state.
    """
    grid = grid + conv(grid)
    x1, y1, w, h = solid
    grid[0, 0, y1:y1+h, x1:x1+w] = 0.0
    return grid


def test_generalization(H, W, steps, solid):
    """Run side-by-side NCA and physics simulations to collect error data.

    The two grids are initialized identically with a unit impulse at the
    center. Both are stepped for the requested number of timesteps; at each
    step the Ez channel history is recorded along with the MSE error between
    the full 3-channel states. The returned histories can be used for plotting
    or animation.

    Args:
        H (int): grid height.
        W (int): grid width.
        steps (int): number of timesteps to simulate.
        solid (tuple): PEC rectangle (x, y, w, h).

    Returns:
        tuple: (nca_history, maxwell_history, error_history), where each
            history is a list of Ez field snapshots or error scalars.
    """
    grid_nca = torch.zeros(1, 3, H, W).to(device)
    grid_maxwell = torch.zeros(1, 3, H, W).to(device)

    # place a source at the center
    grid_nca[0, 0, H//2, W//2] = 1.0
    grid_maxwell[0, 0, H//2, W//2] = 1.0

    nca_history = []
    maxwell_history = []
    error_history = []

    with torch.no_grad():
        for t in range(steps):
            nca_history.append(grid_nca[0, 0].cpu().numpy())
            maxwell_history.append(grid_maxwell[0, 0].cpu().numpy())
            error = ((grid_nca - grid_maxwell) ** 2).mean().item()
            error_history.append(error)
            grid_nca = nca_step(grid_nca, solid)
            grid_maxwell = physics_frame(grid_maxwell, solid)

    return nca_history, maxwell_history, error_history


if __name__ == "__main__":
    # test 1: reproduce training size but run longer in time
    print("Test 1: 32x32 grid, 200 steps (2x training time)")
    solid = (8, 8, 4, 4)
    nca_hist, max_hist, err_hist = test_generalization(32, 32, 200, solid)
    print(f"Final error: {err_hist[-1]:.6f}")
    print(f"Max error: {max(err_hist):.6f}")

    # test 2: much larger grid for spatial generalization
    print("\nTest 2: 128x128 grid, 50 steps (4x training size)")
    solid = (32, 32, 8, 8)
    nca_hist2, max_hist2, err_hist2 = test_generalization(128, 128, 50, solid)
    print(f"Final error: {err_hist2[-1]:.6f}")
    print(f"Max error: {max(err_hist2):.6f}")

    # create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # time generalization
    axes[0, 0].imshow(nca_hist[0], cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 0].set_title('NCA t=0')
    axes[0, 1].imshow(nca_hist[100], cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('NCA t=100')
    axes[0, 2].plot(err_hist)
    axes[0, 2].set_title('Error over time')
    axes[0, 2].set_xlabel('Timestep')
    axes[0, 2].set_ylabel('MSE')

    # space generalization
    axes[1, 0].imshow(nca_hist2[0], cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('NCA t=0 (128x128)')
    axes[1, 1].imshow(nca_hist2[49], cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('NCA t=49 (128x128)')
    axes[1, 2].plot(err_hist2)
    axes[1, 2].set_title('Error over space')
    axes[1, 2].set_xlabel('Timestep')
    axes[1, 2].set_ylabel('MSE')

    plt.tight_layout()
    plt.savefig('generalization_test.png', dpi=150)
    plt.show()