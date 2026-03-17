import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conv1 = nn.Conv2d(25, 128, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(128, 128, kernel_size=1).to(device)
conv3 = nn.Conv2d(128, 21, kernel_size=1).to(device)
seed_embed = nn.Embedding(5, 4).to(device)

checkpoint = torch.load('stacked_conv_weights.pth', map_location=device)
conv1.load_state_dict(checkpoint['conv1'])
conv2.load_state_dict(checkpoint['conv2'])
conv3.load_state_dict(checkpoint['conv3'])
seed_embed.load_state_dict(checkpoint['seed_embed'])

targets = []
t = torch.zeros(40, 40, device=device)
for y in range(40):
    for x in range(40):
        if (x - 20) ** 2 + (y - 20) ** 2 <= 10 ** 2:
            t[y, x] = 1.0
targets.append(t)
t = torch.zeros(40, 40, device=device)
t[10:30, 10:30] = 1.0
targets.append(t)
t = torch.zeros(40, 40, device=device)
t[15:25, 5:35] = 1.0
t[5:35, 15:25] = 1.0
targets.append(t)
t = torch.zeros(40, 40, device=device)
for y in range(20, 35):
    width = y - 20
    t[y, 20-width:20+width+1] = 1.0
targets.append(t)
t = torch.zeros(40, 40, device=device)
t[18:22, 5:35] = 1.0
targets.append(t)

def step(grid):
    x = torch.relu(conv1(grid))
    x = torch.relu(conv2(x))
    update = conv3(x)
    mask = (torch.rand(1, 1, 40, 40, device=device) < 0.5)
    new_grid = grid.clone()
    new_grid[:, 0:21] = grid[:, 0:21] + update * mask * 0.1
    return new_grid

shape_names = ['circle', 'square', 'plus', 'triangle', 'line']
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

print("=== Accuracy ===")
for si in range(5):
    grid = torch.zeros(1, 25, 40, 40, device=device)
    seed = seed_embed(torch.tensor(si, device=device))
    grid[0, 21:25, 20, 20] = seed

    with torch.no_grad():
        for s in range(80):
            grid = step(grid)

    output = torch.sigmoid(grid[0, 0]).cpu()
    target = targets[si].cpu()
    binary = (output > 0.5).float()
    accuracy = (binary == target).float().mean()
    print(f"{shape_names[si]}: {accuracy:.4f}")

    axes[0][si].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0][si].set_title(f'Target: {shape_names[si]}')
    axes[0][si].axis('off')

    axes[1][si].imshow(output, cmap='gray', vmin=0, vmax=1)
    axes[1][si].set_title(f'{shape_names[si]}: {accuracy:.4f}')
    axes[1][si].axis('off')

plt.suptitle('Stacked Conv NCA: One Set of Weights, Five Shapes')
plt.tight_layout()
plt.savefig('stacked_conv_result.png', dpi=150)
plt.show()