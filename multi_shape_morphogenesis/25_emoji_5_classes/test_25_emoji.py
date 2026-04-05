import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture - identical to training
conv1 = nn.Conv2d(192, 256, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(256, 256, kernel_size=1).to(device)
conv3 = nn.Conv2d(256, 128, kernel_size=1).to(device)
class_embed = nn.Embedding(5, 32).to(device)
variant_embed = nn.Embedding(5, 32).to(device)

checkpoint = torch.load('factored_256ch_nomask.pth', map_location=device)
conv1.load_state_dict(checkpoint['conv1'])
conv2.load_state_dict(checkpoint['conv2'])
conv3.load_state_dict(checkpoint['conv3'])
class_embed.load_state_dict(checkpoint['class_embed'])
variant_embed.load_state_dict(checkpoint['variant_embed'])

targets_nested = torch.load('emoji_targets.pt', map_location=device)
targets = [[emoji.to(device) for emoji in emoji_type] for emoji_type in targets_nested]

def step(grid):
    x = torch.relu(conv1(grid))
    x = torch.relu(conv2(x))
    update = conv3(x)
    new_grid = grid.clone()
    new_grid[:, 0:128] = grid[:, 0:128] + update * 0.1
    return new_grid

def run(class_idx, variant_idx):
    grid = torch.zeros(1, 192, 40, 40, device=device)
    seed = torch.cat([
        class_embed(torch.tensor(class_idx, device=device)),
        variant_embed(torch.tensor(variant_idx, device=device))
    ])
    grid[0, 128:192, 20, 20] = seed
    for _ in range(130):
        grid = step(grid)
    return grid

# Test all 25 combinations
fig, axes = plt.subplots(5, 10, figsize=(20, 10))

with torch.no_grad():
    for c in range(5):
        for v in range(5):
            grid = run(c, v)
            output = grid[0, 0:3].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            target = targets[c][v].permute(1, 2, 0).cpu().numpy()
            loss = nn.MSELoss()(grid[0, 0:3], targets[c][v]).item()
            print(f"C{c}V{v} | loss: {loss:.6f}")
            axes[c, v*2].imshow(target)
            axes[c, v*2].set_title(f'C{c}V{v} target', fontsize=6)
            axes[c, v*2].axis('off')

            axes[c, v*2+1].imshow(output)
            axes[c, v*2+1].set_title(f'loss {loss:.4f}', fontsize=6)
            axes[c, v*2+1].axis('off')

plt.tight_layout()
plt.savefig('test_results.png', dpi=150)
plt.show()