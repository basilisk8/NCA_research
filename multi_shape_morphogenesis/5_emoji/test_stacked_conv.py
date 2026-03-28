import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conv1 = nn.Conv2d(80, 128, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(128, 128, kernel_size=1).to(device)
conv3 = nn.Conv2d(128, 64, kernel_size=1).to(device)
seed_embed = nn.Embedding(5, 16).to(device)

checkpoint = torch.load('stacked_conv_emoji_weights.pth', map_location=device)
conv1.load_state_dict(checkpoint['conv1'])
conv2.load_state_dict(checkpoint['conv2'])
conv3.load_state_dict(checkpoint['conv3'])
seed_embed.load_state_dict(checkpoint['seed_embed'])

targets = torch.load('emoji_targets.pt')
targets = [t.to(device) for t in targets]

def step(grid):
    x = torch.relu(conv1(grid))
    x = torch.relu(conv2(x))
    update = conv3(x)
    mask = (torch.rand(1, 1, 40, 40, device=device) < 0.5)
    new_grid = grid.clone()
    new_grid[:, 0:64] = grid[:, 0:64] + update * mask * 0.1
    return new_grid

emoji_names = ['emoji_0', 'emoji_1', 'emoji_2', 'emoji_3', 'emoji_4']
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

print("=== MSE Loss ===")
for ei in range(5):
    grid = torch.zeros(1, 80, 40, 40, device=device)
    seed = seed_embed(torch.tensor(ei, device=device))
    grid[0, 64:80, 20, 20] = seed

    with torch.no_grad():
        for s in range(80):
            grid = step(grid)

    output = grid[0, 0:3].cpu().clamp(0, 1)
    target = targets[ei].cpu()
    
    mse = F.mse_loss(output, target).item()
    print(emoji_names[ei] + ":", round(mse, 6))

    axes[0][ei].imshow(target.permute(1, 2, 0))
    axes[0][ei].set_title('Target: ' + emoji_names[ei])
    axes[0][ei].axis('off')

    axes[1][ei].imshow(output.permute(1, 2, 0))
    axes[1][ei].set_title('Grown: ' + emoji_names[ei] + ' (' + str(round(mse, 4)) + ')')
    axes[1][ei].axis('off')

plt.suptitle('RGB Emoji NCA: One Set of Weights, Five Emojis')
plt.tight_layout()
plt.savefig('emoji_result.png', dpi=150)
plt.show()