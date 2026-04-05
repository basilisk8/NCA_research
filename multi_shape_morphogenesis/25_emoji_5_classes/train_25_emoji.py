import torch
import torch.nn.functional as F
import random
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

state_channels = 128
class_dim = 32
variant_dim = 32
total_channels = state_channels + class_dim + variant_dim

conv1 = nn.Conv2d(192, 256, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(256, 256, kernel_size=1).to(device)
conv3 = nn.Conv2d(256, 128, kernel_size=1).to(device)

nn.init.zeros_(conv3.weight)
nn.init.zeros_(conv3.bias)

class_embed = nn.Embedding(5, 32).to(device)
variant_embed = nn.Embedding(5, 32).to(device)

optimizer = torch.optim.AdamW(
    list(conv1.parameters()) + list(conv2.parameters()) +
    list(conv3.parameters()) + list(class_embed.parameters()) +
    list(variant_embed.parameters()),
    lr=0.0005, weight_decay=0.01
)

targets_nested = torch.load('emoji_targets.pt', map_location=device)
targets = [[emoji.to(device) for emoji in emoji_type] for emoji_type in targets_nested]

def step(grid):
    x = torch.relu(conv1(grid))
    x = torch.relu(conv2(x))
    update = conv3(x)
    
    # NO MASK
    new_grid = grid.clone()
    new_grid[:, 0:128] = grid[:, 0:128] + update * 0.1
    return new_grid

def lossFunc(finalGrid, target):
    output = finalGrid[0, 0:3, :, :]
    loss = nn.MSELoss()(output, target)
    return loss

def trainingLoop(log, class_idx, variant_idx, iteration):
    grid = torch.zeros(1, 192, 40, 40, device=device)
    
    seed = torch.cat([
        class_embed(torch.tensor(class_idx, device=device)),
        variant_embed(torch.tensor(variant_idx, device=device))
    ])
    grid[0, 128:192, 20, 20] = seed

    optimizer.zero_grad()

    steps = random.randint(128, 160)
    for _ in range(steps):
        grid = step(grid)

    loss = lossFunc(grid, targets[class_idx][variant_idx])

    if log:
        print(f"Iter {iteration} | C{class_idx}V{variant_idx} | loss: {loss.item():.6f}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(conv1.parameters()) + list(conv2.parameters()) + list(conv3.parameters()),
        1.0
    )
    optimizer.step()

def save_weights(iteration):
    torch.save({
        'iteration' : iteration,
        'conv1': conv1.state_dict(),
        'conv2': conv2.state_dict(),
        'conv3': conv3.state_dict(),
        'class_embed': class_embed.state_dict(),
        'variant_embed': variant_embed.state_dict(),
        'optimizer': optimizer.state_dict()
    }, 'factored_256ch_nomask.pth')

if __name__ == "__main__":
    for i in range(200000):
        c = random.randint(0, 4)
        v = random.randint(0, 4)
        if i % 1000 == 0:
            trainingLoop(True, c, v, i)
            save_weights(i)
        else:
            trainingLoop(False, c, v, i)

    print("Saved factored_256ch_nomask.pth")