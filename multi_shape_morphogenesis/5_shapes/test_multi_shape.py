import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NCA(nn.Module):
    def __init__(self, ch=25, hidden=128):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        identity = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float32)
        self.register_buffer('filters', torch.stack([identity, sobel_x, sobel_y]))
        self.fc1 = nn.Conv2d(ch * 3, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, 21, 1)

    def perceive(self, grid):
        b, c, h, w = grid.shape
        perceived = []
        for f in self.filters:
            kernel = f.view(1,1,3,3).expand(c,-1,-1,-1)
            perceived.append(F.conv2d(grid, kernel, padding=1, groups=c))
        return torch.cat(perceived, dim=1)

    def forward(self, grid):
        perceived = self.perceive(grid)
        update = self.fc2(torch.relu(self.fc1(perceived)))
        mask = (torch.rand(1, 1, grid.shape[2], grid.shape[3], device=grid.device) < 0.5)
        new_grid = grid.clone()
        new_grid[:, 0:21] = grid[:, 0:21] + update * mask * 0.1
        return new_grid

nca = NCA(ch=25, hidden=128).to(device)
seed_embed = nn.Embedding(5, 4).to(device)

checkpoint = torch.load('nca_weights.pth', map_location=device)
nca.load_state_dict(checkpoint['nca_state_dict'])
seed_embed.load_state_dict(checkpoint['seed_embed_state_dict'])

targets = []
t = torch.zeros(40, 40, device=device)
for y in range(40):
    for x in range(40):
        if (x - 20)**2 + (y - 20)**2 <= 10**2:
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

shape_names = ['circle', 'square', 'plus', 'triangle', 'line']
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

print("=== Accuracy ===")
for si in range(5):
    grid = torch.zeros(1, 25, 40, 40, device=device)
    seed = seed_embed(torch.tensor(si, device=device))
    grid[0, 21:25, 20, 20] = seed

    with torch.no_grad():
        for s in range(80):
            grid = nca(grid)

    output = torch.sigmoid(grid[0, 0]).cpu()
    target = targets[si].cpu()
    binary = (output > 0.5).float()
    accuracy = (binary == target).float().mean()
    print(shape_names[si] + ":", round(accuracy.item(), 4))

    axes[0][si].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0][si].set_title('Target: ' + shape_names[si])
    axes[0][si].axis('off')

    axes[1][si].imshow(output, cmap='gray', vmin=0, vmax=1)
    axes[1][si].set_title('Grown: ' + shape_names[si] + ' (' + str(round(accuracy.item(), 4)) + ')')
    axes[1][si].axis('off')

plt.suptitle('Mordvintsev Architecture: One Set of Weights, Five Shapes')
plt.tight_layout()
plt.savefig('mordvintsev_result.png', dpi=150)
plt.show()