import torch
import random
import torch.nn as nn
from collections import defaultdict

EMPTY = 1
X = 0
O = -1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 32 base channels, 256 hidden channels, 4 conv layers
conv1 = nn.Conv2d(32, 256, kernel_size=3, padding=1).to(device)  # 3x3 perception
conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device)  # 3x3 perception
conv3 = nn.Conv2d(256, 256, kernel_size=1).to(device)  # 1x1 update
conv4 = nn.Conv2d(256, 31, kernel_size=1).to(device)  # 1x1 update (32-1 for channel 0)

optimizer = torch.optim.AdamW(
    list(conv1.parameters()) + 
    list(conv2.parameters()) + 
    list(conv3.parameters()) + 
    list(conv4.parameters()), 
    lr=0.0001
)

def check_winner_flat(g):
    for i in range(3):
        if g[i*3] == g[i*3+1] == g[i*3+2] != EMPTY: return g[i*3]
        if g[i] == g[i+3] == g[i+6] != EMPTY: return g[i]
    if g[0] == g[4] == g[8] != EMPTY: return g[0]
    if g[2] == g[4] == g[6] != EMPTY: return g[2]
    return None

def minimax_flat(g, current_player, maximizing_player):
    winner = check_winner_flat(g)
    if winner == maximizing_player: return 1
    if winner is not None: return -1
    if all(c != EMPTY for c in g): return 0
    opponent = O if current_player == X else X
    if current_player == maximizing_player:
        best = -2
        for i in range(9):
            if g[i] == EMPTY:
                g[i] = current_player
                score = minimax_flat(g, opponent, maximizing_player)
                g[i] = EMPTY
                best = max(best, score)
        return best
    else:
        best = 2
        for i in range(9):
            if g[i] == EMPTY:
                g[i] = current_player
                score = minimax_flat(g, opponent, maximizing_player)
                g[i] = EMPTY
                best = min(best, score)
        return best

def best_moves_flat(g, player):
    best_score, best_moves = -2, []
    opponent = O if player == X else X
    for i in range(9):
        if g[i] == EMPTY:
            g[i] = player
            score = minimax_flat(g, opponent, player)
            g[i] = EMPTY
            if score > best_score:
                best_score, best_moves = score, [i]
            elif score == best_score:
                best_moves.append(i)
    return best_moves

def generate_dataset():
    print("Precomputing dataset...")
    seen = {}

    def recurse(g, depth):
        winner = check_winner_flat(g)
        if winner is not None or all(c != EMPTY for c in g):
            return

        if depth % 2 == 0:
            key = tuple(g)
            if key not in seen:
                moves = best_moves_flat(list(g), X)
                seen[key] = moves
        else:
            flipped = tuple(X if c == O else (O if c == X else EMPTY) for c in g)
            if flipped not in seen:
                moves = best_moves_flat(list(flipped), X)
                seen[flipped] = moves

        current_player = X if depth % 2 == 0 else O
        for i in range(9):
            if g[i] == EMPTY:
                g[i] = current_player
                recurse(g, depth + 1)
                g[i] = EMPTY

    recurse([EMPTY] * 9, 0)

    dataset = list(seen.items())
    print(f"Total unique positions: {len(dataset)}")
    return dataset

dataset = generate_dataset()

def sample_position():
    return random.choice(dataset)

def step(grid):
    """
    NCA step with 4 conv layers.
    Channel 0 is input only, channels 1-31 are updated.
    """
    x = torch.tanh(conv1(grid))
    x = torch.tanh(conv2(x))
    x = torch.tanh(conv3(x))
    update = torch.tanh(conv4(x))
    
    new_grid = grid.clone()
    new_grid[0, 1:, :, :] = grid[0, 1:, :, :] + update[0]
    return new_grid

def loss_func(final_grid, target, board):
    output = final_grid[0, 1, :, :].flatten()
    targets = target[0, 0, :, :].flatten()
    bce_loss = nn.BCEWithLogitsLoss()(output, targets)
    empty_mask = (board[0, 0].flatten() == EMPTY).float().to(device)
    optimal_mask = targets * empty_mask
    num_optimal = optimal_mask.sum().item()
    if num_optimal > 0:
        optimal_indices = (optimal_mask == 1).nonzero(as_tuple=True)[0]
        non_optimal_empty_mask = (empty_mask == 1) & (targets == 0)
        if non_optimal_empty_mask.any():
            non_optimal_indices = non_optimal_empty_mask.nonzero(as_tuple=True)[0]
            min_optimal_score = output[optimal_indices].min()
            max_wrong_score = output[non_optimal_indices].max()
            margin_loss = torch.relu(max_wrong_score - min_optimal_score + 2.0)
            return bce_loss + 2.0 * margin_loss
    return bce_loss

def training_loop(i):
    board_flat, move_indices = sample_position()
    num_empty = sum(1 for c in board_flat if c == EMPTY)

    board = torch.full((1, 32, 3, 3), EMPTY, dtype=torch.float32).to(device)  # 16→32 channels
    for idx, val in enumerate(board_flat):
        board[0, 0, idx // 3, idx % 3] = val
    input_board = board[0, 0, :, :].clone()
    initial_board = board.clone()
    target = torch.zeros((1, 1, 3, 3), dtype=torch.float32).to(device)
    for move_idx in move_indices:
        target[0, 0, move_idx // 3, move_idx % 3] = 1.0

    for _ in range(30):
        board = step(board)

    loss = loss_func(board, target, initial_board)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        output = board[0, 1, :, :]
        print(f"\nIter {i} | Loss: {loss.item():.4f} | Empty squares: {num_empty}")
        print(f"Input:\n{input_board.cpu().numpy()}")
        print(f"Target (multi-hot):\n{target[0, 0].cpu().numpy()}")
        print(f"Output:\n{output.detach().cpu().numpy()}")
        print(f"Num optimal moves: {len(move_indices)}")
        empty_mask = (input_board.flatten() == EMPTY).float()
        optimal_mask = target[0, 0].flatten() * empty_mask
        if optimal_mask.sum() > 0:
            opt_idx = (optimal_mask == 1).nonzero(as_tuple=True)[0]
            non_opt_mask = (empty_mask == 1) & (target[0, 0].flatten() == 0)
            if non_opt_mask.any():
                non_opt_idx = non_opt_mask.nonzero(as_tuple=True)[0]
                min_opt = output.flatten()[opt_idx].min().item()
                max_wrong = output.flatten()[non_opt_idx].max().item()
                print(f"Margin: {min_opt - max_wrong:.4f} (should be > 2.0)")
        torch.save({
            'iter': i,
            'conv1': conv1.state_dict(),
            'conv2': conv2.state_dict(),
            'conv3': conv3.state_dict(),
            'conv4': conv4.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'tictactoe_nca.pth')

    return loss.item()

if __name__ == "__main__":
    for i in range(500000):
        training_loop(i)
