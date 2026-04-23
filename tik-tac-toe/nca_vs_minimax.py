import torch
import torch.nn as nn

EMPTY = 1
X = 0
O = -1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# EXACT SAME ARCHITECTURE
conv1 = nn.Conv2d(32, 256, kernel_size=3, padding=1).to(device)
conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device)
conv3 = nn.Conv2d(256, 256, kernel_size=1).to(device)
conv4 = nn.Conv2d(256, 31, kernel_size=1).to(device)

checkpoint = torch.load('tictactoe_nca (11).pth', map_location=device)
conv1.load_state_dict(checkpoint['conv1'])
conv2.load_state_dict(checkpoint['conv2'])
conv3.load_state_dict(checkpoint['conv3'])
conv4.load_state_dict(checkpoint['conv4'])

conv1.eval()
conv2.eval()
conv3.eval()
conv4.eval()

def step(grid):
    x = torch.tanh(conv1(grid))
    x = torch.tanh(conv2(x))
    x = torch.tanh(conv3(x))
    update = torch.tanh(conv4(x))

    new_grid = grid.clone()
    new_grid[0, 1:, :, :] = grid[0, 1:, :, :] + update[0]
    return new_grid

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

class PureNCAPlayer:
    """NCA always plays as X perspective"""
    def __init__(self):
        pass

    def get_move(self, board_flat, nca_is_x):
        """
        Get move from NCA with perspective shift.

        Args:
            board_flat: 9-element list with EMPTY, X, O
            nca_is_x: True if NCA is playing as X, False if playing as O

        Returns:
            Move index (0-8)
        """
        # If NCA is O, flip perspective so NCA sees itself as X
        if not nca_is_x:
            flipped = [X if c == O else (O if c == X else EMPTY) for c in board_flat]
        else:
            flipped = board_flat[:]

        # Create grid
        grid = torch.full((1, 32, 3, 3), EMPTY, dtype=torch.float32).to(device)
        for idx, val in enumerate(flipped):
            grid[0, 0, idx // 3, idx % 3] = val

        # Run NCA for 30 steps
        with torch.no_grad():
            for _ in range(30):
                grid = step(grid)

        # Extract output from channel 1
        output = grid[0, 1, :, :].cpu().numpy().flatten()

        # Find max value among empty squares
        max_val = -float('inf')
        best_move = None

        for i in range(9):
            if flipped[i] == EMPTY and output[i] > max_val:
                max_val = output[i]
                best_move = i

        return best_move

class MinimaxPlayer:
    """Perfect minimax player"""
    def __init__(self):
        self.systematic_first_moves = list(range(9))
        self.game_count = 0

    def get_move(self, board_flat, minimax_is_x):
        # If this is the first move and minimax is X, use systematic opening
        if minimax_is_x and all(c == EMPTY for c in board_flat):
            move = self.systematic_first_moves[self.game_count % 9]
            return move

        # Otherwise use minimax
        player = X if minimax_is_x else O
        moves = best_moves_flat(board_flat[:], player)
        return moves[0] if moves else None

    def next_game(self):
        self.game_count += 1

def play_game(nca_player, minimax_player, nca_is_x, verbose=False):
    """
    Play one game.

    Returns:
        X if X wins, O if O wins, None if draw
    """
    board = [EMPTY] * 9
    current_player = X

    for turn in range(9):
        if verbose:
            print(f"\nTurn {turn + 1}, Player {'X' if current_player == X else 'O'}:")
            for i in range(3):
                row = [board[i*3], board[i*3+1], board[i*3+2]]
                print([{X: 'X', O: 'O', EMPTY: '.'}.get(x) for x in row])

        # Determine who plays
        if (current_player == X and nca_is_x) or (current_player == O and not nca_is_x):
            # NCA's turn
            move = nca_player.get_move(board, nca_is_x)
        else:
            # Minimax's turn
            minimax_is_x = (current_player == X)
            move = minimax_player.get_move(board, minimax_is_x)

        if move is None:
            break

        board[move] = current_player

        # Check winner
        winner = check_winner_flat(board)
        if winner is not None:
            if verbose:
                print(f"\nPlayer {'X' if winner == X else 'O'} wins!")
            return winner

        current_player = O if current_player == X else X

    if verbose:
        print("\nDraw!")
    return None

def test_nca_vs_minimax():
    print("\n" + "="*50)
    print("NCA vs MINIMAX TEST (10 games)")
    print("="*50)

    nca = PureNCAPlayer()
    minimax = MinimaxPlayer()

    results = {
        'nca_wins': 0,
        'minimax_wins': 0,
        'draws': 0
    }

    # Games 1-9: Minimax is X (systematic openings), NCA is O
    print("\n--- Games 1-9: Minimax (X) vs NCA (O) ---")
    for game_num in range(9):
        print(f"\nGame {game_num + 1}: Minimax opens with position {game_num}")
        winner = play_game(nca, minimax, nca_is_x=False, verbose=True)

        if winner == X:
            results['minimax_wins'] += 1
            print(f"Result: Minimax wins")
        elif winner == O:
            results['nca_wins'] += 1
            print(f"Result: NCA wins")
        else:
            results['draws'] += 1
            print(f"Result: Draw")

        minimax.next_game()

    # Game 10: NCA is X, Minimax is O
    print("\n--- Game 10: NCA (X) vs Minimax (O) ---")
    winner = play_game(nca, minimax, nca_is_x=True, verbose=True)

    if winner == X:
        results['nca_wins'] += 1
        print(f"Result: NCA wins")
    elif winner == O:
        results['minimax_wins'] += 1
        print(f"Result: Minimax wins")
    else:
        results['draws'] += 1
        print(f"Result: Draw")

    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"NCA wins: {results['nca_wins']}/10")
    print(f"Minimax wins: {results['minimax_wins']}/10")
    print(f"Draws: {results['draws']}/10")
    print(f"\nNCA win rate: {100 * results['nca_wins'] / 10:.1f}%")
    print(f"NCA draw rate: {100 * results['draws'] / 10:.1f}%")
    print(f"NCA loss rate: {100 * results['minimax_wins'] / 10:.1f}%")

    if results['minimax_wins'] == 0:
        print("\nNCA NEVER LOSES! (Draws or wins only)")
    elif results['nca_wins'] + results['draws'] >= 8:
        print("\nNCA plays strong (80%+ draw/win rate)")
    else:
        print("\nNCA needs more training")

if __name__ == "__main__":
    test_nca_vs_minimax()