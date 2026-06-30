"""
Optimized depth-limited minimax with alpha-beta pruning for 5x5 (win-4).
Improvements:
- Immediate threat detection (win/block)
- Move ordering for better pruning
- Transposition table for caching
- Faster evaluation
"""

import numpy as np

# Center control weights for 5x5
CENTER_WEIGHTS = np.array([
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 8, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1],
])

# Transposition table for caching positions
transposition_table = {}


def check_winner(board, board_size=5, win_length=4):
    """Check for winner. Returns +1, -1, 0 (draw), or None (ongoing)."""
    b2 = board.reshape(board_size, board_size)
    
    # Check all lines
    for i in range(board_size):
        for j in range(board_size - win_length + 1):
            # Horizontal
            s = b2[i, j:j + win_length].sum()
            if abs(s) == win_length:
                return int(np.sign(s))
            # Vertical
            s = b2[j:j + win_length, i].sum()
            if abs(s) == win_length:
                return int(np.sign(s))
    
    # Diagonals
    for i in range(board_size - win_length + 1):
        for j in range(board_size - win_length + 1):
            # Main diagonal
            s = sum(b2[i+k, j+k] for k in range(win_length))
            if abs(s) == win_length:
                return int(np.sign(s))
            # Anti-diagonal
            s = sum(b2[i+k, j+win_length-1-k] for k in range(win_length))
            if abs(s) == win_length:
                return int(np.sign(s))
    
    # Draw or ongoing
    return 0 if not (board == 0).any() else None


def get_all_lines(board, board_size, win_length):
    """Generator for all length-4 windows on the board."""
    b2 = board.reshape(board_size, board_size)
    
    for i in range(board_size):
        for j in range(board_size - win_length + 1):
            yield b2[i, j:j + win_length]  # Horizontal
            yield b2[j:j + win_length, i]  # Vertical
    
    for i in range(board_size - win_length + 1):
        for j in range(board_size - win_length + 1):
            yield np.array([b2[i+k, j+k] for k in range(win_length)])  # Diagonal
            yield np.array([b2[i+k, j+win_length-1-k] for k in range(win_length)])  # Anti-diag


def heuristic(board, player, board_size=5, win_length=4):
    """
    Evaluate position from player's perspective.
    Positive = good for player.
    """
    score = 0
    
    # Line-based scoring
    for line in get_all_lines(board, board_size, win_length):
        line = np.array(line)
        my = np.sum(line == player)
        opp = np.sum(line == -player)
        
        if my > 0 and opp > 0:
            continue  # Blocked line, no potential
        
        # Offense - building our lines
        if my == 1:
            score += 1
        elif my == 2:
            score += 10
        elif my == 3:
            score += 100
        
        # Defense - blocking opponent lines
        if opp == 1:
            score -= 2
        elif opp == 2:
            score -= 15
        elif opp == 3:
            score -= 200
    
    # Center control bonus
    b2 = board.reshape(board_size, board_size)
    score += np.sum(CENTER_WEIGHTS * (b2 == player)) * 2
    score -= np.sum(CENTER_WEIGHTS * (b2 == -player)) * 2
    
    return score


def order_moves(board, player, legal_moves, board_size=5, win_length=4):
    """
    Order moves for better alpha-beta pruning.
    Priority: winning moves > blocking moves > center > edges
    """
    scores = []
    center = board_size // 2
    
    for move in legal_moves:
        row, col = move // board_size, move % board_size
        score = 0
        
        # Base score: prefer center
        score -= (abs(row - center) + abs(col - center)) * 10
        
        # Check if this is a winning move
        board[move] = player
        if check_winner(board, board_size, win_length) == player:
            score += 10000  # Winning move - highest priority
        board[move] = 0
        
        # Check if this blocks opponent's win
        board[move] = -player
        if check_winner(board, board_size, win_length) == -player:
            score += 5000  # Must block - second highest priority
        board[move] = 0
        
        # Check if move creates a 3-in-a-row threat
        board[move] = player
        for line in get_lines_through_move(board, move, board_size, win_length):
            line = np.array(line)
            if np.sum(line == player) == 3 and np.sum(line == 0) == 1:
                score += 500  # Creates threat
        board[move] = 0
        
        scores.append((score, move))
    
    # Sort by score descending
    scores.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in scores]


def get_lines_through_move(board, move, board_size=5, win_length=4):
    """Get all lines that pass through a specific move."""
    b2 = board.reshape(board_size, board_size)
    row, col = move // board_size, move % board_size
    lines = []
    
    # Horizontal lines through this cell
    for start_col in range(max(0, col - win_length + 1), min(board_size - win_length + 1, col + 1)):
        lines.append(b2[row, start_col:start_col + win_length])
    
    # Vertical lines through this cell
    for start_row in range(max(0, row - win_length + 1), min(board_size - win_length + 1, row + 1)):
        lines.append(b2[start_row:start_row + win_length, col])
    
    # Main diagonal lines
    for k in range(win_length):
        start_row = row - k
        start_col = col - k
        if 0 <= start_row <= board_size - win_length and 0 <= start_col <= board_size - win_length:
            lines.append([b2[start_row + i, start_col + i] for i in range(win_length)])
    
    # Anti-diagonal lines
    for k in range(win_length):
        start_row = row - k
        start_col = col + k
        if 0 <= start_row <= board_size - win_length and win_length - 1 <= start_col < board_size:
            lines.append([b2[start_row + i, start_col - i] for i in range(win_length)])
    
    return lines


def minimax(board, player, depth, alpha, beta, board_size=5, win_length=4, max_depth=6):
    """
    Negamax with alpha-beta pruning and transposition table.
    Returns score from current player's perspective.
    """
    # Check transposition table
    key = (board.tobytes(), player, depth)
    if key in transposition_table:
        return transposition_table[key]
    
    # Terminal check
    winner = check_winner(board, board_size, win_length)
    if winner is not None:
        if winner == 0:
            return 0
        score = winner * player * 10000
        if score > 0:
            score += depth  # Prefer faster wins
        else:
            score -= depth  # Prefer slower losses
        return score
    
    # Depth limit - use heuristic
    if depth == 0:
        return heuristic(board, player, board_size, win_length)
    
    # Get and order legal moves
    legal = np.where(board == 0)[0]
    legal = order_moves(board, player, legal, board_size, win_length)
    
    best = -float('inf')
    for action in legal:
        board[action] = player
        score = -minimax(board, -player, depth - 1, -beta, -alpha, board_size, win_length, max_depth)
        board[action] = 0
        
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Pruning
    
    # Cache result
    transposition_table[key] = best
    return best


def get_best_move(board, player, depth=3, board_size=5, win_length=4):
    """
    Get best move with immediate threat detection and random tie-breaking.
    """
    legal = np.where(board == 0)[0]
    
    # CRITICAL: Check for immediate winning move
    for action in legal:
        board[action] = player
        if check_winner(board, board_size, win_length) == player:
            board[action] = 0
            return action  # Take the win!
        board[action] = 0
    
    # CRITICAL: Check if we must block opponent's win
    for action in legal:
        board[action] = -player
        if check_winner(board, board_size, win_length) == -player:
            board[action] = 0
            return action  # Must block!
        board[action] = 0
    
    # Order moves for better search
    legal = order_moves(board, player, legal, board_size, win_length)
    
    best_moves = []
    best_score = -float('inf')
    
    for action in legal:
        board[action] = player
        score = -minimax(board, -player, depth - 1, -float('inf'), float('inf'),
                         board_size, win_length, depth)
        board[action] = 0
        
        if score > best_score:
            best_score = score
            best_moves = [action]
        elif score == best_score:
            best_moves.append(action)
    
    return np.random.choice(best_moves)


def clear_transposition_table():
    """Clear the transposition table between games."""
    global transposition_table
    transposition_table = {}


def print_board(board, board_size=5):
    """Pretty print the board."""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    b2 = board.reshape(board_size, board_size)
    for row in b2:
        print(' '.join(symbols[int(cell)] for cell in row))
    print()


# Test and demo
if __name__ == "__main__":
    print("Testing optimized 5x5 minimax...")
    print("=" * 40)
    
    # Test 1: Basic functionality
    board = np.zeros(25, dtype=np.float32)
    move = get_best_move(board, player=1, depth=3, board_size=5, win_length=4)
    print(f"Test 1 - Best first move (depth=3): {move}")
    print(f"Position: ({move // 5}, {move % 5})")
    print()
    
    # Test 2: Must take winning move
    print("Test 2 - Should take winning move:")
    board = np.zeros(25, dtype=np.float32)
    board[6] = 1   # (1,1)
    board[7] = 1   # (1,2)
    board[8] = 1   # (1,3)
    # X has 3 in a row, should play (1,4) or (1,0) to win
    print_board(board)
    move = get_best_move(board, player=1, depth=2)
    print(f"X plays: ({move // 5}, {move % 5})")
    assert move in [5, 9], f"Should win at (1,0) or (1,4), got {move}"
    print("✅ Takes winning move!")
    print()
    
    # Test 3: Must block opponent
    print("Test 3 - Should block opponent's win:")
    board = np.zeros(25, dtype=np.float32)
    board[6] = -1   # (1,1)
    board[11] = -1  # (2,1)
    board[16] = -1  # (3,1)
    # O has 3 in a row vertically, X must block at (0,1) or (4,1)
    print_board(board)
    move = get_best_move(board, player=1, depth=2)
    print(f"X plays: ({move // 5}, {move % 5})")
    assert move in [1, 21], f"Should block at (0,1) or (4,1), got {move}"
    print("✅ Blocks opponent!")
    print()
    
    # Test 4: Reproduce the failing scenario
    print("Test 4 - The original failing scenario:")
    clear_transposition_table()
    board = np.zeros(25, dtype=np.float32)
    # Recreate position before move 7
    board[6] = 1    # X at (1,1)
    board[12] = 1   # X at (2,2)
    board[11] = 1   # X at (2,1)
    board[8] = -1   # O at (1,3)
    board[18] = -1  # O at (3,3)
    board[13] = -1  # O at (2,3)
    # O has 3 in column 3, X must block at (0,3) or (4,3)
    print("Before X's move 7:")
    print_board(board)
    move = get_best_move(board, player=1, depth=3)
    print(f"X plays: ({move // 5}, {move % 5})")
    assert move in [3, 23], f"Should block at (0,3) or (4,3), got {move}"
    print("✅ Now correctly blocks the threat!")
    print()
    
    # Test 5: Speed comparison
    print("Test 5 - Speed test:")
    import time
    
    board = np.zeros(25, dtype=np.float32)
    board[12] = 1  # Center
    
    for depth in [2, 3, 4]:
        clear_transposition_table()
        start = time.time()
        move = get_best_move(board.copy(), player=-1, depth=depth)
        elapsed = time.time() - start
        print(f"Depth {depth}: {elapsed:.3f}s, move: ({move // 5}, {move % 5})")
    
    print()
    print("=" * 40)
    print("All tests passed! ✅")
    print("The optimized version should now play much better at depth 2-3.")