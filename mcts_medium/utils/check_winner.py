import numpy as np

# ============================================================
#   Board winner helper (used by MCTS nodes)
# ============================================================

def check_winner_board(board, board_size, win_length):
    """
    Board: 1D np.array of length board_size^2
    Values: +1 (X), -1 (O), 0 (empty)

    Returns:
        1  if player +1 wins
        -1 if player -1 wins
        0  if draw
        None if game ongoing
    """
    b2 = board.reshape(board_size, board_size)

    # Horizontal
    for i in range(board_size):
        for j in range(board_size - win_length + 1):
            window = b2[i, j:j + win_length]
            if 0 not in window and abs(window.sum()) == win_length:
                return int(np.sign(window.sum()))

    # Vertical
    for j in range(board_size):
        for i in range(board_size - win_length + 1):
            window = b2[i:i + win_length, j]
            if 0 not in window and abs(window.sum()) == win_length:
                return int(np.sign(window.sum()))

    # Diagonal \
    for i in range(board_size - win_length + 1):
        for j in range(board_size - win_length + 1):
            window = [b2[i + k, j + k] for k in range(win_length)]
            if 0 not in window and abs(sum(window)) == win_length:
                return int(np.sign(sum(window)))

    # Diagonal /
    for i in range(board_size - win_length + 1):
        for j in range(win_length - 1, board_size):
            window = [b2[i + k, j - k] for k in range(win_length)]
            if 0 not in window and abs(sum(window)) == win_length:
                return int(np.sign(sum(window)))

    # Draw?
    if not (board == 0).any():
        return 0

    return None  # ongoing
