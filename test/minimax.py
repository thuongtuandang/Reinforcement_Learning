# test_minimax_5x5.py
"""
Test minimax_limited for 5x5 board:
1. Minimax vs Random
2. Minimax vs Minimax
"""

import numpy as np

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from utils.minimax_limited import get_best_move, check_winner


def random_move(board):
    """Pick random legal move."""
    legal = np.where(board == 0)[0]
    return np.random.choice(legal)


def play_game(player_x, player_o, board_size=5, win_length=4, show=False):
    """
    Play one game.
    player_x/player_o: 'minimax' or 'random'
    Returns winner: 1, -1, or 0
    """
    board = np.zeros(board_size * board_size, dtype=np.float32)
    current_player = 1
    
    while True:
        winner = check_winner(board, board_size, win_length)
        if winner is not None:
            return winner
        
        # Get move
        if current_player == 1:
            if player_x == 'minimax':
                move = get_best_move(board, 1, depth=2, board_size=board_size, win_length=win_length)
            else:
                move = random_move(board)
        else:
            if player_o == 'minimax':
                move = get_best_move(board, -1, depth=3, board_size=board_size, win_length=win_length)
            else:
                move = random_move(board)
        
        board[move] = current_player
        
        if show:
            print_board(board, board_size)
        
        current_player = -current_player


def print_board(board, board_size):
    """Print board."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    b2d = board.reshape(board_size, board_size)
    for r in range(board_size):
        print(' '.join([symbols[int(b2d[r, c])] for c in range(board_size)]))
    print()


def test_minimax_vs_random(n_games=100):
    """Test minimax vs random player."""
    print("="*60)
    print("TEST: Minimax vs Random")
    print("="*60)
    
    results = {'minimax_wins': 0, 'random_wins': 0, 'draws': 0}
    
    for i in range(n_games):
        if i < n_games // 2:
            # Minimax as X
            winner = play_game('minimax', 'random')
            if winner == 1:
                results['minimax_wins'] += 1
            elif winner == -1:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
        else:
            # Minimax as O
            winner = play_game('random', 'minimax')
            if winner == -1:
                results['minimax_wins'] += 1
            elif winner == 1:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{n_games}")
    
    print(f"\nResults ({n_games} games):")
    print(f"  Minimax wins: {results['minimax_wins']} ({results['minimax_wins']/n_games*100:.1f}%)")
    print(f"  Random wins:  {results['random_wins']} ({results['random_wins']/n_games*100:.1f}%)")
    print(f"  Draws:        {results['draws']} ({results['draws']/n_games*100:.1f}%)")
    print("="*60 + "\n")


def test_minimax_vs_minimax(n_games=20):
    """Test minimax vs itself."""
    print("="*60)
    print("TEST: Minimax vs Minimax")
    print("="*60)
    
    results = {'x_wins': 0, 'o_wins': 0, 'draws': 0}
    
    for i in range(n_games):
        winner = play_game('minimax', 'minimax')
        
        if winner == 1:
            results['x_wins'] += 1
        elif winner == -1:
            results['o_wins'] += 1
        else:
            results['draws'] += 1
        
        print(f"Game {i+1}: {'X wins' if winner == 1 else 'O wins' if winner == -1 else 'Draw'}")
    
    print(f"\nResults ({n_games} games):")
    print(f"  X wins: {results['x_wins']} ({results['x_wins']/n_games*100:.1f}%)")
    print(f"  O wins: {results['o_wins']} ({results['o_wins']/n_games*100:.1f}%)")
    print(f"  Draws:  {results['draws']} ({results['draws']/n_games*100:.1f}%)")
    print("="*60 + "\n")


def show_sample_game():
    """Show one game minimax vs minimax."""
    print("="*60)
    print("SAMPLE GAME: Minimax (X) vs Minimax (O)")
    print("="*60)
    
    board = np.zeros(25, dtype=np.float32)
    current_player = 1
    move_num = 0
    
    while True:
        winner = check_winner(board, 5, 4)
        if winner is not None:
            print(f"\nResult: {'X wins!' if winner == 1 else 'O wins!' if winner == -1 else 'Draw'}")
            break
        if current_player == 1:
            move = get_best_move(board, current_player, depth=2, board_size=5, win_length=4)
        else:
            move = get_best_move(board, current_player, depth=3, board_size=5, win_length=4)
        board[move] = current_player
        move_num += 1
        
        symbol = 'X' if current_player == 1 else 'O'
        row, col = move // 5, move % 5
        print(f"Move {move_num}: {symbol} plays ({row},{col})")
        print_board(board, 5)
        
        current_player = -current_player
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Show one sample game
    show_sample_game()
    
    # Test minimax vs minimax (should be mostly draws or slight X advantage)
    test_minimax_vs_minimax(n_games=10)
    
    # Test minimax vs random (minimax should dominate)
    test_minimax_vs_random(n_games=50)