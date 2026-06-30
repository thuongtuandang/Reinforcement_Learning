# test_minimax.py
"""
Test minimax vs minimax over 500 games.
Print specific games: 5, 100, 200, 305, 405
"""

import numpy as np

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from utils.minimax_limited import get_best_move, check_winner


def play_minimax_vs_minimax(board_size=5, win_length=4, show_board=False):
    """Play one game of minimax vs minimax"""
    board = np.zeros(board_size * board_size, dtype=np.float32)
    current_player = 1  # X starts
    moves = []  # Track moves for replay
    
    while True:
        # Check winner
        winner = check_winner(board, board_size, win_length)
        if winner is not None:
            return winner, moves
        
        # Get best move
        move = get_best_move(board, current_player, board_size, win_length)
        
        if move is None:
            return 0, moves
        
        # Make move
        board[move] = current_player
        moves.append((current_player, move))
        
        # Switch player
        current_player = -current_player


def print_game(game_num, moves, winner, board_size=5):
    """Print a game from move list"""
    print(f"\n{'='*60}")
    print(f"GAME {game_num}:")
    print('='*60)
    
    board = np.zeros(board_size * board_size, dtype=np.float32)
    
    for move_num, (player, move) in enumerate(moves, 1):
        board[move] = player
        
        symbol = 'X' if player == 1 else 'O'
        row = move // board_size
        col = move % board_size
        
        print(f"Move {move_num}: {symbol} plays ({row},{col}) [pos {move}]")
        
        # Print board
        b2d = board.reshape(board_size, board_size)
        for r in range(board_size):
            row_str = ' | '.join([
                'X' if b2d[r,c] == 1 else 'O' if b2d[r,c] == -1 else '.'
                for c in range(board_size)
            ])
            print('  ' + row_str)
        print()
    
    if winner == 0:
        print("✅ Result: Draw")
    elif winner == 1:
        print("❌ Result: X wins (ERROR!)")
    else:
        print("❌ Result: O wins (ERROR!)")


# Main test
print("Testing Minimax vs Minimax over 500 games")
print("="*60)
print("Will print detailed output for games: 5, 100, 200, 305, 405")
print("="*60)

games_to_print = {5, 100, 200, 305, 405}
results = {'x_wins': 0, 'o_wins': 0, 'draws': 0}
all_games = {}

for game_num in range(1, 501):
    winner, moves = play_minimax_vs_minimax(board_size=5, win_length=4)
    
    # Store game if we need to print it
    if game_num in games_to_print:
        all_games[game_num] = (moves, winner)
    
    # Track results
    if winner == 1:
        results['x_wins'] += 1
    elif winner == -1:
        results['o_wins'] += 1
    else:
        results['draws'] += 1
    
    # Progress indicator
    if game_num % 50 == 0:
        print(f"Progress: {game_num}/500 games completed...")

# Print selected games
print("\n" + "="*60)
print("DETAILED GAMES:")
print("="*60)

for game_num in sorted(games_to_print):
    if game_num in all_games:
        moves, winner = all_games[game_num]
        print_game(game_num, moves, winner)

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY (500 games):")
print("="*60)
print(f"X wins:  {results['x_wins']:3d} ({results['x_wins']/500*100:5.1f}%)")
print(f"O wins:  {results['o_wins']:3d} ({results['o_wins']/500*100:5.1f}%)")
print(f"Draws:   {results['draws']:3d} ({results['draws']/500*100:5.1f}%)")
print("="*60)

if results['draws'] == 500:
    print("✅ PERFECT! All 500 games were draws!")
    print("✅ Minimax is truly optimal!")
else:
    print("❌ ERROR: Some games were not draws!")
    print("❌ Minimax has a bug!")

print("="*60)