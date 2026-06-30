# test_minimax_randomness.py
"""
Test if minimax properly randomizes among optimal moves.
"""

import numpy as np

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from utils.minimax_limited import get_best_move


def test_minimax_randomness():
    """
    Test minimax on positions with multiple optimal moves.
    Should see variety in responses.
    """
    
    print("="*60)
    print("Testing Minimax Randomness")
    print("="*60)
    
    # Test 1: Empty board (X to move)
    # All 9 positions should be equally optimal
    print("\nTest 1: Empty board (X to move)")
    print("All positions are optimal - should see variety")
    print("-"*60)
    
    board = np.zeros(9, dtype=np.float32)
    moves = []
    
    for i in range(20):
        move = get_best_move(board.copy(), player=1, board_size=3, win_length=3)
        moves.append(move)
    
    print(f"20 trials: {moves}")
    print(f"Unique moves: {len(set(moves))} out of 9 possible")
    print(f"Distribution: {dict((m, moves.count(m)) for m in set(moves))}")
    
    if len(set(moves)) > 1:
        print("✅ Minimax is randomizing!")
    else:
        print("❌ Minimax is NOT randomizing - always picks same move!")
    
    # Test 2: X at center, O to respond
    # Corner positions (0,2,6,8) are optimal
    print("\n" + "="*60)
    print("Test 2: X at center, O to respond")
    print("Corners (0,2,6,8) are optimal - should see variety")
    print("-"*60)
    
    board = np.zeros(9, dtype=np.float32)
    board[4] = 1  # X at center
    moves = []
    
    for i in range(20):
        move = get_best_move(board.copy(), player=-1, board_size=3, win_length=3)
        moves.append(move)
    
    print(f"20 trials: {moves}")
    print(f"Unique moves: {len(set(moves))} out of 4 corners")
    print(f"Distribution: {dict((m, moves.count(m)) for m in set(moves))}")
    
    corners = {0, 2, 6, 8}
    all_corners = all(m in corners for m in moves)
    
    if all_corners and len(set(moves)) > 1:
        print("✅ Minimax chooses only corners and randomizes among them!")
    elif all_corners:
        print("⚠️  Minimax chooses corners but doesn't randomize")
    else:
        print("❌ Minimax is choosing non-corner moves (bug!)")
    
    # Test 3: Later position with multiple good moves
    print("\n" + "="*60)
    print("Test 3: Mid-game position")
    print("-"*60)
    
    # X: 0, 4, 8 (diagonal threat)
    # O: 1, 3
    # O must block, but might have multiple blocking options
    board = np.zeros(9, dtype=np.float32)
    board[0] = 1   # X
    board[4] = 1   # X
    board[1] = -1  # O
    board[3] = -1  # O
    
    print("Board:")
    print(board.reshape(3,3))
    print("\nO to move (must block X's diagonal)")
    
    moves = []
    for i in range(20):
        move = get_best_move(board.copy(), player=-1, board_size=3, win_length=3)
        moves.append(move)
    
    print(f"20 trials: {moves}")
    print(f"Unique moves: {len(set(moves))}")
    print(f"Distribution: {dict((m, moves.count(m)) for m in set(moves))}")
    
    # Test 4: Check if np.random is working
    print("\n" + "="*60)
    print("Test 4: Sanity check - is np.random.choice working?")
    print("-"*60)
    
    test_choices = [np.random.choice([0, 2, 6, 8]) for _ in range(20)]
    print(f"Random choices from [0,2,6,8]: {test_choices}")
    print(f"Unique: {len(set(test_choices))}")
    
    if len(set(test_choices)) > 1:
        print("✅ np.random.choice is working")
    else:
        print("❌ np.random.choice seems broken!")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If minimax is NOT randomizing, the issue is likely:")
    print("1. np.random.choice is being called with wrong parameters")
    print("2. The list of best_moves only has one element")
    print("3. Random seed is set somewhere")
    print("="*60)


if __name__ == "__main__":
    test_minimax_randomness()