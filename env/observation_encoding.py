# observation_encoding.py
"""
Symmetric observation encoding for self-play

Key difference from standard encoding:
- NO side_bit! 
- Always encodes from current player's POV
- Enforces symmetric learning by design
- Prevents asymmetric collapse in TD learning

This is the recommended encoding for self-play to prevent the network
from learning different strategies for X vs O.
"""

import numpy as np


def encode_obs(board, current_player, cfg):
    """
    Encode board state from current player's perspective (POV encoding)
    
    NO side_bit included - this enforces symmetry by construction!
    
    The network sees the board purely from "my perspective" vs "opponent's perspective"
    without knowing whether it's playing as X or O. This forces the network to learn
    a single, symmetric strategy that works for both players.
    
    Args:
        board: numpy array of shape (board_size^2,) with values in {-1, 0, 1}
               -1 = O's piece, 0 = empty, 1 = X's piece
        current_player: +1 (X) or -1 (O) - who is about to move
        cfg: Config object with BOARD_SIZE
    
    Returns:
        observation: numpy array of shape (2 * board_size^2,)
                     First half: my pieces (current player's pieces)
                     Second half: opponent's pieces
    
    Example:
        Board state: [X, X, O, ., .]  (X=1, O=-1, .=0)
        
        If current_player = 1 (X):
            my_pieces = [1, 1, 0, 0, 0]
            opp_pieces = [0, 0, 1, 0, 0]
            return [1,1,0,0,0, 0,0,1,0,0]
        
        If current_player = -1 (O):
            my_pieces = [0, 0, 1, 0, 0]  (O's pieces)
            opp_pieces = [1, 1, 0, 0, 0]  (X's pieces)
            return [0,0,1,0,0, 1,1,0,0,0]
        
        Note: Different inputs for same board! Network must learn symmetric strategy.
    """
    # My pieces: where board matches current player
    my_pieces = (board == current_player).astype(np.float32)
    
    # Opponent's pieces: where board matches opposite player
    opp_pieces = (board == -current_player).astype(np.float32)
    
    # Concatenate: [my_pieces, opp_pieces]
    # NO side_bit! This is critical for symmetric learning.
    observation = np.concatenate([my_pieces, opp_pieces])
    
    return observation


def encode_obs_batch(boards, current_players, cfg):
    """
    Encode multiple board states at once (for batch processing)
    
    Args:
        boards: numpy array of shape (batch_size, board_size^2)
        current_players: numpy array of shape (batch_size,) with values in {-1, 1}
        cfg: Config object
    
    Returns:
        observations: numpy array of shape (batch_size, 2 * board_size^2)
    """
    batch_size = boards.shape[0]
    board_size_sq = cfg.BOARD_SIZE * cfg.BOARD_SIZE
    
    observations = np.zeros((batch_size, 2 * board_size_sq), dtype=np.float32)
    
    for i in range(batch_size):
        observations[i] = encode_obs(boards[i], current_players[i], cfg)
    
    return observations


# Test
if __name__ == "__main__":
    from env.config import get_config
    
    print("Testing Symmetric Observation Encoding")
    print("=" * 70)
    
    cfg = get_config("small")  # 3x3 board
    
    # Test board:
    # X X O
    # . . .
    # . . .
    board = np.array([1, 1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    print("\nTest Board:")
    print("X X O")
    print(". . .")
    print(". . .")
    print(f"Board array: {board}")
    
    # Encode from X's perspective
    print("\n" + "-" * 70)
    print("Encoding from X's perspective (current_player = 1):")
    obs_x = encode_obs(board, current_player=1, cfg=cfg)
    print(f"Observation shape: {obs_x.shape}")
    print(f"My pieces (X):  {obs_x[:9]}")
    print(f"Opp pieces (O): {obs_x[9:]}")
    
    # Encode from O's perspective
    print("\n" + "-" * 70)
    print("Encoding from O's perspective (current_player = -1):")
    obs_o = encode_obs(board, current_player=-1, cfg=cfg)
    print(f"Observation shape: {obs_o.shape}")
    print(f"My pieces (O):  {obs_o[:9]}")
    print(f"Opp pieces (X): {obs_o[9:]}")
    
    # Verify symmetry
    print("\n" + "-" * 70)
    print("Verification:")
    print("✅ X's my_pieces == O's opp_pieces:", np.array_equal(obs_x[:9], obs_o[9:]))
    print("✅ X's opp_pieces == O's my_pieces:", np.array_equal(obs_x[9:], obs_o[:9]))
    print("\n✅ Different inputs for same board - enforces symmetric learning!")
    
    # Test with medium board
    print("\n" + "=" * 70)
    cfg_medium = get_config("medium")  # 5x5 board
    board_medium = np.zeros(25, dtype=np.float32)
    board_medium[0] = 1  # X at (0,0)
    board_medium[12] = -1  # O at center
    
    print("Testing with 5×5 board:")
    obs_medium = encode_obs(board_medium, current_player=1, cfg=cfg_medium)
    print(f"Observation shape: {obs_medium.shape}")
    print(f"Expected shape: ({cfg_medium.BOARD_SIZE * cfg_medium.BOARD_SIZE * 2},)")
    assert obs_medium.shape[0] == 50, "Should be 50-dimensional (25 my + 25 opp)"
    print("✅ Correct shape!")
    
    # Test batch encoding
    print("\n" + "=" * 70)
    print("Testing batch encoding:")
    boards_batch = np.array([board, board, board])
    players_batch = np.array([1, -1, 1])
    obs_batch = encode_obs_batch(boards_batch, players_batch, cfg)
    print(f"Batch shape: {obs_batch.shape}")
    print(f"Expected: (3, 18)")
    assert obs_batch.shape == (3, 18), "Batch encoding failed"
    print("✅ Batch encoding works!")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    
    print("\nKey Properties:")
    print("1. ✅ No side_bit - prevents asymmetric learning")
    print("2. ✅ Always POV encoding - 'my pieces' vs 'opponent pieces'")
    print("3. ✅ Same board → different inputs for X vs O")
    print("4. ✅ Network must learn symmetric strategy")
    print("\nThis encoding is CRITICAL for preventing TD collapse in self-play!")