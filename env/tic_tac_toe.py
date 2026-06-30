# tic_tac_toe.py

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from env.config import config


class TicTacToe:
    """
    Dynamic Tic-Tac-Toe environment (N×N board with K-in-a-row win condition)
    
    Board representation:
    - Player 1 (X): +1
    - Player 2 (O): -1
    - Empty: 0
    
    Actions: 0 to (board_size^2 - 1) corresponding to board positions
    For a 5×5 board:
    0  | 1  | 2  | 3  | 4
    ---+----+----+----+---
    5  | 6  | 7  | 8  | 9
    ---+----+----+----+---
    10 | 11 | 12 | 13 | 14
    ---+----+----+----+---
    15 | 16 | 17 | 18 | 19
    ---+----+----+----+---
    20 | 21 | 22 | 23 | 24
    """
    
    def __init__(self, cfg=None):
        """
        Initialize the environment
        
        Args:
            cfg: Config object (defaults to global config)
        """
        self.cfg = cfg if cfg is not None else config
        self.board_size = self.cfg.BOARD_SIZE
        self.win_length = self.cfg.WIN_LENGTH
        self.num_cells = self.board_size ** 2
        
        self.board = np.zeros(self.num_cells, dtype=np.float32)
        self.current_player = 1
        
    def reset(self, starting_player=None):
        """
        Reset the board to initial state
        
        Args:
            starting_player: 1, -1, or None (None = random)
            
        Returns:
            state: initial state from starting player's perspective
        """
        self.board = np.zeros(self.num_cells, dtype=np.float32)
        if starting_player is None:
            self.current_player = np.random.choice([1, -1])  # Random for training
        else:
            self.current_player = starting_player  # Fixed for evaluation/play
        return self.get_state(self.current_player)
    
    def get_valid_actions(self):
        """Return list of valid actions (empty positions)"""
        return np.where(self.board == 0)[0].tolist()
    
    def step(self, action):
        """
        Execute an action
        
        Args:
            action: position 0 to (num_cells-1)
            
        Returns:
            state: board state from next player's perspective
            done: whether game is over
            winner: 1 if player 1 wins, -1 if player 2 wins, 0 if draw, None if ongoing
        """
        if self.board[action] != 0:
            raise ValueError(f"Invalid action: position {action} is already occupied")
        
        # Make move
        self.board[action] = self.current_player
        
        # Check if game is over
        winner = self.check_winner()
        done = winner is not None
        
        # Switch player
        self.current_player = -self.current_player
        
        # Get state from new player's perspective
        state = self.get_state(self.current_player)
        
        return state, done, winner
    
    def get_state(self, player):
        """
        Get board state from the perspective of the given player
        
        From player's perspective:
        - Player's pieces: +1
        - Opponent's pieces: -1
        - Empty: 0
        
        Args:
            player: 1 or -1
            
        Returns:
            state: numpy array of shape (num_cells,)
        """
        if player == 1:
            return self.board.copy()
        else:
            return -self.board.copy()
    
    def check_winner(self):
        """
        Check if there's a winner
        
        This checks all possible lines (horizontal, vertical, diagonal) for
        a sequence of win_length consecutive marks.
        
        Returns:
            1 if player 1 (X) wins
            -1 if player 2 (O) wins
            0 if draw
            None if game is still ongoing
        """
        board_2d = self.board.reshape(self.board_size, self.board_size)
        
        # Check horizontal lines
        for row in range(self.board_size):
            winner = self._check_line_for_winner(board_2d[row, :])
            if winner is not None:
                return winner
        
        # Check vertical lines
        for col in range(self.board_size):
            winner = self._check_line_for_winner(board_2d[:, col])
            if winner is not None:
                return winner
        
        # Check diagonal lines (top-left to bottom-right)
        for offset in range(-(self.board_size - self.win_length), 
                           (self.board_size - self.win_length) + 1):
            diag = np.diagonal(board_2d, offset=offset)
            if len(diag) >= self.win_length:
                winner = self._check_line_for_winner(diag)
                if winner is not None:
                    return winner
        
        # Check anti-diagonal lines (top-right to bottom-left)
        flipped_board = np.fliplr(board_2d)
        for offset in range(-(self.board_size - self.win_length), 
                           (self.board_size - self.win_length) + 1):
            diag = np.diagonal(flipped_board, offset=offset)
            if len(diag) >= self.win_length:
                winner = self._check_line_for_winner(diag)
                if winner is not None:
                    return winner
        
        # Check for draw (no empty spaces)
        if not (self.board == 0).any():
            return 0
        
        # Game still ongoing
        return None
    
    def _check_line(self, line):
        """
        Check if a line has win_length consecutive marks (legacy method)
        Used for simple full-line checking
        
        Args:
            line: 1D numpy array
            
        Returns:
            True if line has win_length consecutive identical non-zero marks
        """
        if len(line) < self.win_length:
            return False
        return abs(line.sum()) == self.win_length and (line != 0).sum() == self.win_length
    
    def _check_line_for_winner(self, line):
        """
        Check if a line contains win_length consecutive marks and return winner
        
        Args:
            line: 1D numpy array
            
        Returns:
            1 if player 1 has win_length in a row
            -1 if player 2 has win_length in a row
            None otherwise
        """
        if len(line) < self.win_length:
            return None
        
        # Check for consecutive sequences
        for i in range(len(line) - self.win_length + 1):
            segment = line[i:i + self.win_length]
            if (segment != 0).all() and abs(segment.sum()) == self.win_length:
                return int(segment[0])
        
        return None
    
    def render(self):
        """Print the board (for debugging/visualization)"""
        symbols = {0: self.cfg.EMPTY_SYMBOL, 
                   1: self.cfg.PLAYER1_SYMBOL, 
                   -1: self.cfg.PLAYER2_SYMBOL}
        board_2d = self.board.reshape(self.board_size, self.board_size)
        
        # Print column numbers
        col_header = "  " + "   ".join([str(i) for i in range(self.board_size)])
        print("\n" + col_header)
        
        for i in range(self.board_size):
            row = " | ".join([symbols[board_2d[i, j]] for j in range(self.board_size)])
            print(f"{i} {row}")
            if i < self.board_size - 1:
                separator = "-" * (4 * self.board_size - 1)
                print("  " + separator)
        print()


# Quick test
if __name__ == "__main__":
    from env.config import Config, get_config
    
    print("Testing Dynamic Tic-Tac-Toe environment...")
    print("=" * 60)
    
    # Method 1: Use preset
    print("\n[Method 1] Using preset:")
    cfg = get_config("medium")
    env = TicTacToe(cfg)
    print(f"  {cfg.BOARD_SIZE}×{cfg.BOARD_SIZE} board, {cfg.WIN_LENGTH}-in-a-row")
    
    # Method 2: Custom config - CHANGE THESE VALUES
    print("\n[Method 2] Custom config:")
    my_cfg = Config()
    my_cfg.BOARD_SIZE = 10      # ← Change this
    my_cfg.WIN_LENGTH = 3       # ← Change this
    my_cfg.validate()
    env = TicTacToe(my_cfg)
    print(f"  {my_cfg.BOARD_SIZE}×{my_cfg.BOARD_SIZE} board, {my_cfg.WIN_LENGTH}-in-a-row")
    env.render()
    
    print("=" * 60)
    print("✅ Test complete!")