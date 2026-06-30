# config.py
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

"""
Configuration file for Tic-Tac-Toe RL Agent

This file contains all hyperparameters and settings for the environment,
network architecture, and training process.
"""


class Config:
    """Configuration class for Tic-Tac-Toe"""
    
    # ==================== Environment Settings ====================
    
    # Board dimensions (for N×N board)
    BOARD_SIZE = 5
    
    # Number of marks in a row needed to win
    WIN_LENGTH = 3
    
    # ==================== Network Architecture ====================
    
    # Hidden layer size for policy network
    HIDDEN_SIZE = 64
    
    # Input size is calculated as: board_size^2 * 2 + 1
    # (my_pieces + opponent_pieces + side_to_move_bit)
    @property
    def input_size(self):
        return self.BOARD_SIZE ** 2 * 2
    
    # Output size is the number of possible actions (one per cell)
    @property
    def output_size(self):
        return self.BOARD_SIZE ** 2
    
    # ==================== MCTS Settings ====================
    
    # Learning rate for optimizer
    LEARNING_RATE = 1e-3

    # Number of MCTS simulations per move
    MCTS_SIMULATIONS = 200
    
    # UCB exploration constant
    MCTS_C_PUCT = 1.4
    
    # Dirichlet noise for root exploration
    MCTS_DIRICHLET_ALPHA = 0.3
    
    # Sampling temperature (1.0 = stochastic, 0.0 = deterministic)
    MCTS_TEMPERATURE = 1.0
    
    # MCTS training episodes
    NUM_EPISODES_MCTS = 20000
    
    # Mini-batch size for MCTS training
    MCTS_BATCH_SIZE = 32
    
    # ==================== Evaluation Settings ====================
    
    # Number of games to play during evaluation
    EVAL_GAMES = 100
    
    # ==================== File Paths ====================
    
    # Directory to save trained models
    MODEL_DIR = "models"
    
    # Directory to save training plots
    PLOT_DIR = "plots"
    
    # Default model filename
    MODEL_NAME = "trained_agent.pth"
    
    @property
    def model_path(self):
        return f"{self.MODEL_DIR}/{self.MODEL_NAME}"
    
    # ==================== Display Settings ====================
    
    # Symbol for empty cell
    EMPTY_SYMBOL = '.'
    
    # Symbol for player 1 (X)
    PLAYER1_SYMBOL = 'X'
    
    # Symbol for player 2 (O)
    PLAYER2_SYMBOL = 'O'
    
    # ==================== Validation ====================
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.BOARD_SIZE >= 3, "Board size must be at least 3"
        assert self.WIN_LENGTH >= 3, "Win length must be at least 3"
        assert self.WIN_LENGTH <= self.BOARD_SIZE, "Win length cannot exceed board size"
        assert self.HIDDEN_SIZE > 0, "Hidden size must be positive"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"
        assert self.NUM_EPISODES > 0, "Number of episodes must be positive"
        
    def __repr__(self):
        """String representation of config"""
        return f"""
TicTacToe Configuration:
========================
Environment:
  - Board Size: {self.BOARD_SIZE}×{self.BOARD_SIZE}
  - Win Length: {self.WIN_LENGTH}
  - Total Cells: {self.BOARD_SIZE ** 2}
  - State Space: ~3^{self.BOARD_SIZE ** 2}

Network:
  - Input Size: {self.input_size}
  - Hidden Size: {self.HIDDEN_SIZE}
  - Output Size: {self.output_size}

Training:
  - Episodes: {self.NUM_EPISODES}
  - Learning Rate: {self.LEARNING_RATE}
  - Eval Games: {self.EVAL_GAMES}

Paths:
  - Model: {self.model_path}
  - Plots: {self.PLOT_DIR}
========================
"""


# Global config instance
config = Config()


# ==================== Preset Configurations ====================

class SmallConfig(Config):
    """3×3 classic Tic-Tac-Toe"""
    BOARD_SIZE = 3
    WIN_LENGTH = 3
    HIDDEN_SIZE = 128
    NUM_EPISODES = 10000


class MediumConfig(Config):
    """5×5 board with 4-in-a-row"""
    BOARD_SIZE = 5
    WIN_LENGTH = 4
    HIDDEN_SIZE = 256
    NUM_EPISODES = 2000000


class LargeConfig(Config):
    """10×10 board with 4-in-a-row (current default)"""
    BOARD_SIZE = 8
    WIN_LENGTH = 5
    HIDDEN_SIZE = 256
    NUM_EPISODES = 1000000


class GomokuConfig(Config):
    """15×15 Gomoku (5-in-a-row)"""
    BOARD_SIZE = 15
    WIN_LENGTH = 5
    HIDDEN_SIZE = 512
    NUM_EPISODES = 2000000


# ==================== Helper Functions ====================

def get_config(preset="large"):
    """
    Get a configuration preset
    
    Args:
        preset: one of "small", "medium", "large", "gomoku", or "default"
        
    Returns:
        Config instance
    """
    presets = {
        "small": SmallConfig(),
        "medium": MediumConfig(),
        "large": LargeConfig(),
        "gomoku": GomokuConfig(),
        "default": Config()
    }
    
    if preset not in presets:
        print(f"Warning: Unknown preset '{preset}', using 'large'")
        preset = "large"
    
    cfg = presets[preset]
    cfg.validate()
    return cfg


# Quick test
if __name__ == "__main__":
    print("Testing configuration system...\n")
    
    # Test default config
    print("=" * 60)
    print("DEFAULT CONFIG:")
    print(config)
    config.validate()
    
    # Test all presets
    for preset_name in ["small", "medium", "medium4", "large", "gomoku"]:
        print("=" * 60)
        print(f"{preset_name.upper()} PRESET:")
        cfg = get_config(preset_name)
        print(cfg)
    
    print("=" * 60)
    print("✅ Configuration test complete!")