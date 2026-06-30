# play_mcts_human.py

import torch
import numpy as np

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.config import get_config
from env.tic_tac_toe import TicTacToe
from train_agent.mcts import MCTS
from train_agent.network import PolicyValueNet


# ------------------------------------------------------------
#   Pretty-print board
# ------------------------------------------------------------
def print_board(board, board_size=3):
    """
    board: 1D array of length board_size^2, values in {1, -1, 0}
    1  -> 'X'
    -1 -> 'O'
    0  -> '.'
    """
    symbols = {1: 'X', -1: 'O', 0: '.'}
    b2 = board.reshape(board_size, board_size)
    print("\nBoard:")
    for r in range(board_size):
        row_syms = [symbols[int(x)] for x in b2[r]]
        print(" " + " ".join(row_syms))
    print()


# ------------------------------------------------------------
#   Human move input
# ------------------------------------------------------------
def get_human_action(board, board_size=3):
    """
    Let the human choose a move.
    Supports:
        - single index: 0-8
        - row col: e.g. '0 2', '1 1' for (row, col)
    """
    legal = np.where(board == 0)[0]
    legal_set = set(int(i) for i in legal)

    while True:
        user_input = input(
            f"Your move (0–{board_size*board_size-1}) or 'row col' (e.g. '0 2'): "
        ).strip()

        if " " in user_input:
            # try parse as row col
            parts = user_input.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                r = int(parts[0])
                c = int(parts[1])
                if 0 <= r < board_size and 0 <= c < board_size:
                    idx = r * board_size + c
                    if idx in legal_set:
                        return idx
                    else:
                        print("That cell is not empty. Try again.")
                        continue
            print("Invalid input format. Use 'row col', e.g. '0 2'.")
        else:
            # try parse as flat index
            if not user_input.isdigit():
                print("Please enter a number.")
                continue
            idx = int(user_input)
            if idx not in legal_set:
                print("Illegal move (cell occupied or out of range). Try again.")
                continue
            return idx


# ------------------------------------------------------------
#   Play one game: Human vs MCTS Agent
# ------------------------------------------------------------
def play_human_vs_mcts(model_path="models/trained_agent.pth", preset="medium"):
    cfg = get_config(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board_size = cfg.BOARD_SIZE

    # Load model
    net = PolicyValueNet(cfg).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    env = TicTacToe(cfg)
    mcts = MCTS(net, cfg, device=device)

    print("=== Human vs MCTS Agent (Tic-Tac-Toe) ===")
    print(f"Board: {board_size}×{board_size}, {cfg.WIN_LENGTH} in a row to win.")

    # Choose side
    while True:
        choice = input("Do you want to play first as X? (y/n): ").strip().lower()
        if choice in ["y", "yes"]:
            human_player = 1   # X
            break
        elif choice in ["n", "no"]:
            human_player = -1  # O
            break
        else:
            print("Please type 'y' or 'n'.")

    agent_player = -human_player

    # Explicitly choose who starts in the env
    # If human is X: human starts
    # If human is O: agent starts
    if human_player == 1:
        starting_player = human_player   # human starts as X
    else:
        starting_player = agent_player   # agent starts as X / first

    env.reset(starting_player=starting_player)

    print("\nGame start!")
    print(f"You are {'X' if human_player == 1 else 'O'}")
    print_board(env.board, board_size)

    done = False
    winner = None

    while not done:
        board = env.board.copy()
        current_player = env.current_player

        if current_player == human_player:
            # Human's turn
            print("Your turn.")
            action = get_human_action(board, board_size=board_size)
        else:
            # Agent's turn using MCTS (deterministic best move)
            print("Agent is thinking...")
            pi = mcts.run(board, current_player, add_dirichlet=False)
            action = int(np.argmax(pi))
            print(f"Agent plays at index {action}.")

        _, done, winner = env.step(action)
        print_board(env.board, board_size)

    # Game over
    if winner == 0:
        print("Game over: Draw.")
    elif winner == human_player:
        print("Game over: You win! 🎉")
    else:
        print("Game over: Agent wins. 🤖")

    print("Thanks for playing!")


if __name__ == "__main__":
    play_human_vs_mcts()
