# play_with_temperature.py

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


def sample_action(pi, temperature=0.3):
    """
    Sample action from policy with temperature.
    
    Args:
        pi: probability distribution over actions
        temperature: sampling temperature
            - 0.0 = deterministic (argmax)
            - 0.3 = mostly best moves, some exploration
            - 1.0 = fully stochastic
    
    Returns:
        action index
    """
    if temperature == 0:
        return int(np.argmax(pi))
    
    # Apply temperature
    pi_temp = np.power(pi, 1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum()
    
    return int(np.random.choice(len(pi_temp), p=pi_temp))


def print_board(board, board_size=3):
    """Pretty-print the board"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    b2 = board.reshape(board_size, board_size)
    print("\nBoard:")
    for r in range(board_size):
        row_syms = [symbols[int(x)] for x in b2[r]]
        print(" " + " ".join(row_syms))
    print()


def get_human_action(board, board_size=3):
    """Get human move input"""
    legal = np.where(board == 0)[0]
    legal_set = set(int(i) for i in legal)

    while True:
        user_input = input(
            f"Your move (0–{board_size*board_size-1}) or 'row col' (e.g. '0 2'): "
        ).strip()

        if " " in user_input:
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
            if not user_input.isdigit():
                print("Please enter a number.")
                continue
            idx = int(user_input)
            if idx not in legal_set:
                print("Illegal move (cell occupied or out of range). Try again.")
                continue
            return idx


def play_human_vs_agent(model_path="saved_models/trained_agent_small_50k.pth", 
                        preset="medium", 
                        agent_temperature=0.3):
    """
    Play against the agent with configurable temperature.
    
    Args:
        model_path: path to trained model
        preset: config preset
        agent_temperature: temperature for agent's move sampling
    """
    cfg = get_config(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board_size = cfg.BOARD_SIZE

    # Load model
    net = PolicyValueNet(cfg).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    env = TicTacToe(cfg)
    mcts = MCTS(net, cfg, device=device)

    print("=" * 60)
    print("Human vs MCTS Agent (Tic-Tac-Toe)")
    print("=" * 60)
    print(f"Board: {board_size}×{board_size}, {cfg.WIN_LENGTH} in a row to win.")
    print(f"Agent temperature: {agent_temperature}")
    if agent_temperature == 0:
        print("  (Deterministic - always picks best move)")
    elif agent_temperature < 0.5:
        print("  (Mostly best moves, slight exploration)")
    else:
        print("  (More stochastic, explores alternatives)")
    print("=" * 60)

    # Choose side
    while True:
        choice = input("\nDo you want to play first as X? (y/n): ").strip().lower()
        if choice in ["y", "yes"]:
            human_player = 1   # X
            break
        elif choice in ["n", "no"]:
            human_player = -1  # O
            break
        else:
            print("Please type 'y' or 'n'.")

    agent_player = -human_player

    # Set starting player
    if human_player == 1:
        starting_player = human_player
    else:
        starting_player = agent_player

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
            # Agent's turn with temperature sampling
            print("Agent is thinking...")
            pi = mcts.run(board, current_player, add_dirichlet=False)
            
            # Sample with temperature
            action = sample_action(pi, temperature=agent_temperature)
            
            row = action // board_size
            col = action % board_size
            print(f"Agent plays at ({row},{col}) [index {action}]")

        _, done, winner = env.step(action)
        print_board(env.board, board_size)

    # Game over
    print("=" * 60)
    if winner == 0:
        print("Game over: Draw.")
    elif winner == human_player:
        print("Game over: You win! 🎉")
    else:
        print("Game over: Agent wins. 🤖")
    print("=" * 60)
    print("Thanks for playing!")


if __name__ == "__main__":
    # You can change these parameters
    play_human_vs_agent(
        model_path="models/trained_agent.pth",
        preset="medium",
        agent_temperature=0.3  # Change this: 0.0, 0.3, 0.7, 1.0
    )