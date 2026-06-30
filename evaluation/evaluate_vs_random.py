# evaluate_random.py

import torch
import numpy as np

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.tic_tac_toe import TicTacToe
from env.config import get_config
import env.observation_encoding as observation_encoding
from train_agent.mcts import MCTS
from train_agent.network import PolicyValueNet

# ------------------------------------------------------------
#   Random player policy
# ------------------------------------------------------------
def random_action(board):
    legal = np.where(board == 0)[0]
    return int(np.random.choice(legal))


# ------------------------------------------------------------
#   MCTS Agent vs Random
# ------------------------------------------------------------
def play_game_mcts_vs_random(env, mcts, cfg, device, agent_first=True):
    """
    Play one game: MCTS agent vs random player.

    agent_first = True  -> agent is X (player +1, moves first)
    agent_first = False -> agent is O (player -1, moves second)

    Returns:
        winner: +1 (X), -1 (O), or 0 (draw)
    """
    env.reset()

    # Most likely env.current_player starts as +1 (X).
    # If agent should NOT go first, we let the random player move first.
    # We don't need to manually flip current_player; we just decide
    # who acts when based on agent_first and current_player.
    done = False
    winner = None

    while not done:
        board = env.board.copy()
        player = env.current_player  # +1 (X) or -1 (O)

        # Decide who moves: agent or random
        if agent_first:
            # Agent is X
            is_agent_turn = (player == 1)
        else:
            # Agent is O
            is_agent_turn = (player == -1)

        if is_agent_turn:
            # Agent move using MCTS (no exploration)
            pi = mcts.run(board, player, add_dirichlet=False)
            action = int(np.argmax(pi))
        else:
            # Random move
            action = random_action(board)

        _, done, winner = env.step(action)

    return winner


# ------------------------------------------------------------
#   Raw NN vs Random (no MCTS)
# ------------------------------------------------------------
def play_game_nn_vs_random(env, net, cfg, device, agent_first=True):
    """
    Play one game: raw NN policy vs random, no MCTS.

    agent_first = True  -> NN is X
    agent_first = False -> NN is O

    Returns:
        winner: +1, -1, or 0
    """
    env.reset()
    done = False
    winner = None

    while not done:
        board = env.board.copy()
        player = env.current_player

        if agent_first:
            is_agent_turn = (player == 1)
        else:
            is_agent_turn = (player == -1)

        if is_agent_turn:
            # Raw NN policy: argmax over legal moves
            obs = observation_encoding.encode_obs(board, player, cfg)
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            net.eval()
            with torch.no_grad():
                logits, _ = net(obs_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            valid_mask = (board == 0)
            probs = probs * valid_mask
            s = probs.sum()

            if s <= 0:
                legal = np.where(valid_mask)[0]
                action = int(np.random.choice(legal))
            else:
                probs /= s
                action = int(np.argmax(probs))
        else:
            # Random
            action = random_action(board)

        _, done, winner = env.step(action)

    return winner


# ------------------------------------------------------------
#   Evaluation Runner
# ------------------------------------------------------------
def evaluate_against_random(
    model_path="models/trained_agent.pth",
    preset="medium",
    num_games=200,
):
    cfg = get_config(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained network
    net = PolicyValueNet(cfg).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    env = TicTacToe(cfg)
    mcts = MCTS(net, cfg, device=device)

    # --------------------------------------------------------
    #   MCTS Agent as X (first player)
    # --------------------------------------------------------
    x_wins = o_wins = draws = 0
    for _ in range(num_games):
        winner = play_game_mcts_vs_random(env, mcts, cfg, device, agent_first=True)
        if winner == 1:
            x_wins += 1
        elif winner == -1:
            o_wins += 1
        else:
            draws += 1

    print("\n=== MCTS Agent (X) vs Random (O) ===")
    print(f"Agent wins as X : {x_wins}")
    print(f"Random wins as O: {o_wins}")
    print(f"Draws           : {draws}")
    print(f"Agent win rate as X: {x_wins / num_games:.3f}")

    # --------------------------------------------------------
    #   MCTS Agent as O (second player)
    # --------------------------------------------------------
    x_wins = o_wins = draws = 0
    for _ in range(num_games):
        winner = play_game_mcts_vs_random(env, mcts, cfg, device, agent_first=False)
        if winner == 1:
            x_wins += 1
        elif winner == -1:
            o_wins += 1
        else:
            draws += 1

    print("\n=== Random (X) vs MCTS Agent (O) ===")
    print(f"Random wins as X: {x_wins}")
    print(f"Agent wins as O : {o_wins}")
    print(f"Draws           : {draws}")
    print(f"Agent win rate as O: {o_wins / num_games:.3f}")

    # --------------------------------------------------------
    #   Raw NN vs Random (optional sanity check)
    # --------------------------------------------------------
    print("\n[Raw NN vs Random]")

    x_wins = o_wins = draws = 0
    for _ in range(num_games):
        winner = play_game_nn_vs_random(env, net, cfg, device, agent_first=True)
        if winner == 1:
            x_wins += 1
        elif winner == -1:
            o_wins += 1
        else:
            draws += 1

    print("\n=== Raw NN (X) vs Random (O) ===")
    print(f"NN wins : {x_wins}")
    print(f"Random wins: {o_wins}")
    print(f"Draws   : {draws}")
    print(f"NN win rate as X: {x_wins / num_games:.3f}")


if __name__ == "__main__":
    evaluate_against_random()
