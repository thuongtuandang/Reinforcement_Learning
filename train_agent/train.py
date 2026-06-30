# train_mcts_ttt.py
import random
import numpy as np
import torch
import torch.nn.functional as F

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.config import get_config
from env.tic_tac_toe import TicTacToe
from train_agent.network import PolicyValueNet
from train_agent.mcts import MCTS, self_play_game, make_training_data

# ============================================================
#   Training loop
# ============================================================

def train_alpha_zero_ttt(preset="small"):
    cfg = get_config(preset)  # should define BOARD_SIZE=3, WIN_LENGTH=3 etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TicTacToe(cfg)
    net = PolicyValueNet(cfg).to(device)

    lr = getattr(cfg, "LEARNING_RATE", 1e-3)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    mcts = MCTS(net, cfg, device=device)

    replay_buffer = []
    max_buffer_size = 10000
    batch_size = 64
    train_steps_per_game = 4

    num_episodes = getattr(cfg, "NUM_EPISODES_MCTS", 5000)
    print_every = getattr(cfg, "PRINT_EVERY", 100)

    print(f"Training AlphaZero-style Tic-Tac-Toe ({cfg.BOARD_SIZE}x{cfg.BOARD_SIZE})")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}, MCTS sims: {mcts.num_simulations}")

    for episode in range(1, num_episodes + 1):
        # ----- Self-play game -----
        states, pis, players, winner = self_play_game(env, mcts, cfg)
        game_data = make_training_data(states, pis, players, winner, cfg)

        replay_buffer.extend(game_data)
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        # ----- Train on minibatches -----
        if len(replay_buffer) >= batch_size:
            for _ in range(train_steps_per_game):
                batch = random.sample(replay_buffer, batch_size)
                obs_batch = torch.from_numpy(
                    np.stack([d[0] for d in batch])
                ).float().to(device)
                pi_batch = torch.from_numpy(
                    np.stack([d[1] for d in batch])
                ).float().to(device)
                z_batch = torch.tensor(
                    [d[2] for d in batch]
                ).float().to(device)

                logits, values = net(obs_batch)
                log_probs = torch.log_softmax(logits, dim=1)

                policy_loss = -(pi_batch * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(values, z_batch)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % print_every == 0:
            print(f"[Episode {episode}] last game winner: {winner}, "
                  f"buffer size: {len(replay_buffer)}")

    # ----- Save model -----
    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), f"{cfg.MODEL_DIR}/{cfg.MODEL_NAME}")
    print("\nModel saved to models/mcts_ttt_cnn.pth\n")


if __name__ == "__main__":
    train_alpha_zero_ttt("medium")