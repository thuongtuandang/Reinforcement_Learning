# train_with_checkpoints.py
"""
Modified training script that saves checkpoints every 10k episodes.
Just add these lines to your existing train.py
"""

# Add this after line 543 in your train.py (after the print_every block):

"""
        # ----- Save checkpoints every 10k episodes -----
        if episode % 10000 == 0:
            checkpoint_path = f"{cfg.MODEL_DIR}/checkpoint_{episode}.pth"
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
"""

# Full modified training function for reference:
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.config import get_config
from env.tic_tac_toe import TicTacToe
from train_agent.mcts import MCTS, self_play_game, make_training_data
from train_agent.network import PolicyValueNet

def train_with_checkpoints(preset="small", checkpoint_every=10000):
    """
    Train with checkpoint saving every N episodes.
    
    Args:
        preset: config preset name
        checkpoint_every: save checkpoint every N episodes (default 10000)
    """
    cfg = get_config(preset)
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

    num_episodes = getattr(cfg, "NUM_EPISODES_MCTS", 50000)
    print_every = getattr(cfg, "PRINT_EVERY", 100)

    print("="*60)
    print(f"Training AlphaZero-style Tic-Tac-Toe ({cfg.BOARD_SIZE}x{cfg.BOARD_SIZE})")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"MCTS simulations: {mcts.num_simulations}")
    print(f"Checkpoints every: {checkpoint_every} episodes")
    print("="*60)

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
            print(f"[Episode {episode}/{num_episodes}] last game winner: {winner}, "
                  f"buffer size: {len(replay_buffer)}")

        # ----- Save checkpoints -----
        if episode % checkpoint_every == 0:
            os.makedirs(cfg.MODEL_DIR, exist_ok=True)
            checkpoint_path = f"{cfg.MODEL_DIR}/checkpoint_{episode}.pth"
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # ----- Save final model -----
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    final_path = f"{cfg.MODEL_DIR}/{cfg.MODEL_NAME}"
    torch.save(net.state_dict(), final_path)
    print(f"\nâœ… Final model saved: {final_path}\n")

    return net


if __name__ == "__main__":
    # Example: train with default settings
    train_with_checkpoints(preset="small", checkpoint_every=10000)