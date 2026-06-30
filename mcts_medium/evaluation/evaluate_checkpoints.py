# evaluate_checkpoint.py
"""
Test a specific checkpoint against minimax.
Usage: python test_checkpoint.py checkpoint_10000.pth
"""

import sys
import os
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.config import get_config
from train_agent.mcts import MCTS
from train_agent.network import PolicyValueNet
import utils.minimax_limited as minimax_limited


def sample_action(pi, temperature=0.7):
    """Sample action from policy with temperature"""
    if temperature == 0:
        return int(np.argmax(pi))
    
    pi_temp = np.power(pi, 1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum()
    
    return int(np.random.choice(len(pi_temp), p=pi_temp))


def play_game(agent_mcts, agent_player, board_size, win_length, 
              agent_temperature, device):
    """Play one game: Agent vs Minimax"""
    board = np.zeros(board_size * board_size, dtype=np.float32)
    current_player = 1  # X always starts
    
    while True:
        winner = minimax_limited.check_winner(board, board_size, win_length)
        if winner is not None:
            return winner
        
        if current_player == agent_player:
            # Agent's turn
            pi = agent_mcts.run(board, current_player, add_dirichlet=False)
            action = sample_action(pi, temperature=agent_temperature)
        else:
            # Minimax's turn
            action = minimax_limited.get_best_move(board, current_player, 
                                          board_size, win_length)
        
        board[action] = current_player
        current_player = -current_player


def test_checkpoint(checkpoint_path, preset="small", n_games=1000, 
                   agent_temperature=0.7):
    """Test a checkpoint against minimax"""
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    cfg = get_config(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    net = PolicyValueNet(cfg).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    
    agent_mcts = MCTS(net, cfg, device=device)
    
    board_size = cfg.BOARD_SIZE
    win_length = cfg.WIN_LENGTH
    
    print("="*60)
    print(f"Testing: {os.path.basename(checkpoint_path)}")
    print(f"Board: {board_size}×{board_size}, {win_length}-in-a-row")
    print(f"Agent temperature: {agent_temperature}")
    print(f"Games: {n_games}")
    print("="*60)
    
    results = {
        'agent_wins': 0,
        'minimax_wins': 0,
        'draws': 0,
        'as_x_wins': 0,
        'as_x_losses': 0,
        'as_x_draws': 0,
        'as_o_wins': 0,
        'as_o_losses': 0,
        'as_o_draws': 0
    }
    
    # Play games
    for i in range(n_games):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{n_games} games...")
        
        if i < n_games // 2:
            # Agent as X
            winner = play_game(agent_mcts, agent_player=1,
                             board_size=board_size, win_length=win_length,
                             agent_temperature=agent_temperature, device=device)
            
            if winner == 1:
                results['agent_wins'] += 1
                results['as_x_wins'] += 1
            elif winner == -1:
                results['minimax_wins'] += 1
                results['as_x_losses'] += 1
            else:
                results['draws'] += 1
                results['as_x_draws'] += 1
        else:
            # Agent as O
            winner = play_game(agent_mcts, agent_player=-1,
                             board_size=board_size, win_length=win_length,
                             agent_temperature=agent_temperature, device=device)
            
            if winner == -1:
                results['agent_wins'] += 1
                results['as_o_wins'] += 1
            elif winner == 1:
                results['minimax_wins'] += 1
                results['as_o_losses'] += 1
            else:
                results['draws'] += 1
                results['as_o_draws'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Overall:")
    print(f"  Agent wins:   {results['agent_wins']:4d} ({results['agent_wins']/n_games*100:5.1f}%)")
    print(f"  Minimax wins: {results['minimax_wins']:4d} ({results['minimax_wins']/n_games*100:5.1f}%)")
    print(f"  Draws:        {results['draws']:4d} ({results['draws']/n_games*100:5.1f}%)")
    
    print(f"\nAgent as X (first player):")
    print(f"  Wins:   {results['as_x_wins']:4d} ({results['as_x_wins']/(n_games//2)*100:5.1f}%)")
    print(f"  Losses: {results['as_x_losses']:4d} ({results['as_x_losses']/(n_games//2)*100:5.1f}%)")
    print(f"  Draws:  {results['as_x_draws']:4d} ({results['as_x_draws']/(n_games//2)*100:5.1f}%)")
    
    print(f"\nAgent as O (second player):")
    print(f"  Wins:   {results['as_o_wins']:4d} ({results['as_o_wins']/(n_games//2)*100:5.1f}%)")
    print(f"  Losses: {results['as_o_losses']:4d} ({results['as_o_losses']/(n_games//2)*100:5.1f}%)")
    print(f"  Draws:  {results['as_o_draws']:4d} ({results['as_o_draws']/(n_games//2)*100:5.1f}%)")
    print("="*60)
    
    loss_rate = results['minimax_wins'] / n_games
    loss_as_x = results['as_x_losses'] / (n_games // 2)
    loss_as_o = results['as_o_losses'] / (n_games // 2)
    
    print(f"\nSummary:")
    print(f"  Overall loss rate: {loss_rate:.1%}")
    print(f"  Loss as X: {loss_as_x:.1%}")
    print(f"  Loss as O: {loss_as_o:.1%}")
    
    if loss_rate == 0:
        print("\n✅ PERFECT! Nash equilibrium reached!")
    elif loss_rate < 0.05:
        print("\n✅ Nearly optimal!")
    elif loss_rate < 0.20:
        print("\n⚠️  Good but room for improvement")
    else:
        print("\n❌ Significant weaknesses remain")
    
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Usage examples:
    
    # From command line: python test_checkpoint.py models/checkpoint_10000.pth
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        test_checkpoint(checkpoint_path, preset="small", n_games=1000, 
                       agent_temperature=0.7)
    else:
        # Default: test checkpoint_10000.pth if it exists
        checkpoint_path = "models/checkpoint_50000.pth"
        
        if os.path.exists(checkpoint_path):
            print("Testing default checkpoint: models/checkpoint_60000.pth\n")
            test_checkpoint(checkpoint_path, preset="small", n_games=1000,
                           agent_temperature=0.0)
        else:
            print("Usage: python test_checkpoint.py <checkpoint_path>")
            print("\nExample:")
            print("  python test_checkpoint.py models/checkpoint_10000.pth")
            print("  python test_checkpoint.py models/checkpoint_20000.pth")
            print("\nOr edit the script to change default checkpoint.")