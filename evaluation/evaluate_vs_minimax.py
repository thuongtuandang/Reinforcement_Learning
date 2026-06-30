# evaluate_minimax.py
"""
Evaluate trained agent against minimax.

Uses:
- Agent: samples from MCTS policy with temperature
- Minimax: randomly breaks ties among optimal moves
"""
import numpy as np
import torch

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from env.config import get_config
from train_agent.mcts import MCTS
from train_agent.network import PolicyValueNet
import utils.minimax_limited as minimax_limited


def sample_action(pi, temperature=0.3):
    """
    Sample action from policy with temperature.
    
    Lower temperature = more deterministic (exploits best moves)
    Higher temperature = more exploration
    
    Args:
        pi: probability distribution over actions
        temperature: sampling temperature (0 = argmax, 1 = full stochastic)
    
    Returns:
        action index
    """
    if temperature == 0:
        return int(np.argmax(pi))
    
    # Apply temperature
    pi_temp = np.power(pi, 1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum()
    
    return int(np.random.choice(len(pi_temp), p=pi_temp))


def play_game(agent_mcts, agent_player, board_size, win_length, 
              agent_temperature=0.3, device="cpu"):
    """
    Play one game: Agent vs Minimax
    
    Args:
        agent_mcts: MCTS object for agent
        agent_player: +1 if agent is X, -1 if agent is O
        board_size: board size
        win_length: win condition
        agent_temperature: temperature for agent sampling
        device: torch device
    
    Returns:
        winner: +1, -1, or 0
    """
    board = np.zeros(board_size * board_size, dtype=np.float32)
    current_player = 1  # X always starts
    
    while True:
        winner = minimax_limited.check_winner(board, board_size, win_length)
        if winner is not None:
            return winner
        
        if current_player == agent_player:
            # Agent's turn (with temperature sampling)
            pi = agent_mcts.run(board, current_player, add_dirichlet=False)
            action = sample_action(pi, temperature=agent_temperature)
        else:
            # Minimax's turn (with random tie-breaking)
            action = minimax_limited.get_best_move(board, 
                                                   current_player,
                                                   depth=3,
                                                   board_size = board_size, 
                                                   win_length = win_length)
        
        # Make move
        board[action] = current_player
        current_player = -current_player


def evaluate_agent(model_path, preset="medium", n_games=100, 
                   agent_temperature=0.3):
    """
    Evaluate agent against minimax.
    
    Args:
        model_path: path to saved model
        preset: config preset
        n_games: number of games to play
        agent_temperature: sampling temperature for agent
    
    Returns:
        dict with results
    """
    cfg = get_config(preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    net = PolicyValueNet(cfg).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    agent_mcts = MCTS(net, cfg, device=device)
    
    board_size = cfg.BOARD_SIZE
    win_length = cfg.WIN_LENGTH
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"Board: {board_size}×{board_size}, {win_length}-in-a-row")
    print(f"Agent temperature: {agent_temperature}")
    print(f"Games: {n_games}")
    print(f"{'='*60}\n")
    
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
    
    # Play games (alternate who goes first)
    for i in range(n_games):
        if i % 100 == 0 and i > 0:
            print(f"Progress: {i}/{n_games} games...")
        
        if i < n_games // 2:
            # Agent plays as X (first)
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
            # Agent plays as O (second)
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
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
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
    print(f"{'='*60}")
    
    loss_rate = results['minimax_wins'] / n_games
    
    if loss_rate == 0:
        print("✅ PERFECT! Agent never loses - at Nash equilibrium!")
    elif loss_rate < 0.01:
        print("⚠️  Nearly optimal - very few losses")
    elif loss_rate < 0.05:
        print("⚠️  Good but has weaknesses")
    else:
        print("❌ Agent has significant weaknesses")
    
    print(f"{'='*60}\n")
    
    return results


def evaluate_all_models(models_dir="saved_models", preset="medium", 
                       n_games=1000, agent_temperature=0.3):
    """
    Evaluate all models in a directory.
    """
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"No .pth files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model(s) to evaluate\n")
    
    all_results = {}
    
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file)
        results = evaluate_agent(model_path, preset, n_games, agent_temperature)
        all_results[model_file] = results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS:")
    print("="*60)
    print(f"{'Model':<35s} | {'Overall Loss':>12s} | {'Loss as X':>10s} | {'Loss as O':>10s}")
    print("-"*60)
    for model_file, results in all_results.items():
        loss_rate = results['minimax_wins'] / n_games
        loss_as_x = results['as_x_losses'] / (n_games // 2)
        loss_as_o = results['as_o_losses'] / (n_games // 2)
        print(f"{model_file:<35s} | {loss_rate:11.1%} | {loss_as_x:9.1%} | {loss_as_o:9.1%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage:
    
    # Evaluate single model
    evaluate_agent("models/trained_agent.pth", 
                   preset="medium", n_games=1000, agent_temperature=0.0)
    
    # Evaluate all models in directory
    # evaluate_all_models("models", preset="small", 
    #                    n_games=1000, agent_temperature=0.3)