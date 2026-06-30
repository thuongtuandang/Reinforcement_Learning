# evaluate_with_game_display.py
"""
Evaluate agent vs minimax and display sample games.
Shows 5 games as X and 5 games as O.
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


def sample_action(pi, temperature=0.7):
    """Sample action from policy with temperature"""
    if temperature == 0:
        return int(np.argmax(pi))
    
    pi_temp = np.power(pi, 1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum()
    
    return int(np.random.choice(len(pi_temp), p=pi_temp))


def print_board(board, board_size):
    """Print board state"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    b2d = board.reshape(board_size, board_size)
    for r in range(board_size):
        row_str = ' | '.join([symbols[int(b2d[r, c])] for c in range(board_size)])
        print('  ' + row_str)


def play_game_with_display(agent_mcts, agent_player, board_size, win_length,
                           agent_temperature, device, game_num, display=False):
    """
    Play one game and optionally display it.
    
    Returns:
        winner: 1, -1, or 0
        moves: list of (player, action) tuples
    """
    board = np.zeros(board_size * board_size, dtype=np.float32)
    current_player = 1  # X always starts
    moves = []
    
    if display:
        agent_symbol = 'X' if agent_player == 1 else 'O'
        minimax_symbol = 'O' if agent_player == 1 else 'X'
        print(f"\n{'='*60}")
        print(f"Game {game_num}: Agent is {agent_symbol}, Minimax is {minimax_symbol}")
        print('='*60)
        print("Starting position:")
        print_board(board, board_size)
        print()
    
    move_num = 0
    while True:
        winner = minimax_limited.check_winner(board, board_size, win_length)
        if winner is not None:
            if display:
                if winner == 0:
                    print("Result: Draw")
                elif winner == agent_player:
                    print(f"Result: Agent ({agent_symbol}) wins!")
                else:
                    print(f"Result: Minimax ({minimax_symbol}) wins!")
            return winner, moves
        
        if current_player == agent_player:
            # Agent's turn
            pi = agent_mcts.run(board, current_player, add_dirichlet=False)
            action = sample_action(pi, temperature=agent_temperature)
            player_name = f"Agent ({agent_symbol})"
        else:
            # Minimax's turn
            action = minimax_limited.get_best_move(board, 
                                                   current_player,
                                                   depth=2, 
                                                   board_size=board_size, 
                                                   win_length=win_length)
            player_name = f"Minimax ({minimax_symbol})"
        
        board[action] = current_player
        moves.append((current_player, action))
        move_num += 1
        
        if display:
            row = action // board_size
            col = action % board_size
            print(f"Move {move_num}: {player_name} plays ({row},{col}) [pos {action}]")
            print_board(board, board_size)
            print()
        
        current_player = -current_player


def evaluate_with_display(model_path, preset="medium", n_games=1000,
                          agent_temperature=0.7, n_display=5):
    """
    Evaluate agent vs minimax and display sample games.
    
    Args:
        model_path: path to model
        preset: config preset
        n_games: total games to play
        agent_temperature: sampling temperature
        n_display: number of games to display for each side (X and O)
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
    
    print("="*60)
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"Board: {board_size}×{board_size}, {win_length}-in-a-row")
    print(f"Agent temperature: {agent_temperature}")
    print(f"Total games: {n_games}")
    print(f"Displaying: {n_display} games as X, {n_display} games as O")
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
    
    displayed_as_x = 0
    displayed_as_o = 0
    
    # Play games
    for i in range(n_games):
        if i % 100 == 0 and i > 0:
            print(f"\nProgress: {i}/{n_games} games...")
        
        if i < n_games // 2:
            # Agent as X
            display = (displayed_as_x < n_display)
            if display:
                displayed_as_x += 1
            
            winner, moves = play_game_with_display(
                agent_mcts, agent_player=1,
                board_size=board_size, win_length=win_length,
                agent_temperature=agent_temperature, device=device,
                game_num=displayed_as_x if display else i+1,
                display=display
            )
            
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
            display = (displayed_as_o < n_display)
            if display:
                displayed_as_o += 1
            
            winner, moves = play_game_with_display(
                agent_mcts, agent_player=-1,
                board_size=board_size, win_length=win_length,
                agent_temperature=agent_temperature, device=device,
                game_num=displayed_as_o if display else i+1,
                display=display
            )
            
            if winner == -1:
                results['agent_wins'] += 1
                results['as_o_wins'] += 1
            elif winner == 1:
                results['minimax_wins'] += 1
                results['as_o_losses'] += 1
            else:
                results['draws'] += 1
                results['as_o_draws'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS:")
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
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    evaluate_with_display(
        model_path="saved_models/trained_agent_medium_final.pth",
        preset="medium",
        n_games=4,
        agent_temperature=0.0,  # Change to 0.0 for deterministic
        n_display=5  # Show 5 games as X and 5 as O
    )