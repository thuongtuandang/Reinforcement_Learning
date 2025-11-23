import numpy as np
import torch
import time

from gridworld import GridWorld, GridConfig
from train_rl import PolicyNet

def test_policy(model_path="models/policy.pt", num_episodes=5, delay=0.5):
    """
    Test the trained policy on random grids with visualization.
    
    Args:
        model_path: Path to the saved policy model
        num_episodes: Number of test episodes to run
        delay: Delay in seconds between steps for visualization
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load policy
    policy = PolicyNet(num_actions=4).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    print(f"Loaded policy from {model_path}")
    print(f"Testing on {num_episodes} random grids\n")
    print("=" * 50)
    
    # Environment config with LARGE rewards for easy visualization
    cfg = GridConfig(
        size=8,
        obstacle_prob=0.20,
        max_steps=64,
        step_penalty=-1.0,    # penalty for empty cell
        wall_penalty=-5.0,    # penalty for moving through wall
        goal_reward=10.0
    )
    
    # Test RNG
    test_rng = np.random.default_rng(101)
    
    # Action names for display
    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    # Statistics
    total_successes = 0
    total_steps = []
    total_rewards = []
    
    for episode in range(1, num_episodes + 1):
        print(f"\n{'=' * 50}")
        print(f"EPISODE {episode}/{num_episodes}")
        print(f"{'=' * 50}\n")
        
        # Create new random grid
        env = GridWorld(cfg=cfg, rng=test_rng)
        obs_np = env.reset()
        
        print("Initial Grid:")
        env.render()
        print(f"\nAgent: A | Goal: G | Wall: # | Empty: .")
        print(f"Start: {env.start} | Goal: {env.goal}\n")
        
        done = False
        step = 0
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Store previous position
            prev_pos = env.agent
            
            # Get action from policy (greedy with action masking)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Get valid actions and create mask
                valid_actions = env.get_valid_actions()
                action_mask = torch.zeros(1, 4, dtype=torch.bool, device=device)
                action_mask[0, valid_actions] = True
                
                # Get logits and mask invalid actions
                logits = policy.forward(obs_t)
                logits = logits.masked_fill(~action_mask, float('-inf'))
                
                # Choose best valid action
                action = torch.argmax(logits, dim=1).item()
            
            # Take step
            obs_np, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1
            
            # Check if agent moved or stayed in place
            moved = (env.agent != prev_pos)
            move_status = "✓ moved" if moved else "✗ stayed (invalid/boundary)"
            
            # Display with position update
            print(f"\nStep {step}: Action = {ACTION_NAMES[action]} | Reward = {reward:.2f} | Position = {env.agent} | {move_status}")
            env.render()
            
            if done:
                if reward > 0:
                    print(f"\nSUCCESS! Reached goal in {step} steps!")
                    total_successes += 1
                else:
                    print(f"\nFAILED! Max steps reached ({cfg.max_steps})")
                print(f"Total Episode Reward: {episode_reward:.2f}")
            else:
                # Pause between steps for visualization
                time.sleep(delay)
        
        total_steps.append(step)
        total_rewards.append(episode_reward)
    
    # Final statistics
    print(f"\n{'=' * 50}")
    print(f"FINAL STATISTICS")
    print(f"{'=' * 50}")
    print(f"Success Rate: {total_successes}/{num_episodes} ({100*total_successes/num_episodes:.1f}%)")
    print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    if total_successes > 0:
        # Filter successful episodes (reward > 0)
        successful_indices = [i for i in range(num_episodes) if total_rewards[i] > 0]
        if successful_indices:
            successful_steps = [total_steps[i] for i in successful_indices]
            print(f"Average Steps (successful episodes only): {np.mean(successful_steps):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained RL policy on gridworld")
    parser.add_argument("--model", type=str, default="models/policy.pt", 
                       help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of test episodes")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between steps in seconds")
    
    args = parser.parse_args()
    
    test_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        delay=args.delay
    )