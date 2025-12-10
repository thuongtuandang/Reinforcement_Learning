import os
import numpy as np
import torch

from gridworld import GridWorld, GridConfig
from train_rl import PolicyNet

def evaluate_policy(model_path="models/policy.pt", num_episodes=1000, seed=1000):
    """
    Evaluate trained policy on new, unseen grids.
    
    Args:
        model_path: Path to saved policy model
        num_episodes: Number of test episodes
        seed: Random seed for generating test grids
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load policy
    policy = PolicyNet(num_actions=4).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    print(f"Loaded policy from {model_path}")
    print(f"Evaluating on {num_episodes} new grids (seed={seed})\n")
    
    # Environment config (same as training)
    cfg = GridConfig(
        size=8,
        obstacle_prob=0.20,
        max_steps=64,
        step_penalty=-0.1,
        wall_penalty=-0.5,
        goal_reward=1.0
    )
    
    # Evaluation RNG (different from training seed)
    eval_rng = np.random.default_rng(seed)
    
    # Metrics
    successes = []
    path_lengths = []
    wall_hits = []
    
    # Run episodes
    for ep in range(num_episodes):
        env = GridWorld(cfg=cfg, rng=eval_rng)
        obs_np = env.reset()
        done = False
        steps = 0
        walls = 0
        
        while not done:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy.forward(obs_t)
                action = torch.argmax(logits, dim=1).item()  # Greedy
            
            obs_np, r, done, _ = env.step(action)
            steps += 1
            
            # Count wall hits
            if abs(r - cfg.wall_penalty) < 1e-6:
                walls += 1
        
        # Check success
        success = (env.agent == env.goal)
        successes.append(success)
        
        if success:
            path_lengths.append(steps)
            wall_hits.append(walls)
    
    # Print results
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success rate: {np.mean(successes):.2%}")
    
    if path_lengths:
        print(f"Average path length: {np.mean(path_lengths):.1f} steps")
        print(f"Average wall hits: {np.mean(wall_hits):.1f}")
    else:
        print("No successful episodes")
    
    return {
        'success_rate': np.mean(successes),
        'path_lengths': path_lengths,
        'wall_hits': wall_hits
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy")
    parser.add_argument("--model", type=str, default="models/policy.pt",
                       help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000,
                       help="Random seed for test grids")
    
    args = parser.parse_args()
    
    evaluate_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        seed=args.seed
    )