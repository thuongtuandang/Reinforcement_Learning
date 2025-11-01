from gridworld import GridWorld
from q_learning import QLearningAgent
import matplotlib.pyplot as plt
import pickle
import os

def plot_training_progress(rewards, window=100):
    """Plot training rewards over episodes"""
    # Calculate moving average
    moving_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(sum(rewards[start:i+1]) / (i - start + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(moving_avg, label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_progress.png')
    plt.close()
    print("Training plot saved to models/training_progress.png")

def save_grid(env, filepath='models/grid_config.pkl'):
    """Save the grid configuration"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    grid_data = {
        'grid': env.grid,
        'size': env.size,
        'start': env.start,
        'target': env.target
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(grid_data, f)
    
    print(f"Grid configuration saved to {filepath}")

def main():
    print("=" * 50)
    print("Training Q-Learning Agent for GridWorld")
    print("=" * 50)
    
    # Create ONE fixed grid for training
    env = GridWorld(size=10, num_obstacles=50)  # More obstacles!
    print(f"\nEnvironment: {env.size}x{env.size} grid with 50 obstacles")
    print(f"Start: {env.start}, Target: {env.target}")
    print("\nGrid layout:")
    env.render()
    
    # Save this grid configuration
    save_grid(env, 'models/grid_config.pkl')
    
    # Create agent
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,           # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent on this specific grid
    print("\nStarting training on this fixed grid...")
    rewards = agent.train(env, n_episodes=5000, max_steps=200, verbose=True)
    
    # Save model
    print("\nTraining complete!")
    agent.save('models/q_table.pkl')
    
    # Plot results
    plot_training_progress(rewards)
    
    # Test the trained agent on THE SAME grid
    print("\n" + "=" * 50)
    print("Testing trained agent on the same grid (greedy policy):")
    print("=" * 50)
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nInitial state:")
    env.render()
    
    print("\nAgent playing...")
    while steps < 100:
        action = agent.get_action(state, explore=False)  # Greedy
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            print(f"\n✅ Target reached in {steps} steps!")
            print(f"Final score: {env.score}")
            env.render()
            break
    
    if not done:
        print(f"\n❌ Did not reach target in {steps} steps")
        print(f"Final score: {env.score}")
        env.render()
    
    print("\n" + "=" * 50)
    print(f"Q-table size: {len(agent.q_table)} states learned")
    print("Saved files:")
    print("  - models/q_table.pkl (Q-learning model)")
    print("  - models/grid_config.pkl (Grid configuration)")
    print("  - models/training_progress.png (Training plot)")
    print("=" * 50)

if __name__ == "__main__":
    main()