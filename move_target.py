"""
Experiment: Train Q-learning on one target, then test on different targets
This demonstrates that Q-values are target-dependent (through reward function)
"""

import numpy as np
from gridworld import GridWorld
from q_learning import QLearningAgent
import matplotlib.pyplot as plt

def visualize_policy(env, agent, title="Policy Visualization"):
    """Visualize the learned policy on the grid"""
    print(f"\n{title}")
    print("=" * 50)
    
    action_symbols = {
        0: '↑',  # up
        1: '↓',  # down
        2: '←',  # left
        3: '→'   # right
    }
    
    for i in range(env.size):
        row = []
        for j in range(env.size):
            if (i, j) == env.start:
                row.append('S')
            elif (i, j) == env.target:
                row.append('T')
            elif env.grid[i][j] == -5:
                row.append('X')
            else:
                # Get best action from Q-table
                state = (i, j)
                if state in agent.q_table:
                    best_action = np.argmax(agent.q_table[state])
                    row.append(action_symbols[best_action])
                else:
                    row.append('?')  # Unexplored
        print(' '.join(row))
    print()

def test_on_target(env, agent, target_pos, test_name):
    """Test the agent on a specific target position"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Target at: {target_pos}")
    print("=" * 60)
    
    # Change target
    old_target = env.target
    env.target = target_pos
    env.reset()
    
    # Show grid
    print("\nGrid:")
    env.render()
    
    # Run agent greedily
    state = env.get_state()
    steps = 0
    max_steps = 100
    path = [state]
    
    while steps < max_steps:
        # Get action from Q-table
        action = agent.get_action(state, explore=False)
        state, reward, done = env.step(action)
        path.append(state)
        steps += 1
        
        if done:
            print(f"\n✅ Reached target in {steps} steps!")
            print(f"Final score: {env.score}")
            break
        
        # Check if stuck in loop
        if len(path) > 10 and len(set(path[-10:])) < 3:
            print(f"\n⚠️  Agent stuck in loop after {steps} steps")
            print(f"Score: {env.score}")
            break
    
    if not done and steps >= max_steps:
        print(f"\n❌ Did not reach target in {max_steps} steps")
        print(f"Final score: {env.score}")
        print(f"Final position: {state}")
    
    # Restore original target
    env.target = old_target
    
    return {
        'success': done,
        'steps': steps,
        'score': env.score
    }

def main():
    print("=" * 60)
    print("EXPERIMENT: Moving Target Test")
    print("Train on one target, test on different targets")
    print("=" * 60)
    
    # Create environment with fixed grid
    print("\n📋 Creating training environment...")
    env = GridWorld(size=10, num_obstacles=15)
    original_target = env.target
    original_start = env.start
    
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Obstacles: 15")
    print(f"Original start: {original_start}")
    print(f"Original target: {original_target}")
    
    print("\nOriginal grid:")
    env.render()
    
    # Train Q-learning agent
    print("\n" + "=" * 60)
    print("TRAINING Q-Learning Agent")
    print("=" * 60)
    
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print(f"\nTraining for 5000 episodes on target {original_target}...")
    rewards = agent.train(env, n_episodes=5000, max_steps=200, verbose=False)
    
    # Show training results
    final_100_avg = np.mean(rewards[-100:])
    print(f"✅ Training complete!")
    print(f"Average reward (last 100 episodes): {final_100_avg:.2f}")
    print(f"States learned: {len(agent.q_table)}")
    
    # Visualize learned policy
    visualize_policy(env, agent, "Learned Policy (Original Target)")
    
    # Test on original target
    print("\n" + "=" * 60)
    print("TESTING PHASE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Original target (should work well)
    result1 = test_on_target(env, agent, original_target, "Original Target (Control)")
    results.append(("Original", result1))
    
    # Test 2: Target in opposite corner
    new_target1 = (0, env.size - 1)  # Top-right
    if new_target1 != original_start and env.grid[new_target1[0]][new_target1[1]] != -5:
        result2 = test_on_target(env, agent, new_target1, "Target at Top-Right")
        results.append(("Top-Right", result2))
    
    # Test 3: Target in center
    new_target2 = (env.size // 2, env.size // 2)  # Center
    if new_target2 != original_start and env.grid[new_target2[0]][new_target2[1]] != -5:
        result3 = test_on_target(env, agent, new_target2, "Target at Center")
        results.append(("Center", result3))
    
    # Test 4: Target near start
    new_target3 = (original_start[0] + 1, original_start[1] + 1)
    if (new_target3[0] < env.size and new_target3[1] < env.size and 
        new_target3 != original_start and env.grid[new_target3[0]][new_target3[1]] != -5):
        result4 = test_on_target(env, agent, new_target3, "Target Near Start")
        results.append(("Near Start", result4))
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nTrained on target: {original_target}")
    print("\nResults:")
    print("-" * 60)
    print(f"{'Target Location':<20} {'Success':<10} {'Steps':<10} {'Score':<10}")
    print("-" * 60)
    
    for name, result in results:
        success = "✅ Yes" if result['success'] else "❌ No"
        print(f"{name:<20} {success:<10} {result['steps']:<10} {result['score']:<10}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("If the agent performs well ONLY on the original target but")
    print("poorly on other targets, this confirms that Q-values are")
    print("target-dependent through the reward function!")
    print("=" * 60)

if __name__ == "__main__":
    main()