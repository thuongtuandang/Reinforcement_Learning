from gridworld import GridWorld
from bfs import bfs_shortest_path, get_next_bfs_move, visualize_path

def test_bfs():
    print("=" * 50)
    print("Testing BFS Algorithm")
    print("=" * 50)
    
    # Create a grid
    env = GridWorld(size=10, num_obstacles=10)
    
    print("\nGrid layout:")
    env.render()
    
    print(f"\nStart: {env.start}")
    print(f"Target: {env.target}")
    
    # Find shortest path using BFS
    print("\n" + "-" * 50)
    print("Running BFS to find shortest path...")
    print("-" * 50)
    
    path = bfs_shortest_path(env.grid, env.start, env.target)
    
    if path:
        print(f"\n✅ Path found!")
        visualize_path(env.grid, path, env.start, env.target)
        print(f"\nOptimal path: {path}")
    else:
        print("\n❌ No path found!")
        return
    
    # Test following the BFS path
    print("\n" + "=" * 50)
    print("Testing BFS navigation (following the path):")
    print("=" * 50)
    
    env.reset()
    state = env.get_state()
    steps = 0
    
    print("\nStarting navigation...")
    
    while steps < 100:
        # Get next BFS move
        action = get_next_bfs_move(env.grid, state, env.target)
        
        if action is None:
            print(f"\n❌ BFS returned no action at step {steps}")
            break
        
        # Take action
        state, reward, done = env.step(action)
        steps += 1
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {steps}: {action_names[action]} -> Position: {state}, Reward: {reward}")
        
        if done:
            print(f"\n✅ Target reached in {steps} steps!")
            print(f"Final score: {env.score}")
            print("\nFinal state:")
            env.render()
            break
    
    if not done:
        print(f"\n❌ Did not reach target in {steps} steps")
        env.render()
    
    # Compare with optimal path length
    if path and done:
        print("\n" + "-" * 50)
        print(f"BFS optimal path length: {len(path)} steps")
        print(f"Actual steps taken: {steps} steps")
        if steps == len(path):
            print("✅ Perfect! Followed optimal path exactly.")
        else:
            print("⚠️  Took more steps than optimal (this shouldn't happen with BFS)")

if __name__ == "__main__":
    test_bfs()