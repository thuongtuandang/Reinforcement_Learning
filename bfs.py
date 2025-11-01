from collections import deque

def bfs_shortest_path(grid, start, target):
    """
    Find shortest path from start to target using BFS
    
    Args:
        grid: 2D numpy array (obstacles have value -5)
        start: tuple (x, y) starting position
        target: tuple (x, y) target position
    
    Returns:
        path: list of positions [(x1,y1), (x2,y2), ...] or None if no path
    """
    size = len(grid)
    
    # BFS queue: (position, path_to_position)
    queue = deque([(start, [start])])
    visited = {start}
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (x, y), path = queue.popleft()
        
        # Check if we reached target
        if (x, y) == target:
            return path
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if valid move
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                queue.append(((nx, ny), new_path))
    
    # No path found
    return None

def get_next_bfs_move(grid, current_pos, target):
    """
    Get the next move based on BFS shortest path
    
    Args:
        grid: 2D numpy array
        current_pos: tuple (x, y) current position
        target: tuple (x, y) target position
    
    Returns:
        action: 0=up, 1=down, 2=left, 3=right, or None if no path
    """
    path = bfs_shortest_path(grid, current_pos, target)
    
    if path is None or len(path) < 2:
        return None
    
    # Get next position in path
    next_pos = path[1]
    
    # Convert to action
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    # Map direction to action
    direction_to_action = {
        (-1, 0): 0,  # up
        (1, 0): 1,   # down
        (0, -1): 2,  # left
        (0, 1): 3    # right
    }
    
    return direction_to_action.get((dx, dy))

def visualize_path(grid, path, start, target):
    """
    Print grid with path visualized
    
    Args:
        grid: 2D numpy array
        path: list of positions in the path
        start: starting position
        target: target position
    """
    size = len(grid)
    path_set = set(path) if path else set()
    
    print("\nBFS Shortest Path:")
    for i in range(size):
        row = []
        for j in range(size):
            if (i, j) == start:
                row.append('S')  # Start
            elif (i, j) == target:
                row.append('T')  # Target
            elif (i, j) in path_set:
                row.append('*')  # Path
            elif grid[i][j] == -5:
                row.append('X')  # Obstacle
            else:
                row.append('.')  # Empty
        print(' '.join(row))
    
    if path:
        print(f"\nPath length: {len(path)} steps")
    else:
        print("\nNo path found!")