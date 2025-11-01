import numpy as np
import random

class GridWorld:
    def __init__(self, size=10, num_obstacles=10):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.target = (size - 1, size - 1)
        self.current_pos = list(self.start)
        self.score = 0
        self.moves = 0
        
        # Place obstacles randomly (avoid start and target)
        obstacles_placed = 0
        while obstacles_placed < num_obstacles:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if (x, y) not in [self.start, self.target] and self.grid[x][y] == 0:
                self.grid[x][y] = -5  # X cell penalty
                obstacles_placed += 1
    
    def reset(self):
        """Reset the game to initial state"""
        self.current_pos = list(self.start)
        self.score = 0
        self.moves = 0
        return self.get_state()  # Return tuple instead of list
    
    def get_state(self):
        """Get current state as tuple (for Q-learning)"""
        return tuple(self.current_pos)
    
    def is_valid_move(self, pos):
        """Check if position is within grid bounds"""
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        dx, dy = moves[action]
        new_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy]
        
        # Check if valid move
        if not self.is_valid_move(new_pos):
            # Hit wall - stay in place, penalty
            reward = -5
            self.moves += 1
        else:
            # Move to new position
            self.current_pos = new_pos
            self.moves += 1
            
            # Calculate reward
            if tuple(self.current_pos) == self.target:
                reward = 10
            elif self.grid[self.current_pos[0]][self.current_pos[1]] == -5:
                reward = -5  # Obstacle
            else:
                reward = -1  # Normal move
        
        self.score += reward
        done = tuple(self.current_pos) == self.target
        
        return self.get_state(), reward, done
    
    def render(self):
        """Display the grid with current position"""
        display = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.current_pos:
                    row.append('A')  # Agent
                elif (i, j) == self.target:
                    row.append('T')  # Target
                elif self.grid[i][j] == -5:
                    row.append('X')  # Obstacle
                else:
                    row.append('.')  # Empty
            display.append(' '.join(row))
        
        print('\n'.join(display))
        print(f"\nPosition: {self.current_pos}, Score: {self.score}, Moves: {self.moves}")
    
    def get_grid_string(self):
        """Return grid as string for LLM"""
        lines = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.current_pos:
                    row.append('A')
                elif (i, j) == self.target:
                    row.append('T')
                elif self.grid[i][j] == -5:
                    row.append('X')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        return '\n'.join(lines)