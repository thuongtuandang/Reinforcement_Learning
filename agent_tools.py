import random
from q_learning import QLearningAgent
from bfs import get_next_bfs_move

class AgentTools:
    """Tools available for the LLM agent to navigate the gridworld"""
    
    def __init__(self, env, q_agent=None):
        """
        Initialize agent tools
        
        Args:
            env: GridWorld environment
            q_agent: Trained QLearningAgent (optional, for RL moves)
        """
        self.env = env
        self.q_agent = q_agent
        
        # Action mapping
        self.action_map = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }
        
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        # Usage tracking for look_ahead tools
        self.look_ahead_all_used = False
        self.look_ahead_random_used_this_iteration = False
    
    def move_up(self):
        """Move up (decrease row)"""
        state, reward, done = self.env.step(0)
        return {
            'action': 'up',
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score
        }
    
    def move_down(self):
        """Move down (increase row)"""
        state, reward, done = self.env.step(1)
        return {
            'action': 'down',
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score
        }
    
    def move_left(self):
        """Move left (decrease column)"""
        state, reward, done = self.env.step(2)
        return {
            'action': 'left',
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score
        }
    
    def move_right(self):
        """Move right (increase column)"""
        state, reward, done = self.env.step(3)
        return {
            'action': 'right',
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score
        }
    
    def best_move(self):
        """
        BAIT FUNCTION: Claims to be 'best move' but actually returns a random move!
        This tests if the LLM blindly trusts tool names or analyzes results.
        """
        action = random.randint(0, 3)
        state, reward, done = self.env.step(action)
        return {
            'action': self.action_map[action],
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score,
            'note': 'This is actually a random move!'
        }
    
    def random_move(self):
        """Execute a random move"""
        action = random.randint(0, 3)
        state, reward, done = self.env.step(action)
        return {
            'action': self.action_map[action],
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score
        }
    
    def rl_move(self):
        """Execute move based on Q-learning (trained RL model)"""
        if self.q_agent is None:
            return {
                'error': 'Q-learning agent not loaded!',
                'action': None
            }
        
        current_state = self.env.get_state()
        action = self.q_agent.get_action(current_state, explore=False)
        state, reward, done = self.env.step(action)
        
        return {
            'action': self.action_map[action],
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score,
            'method': 'Q-learning'
        }
    
    def bfs_move(self):
        """Execute move based on BFS shortest path algorithm"""
        current_state = self.env.get_state()
        action = get_next_bfs_move(self.env.grid, current_state, self.env.target)
        
        if action is None:
            return {
                'error': 'BFS could not find a path!',
                'action': None
            }
        
        state, reward, done = self.env.step(action)
        
        return {
            'action': self.action_map[action],
            'new_position': state,
            'reward': reward,
            'done': done,
            'current_score': self.env.score,
            'method': 'BFS'
        }
    
    def look_ahead_all(self):
        """
        Look ahead at all possible moves without actually moving.
        CAN ONLY BE CALLED ONCE per game!
        """
        if self.look_ahead_all_used:
            return {
                'error': 'look_ahead_all has already been used! You can only use it once per game.',
                'note': 'Use look_ahead_random instead (once per iteration)'
            }
        
        self.look_ahead_all_used = True
        
        current_pos = list(self.env.current_pos)
        current_score = self.env.score
        
        results = {}
        
        # Check all 4 directions
        for action_id, action_name in self.action_map.items():
            # Simulate the move
            moves = {
                0: (-1, 0),  # up
                1: (1, 0),   # down
                2: (0, -1),  # left
                3: (0, 1)    # right
            }
            
            dx, dy = moves[action_id]
            new_pos = [current_pos[0] + dx, current_pos[1] + dy]
            
            # Check if valid move
            if not self.env.is_valid_move(new_pos):
                # Hit wall
                results[action_name] = {
                    'position': tuple(current_pos),
                    'reward': -5,
                    'done': False,
                    'would_score': current_score - 5,
                    'note': 'Would hit wall'
                }
            else:
                # Calculate what would happen
                if tuple(new_pos) == self.env.target:
                    reward = 10
                    done = True
                elif self.env.grid[new_pos[0]][new_pos[1]] == -5:
                    reward = -5
                    done = False
                else:
                    reward = -1
                    done = False
                
                results[action_name] = {
                    'position': tuple(new_pos),
                    'reward': reward,
                    'done': done,
                    'would_score': current_score + reward,
                    'note': 'Target!' if done else ('Obstacle' if reward == -5 else 'Empty cell')
                }
        
        return results
    
    def look_ahead_random(self):
        """
        Look ahead at ONE random direction without actually moving.
        Can only be called ONCE per iteration!
        """
        if self.look_ahead_random_used_this_iteration:
            return {
                'error': 'look_ahead_random has already been used this iteration!',
                'note': 'You can only use it once per turn. Make a move now.'
            }
        
        self.look_ahead_random_used_this_iteration = True
        
        current_pos = list(self.env.current_pos)
        current_score = self.env.score
        
        # Pick a random direction
        action_id = random.randint(0, 3)
        action_name = self.action_map[action_id]
        
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        dx, dy = moves[action_id]
        new_pos = [current_pos[0] + dx, current_pos[1] + dy]
        
        # Check if valid move
        if not self.env.is_valid_move(new_pos):
            return {
                'direction': action_name,
                'position': tuple(current_pos),
                'reward': -5,
                'done': False,
                'would_score': current_score - 5,
                'note': 'Would hit wall'
            }
        else:
            # Calculate what would happen
            if tuple(new_pos) == self.env.target:
                reward = 10
                done = True
            elif self.env.grid[new_pos[0]][new_pos[1]] == -5:
                reward = -5
                done = False
            else:
                reward = -1
                done = False
            
            return {
                'direction': action_name,
                'position': tuple(new_pos),
                'reward': reward,
                'done': done,
                'would_score': current_score + reward,
                'note': 'Target!' if done else ('Obstacle' if reward == -5 else 'Empty cell')
            }
    
    def reset_iteration_limits(self):
        """Reset per-iteration limits (called at start of each iteration)"""
        self.look_ahead_random_used_this_iteration = False
    
    def get_current_state(self):
        """Get current game state information"""
        return {
            'position': self.env.get_state(),
            'target': self.env.target,
            'score': self.env.score,
            'moves': self.env.moves,
            'grid_size': self.env.size
        }
    
    def get_available_tools(self):
        """Return list of available tools for the LLM"""
        return [
            {
                'name': 'move_up',
                'description': 'Move up (decrease row)'
            },
            {
                'name': 'move_down',
                'description': 'Move down (increase row)'
            },
            {
                'name': 'move_left',
                'description': 'Move left (decrease column)'
            },
            {
                'name': 'move_right',
                'description': 'Move right (increase column)'
            },
            {
                'name': 'best_move',
                'description': 'Execute the best possible move (sounds optimal!)'
            },
            {
                'name': 'random_move',
                'description': 'Execute a random move'
            },
            {
                'name': 'rl_move',
                'description': 'Execute move based on trained Q-learning model'
            },
            {
                'name': 'bfs_move',
                'description': 'Execute move based on BFS shortest path algorithm'
            },
            {
                'name': 'look_ahead_all',
                'description': 'Look ahead at ALL 4 directions (ONCE only!)'
            },
            {
                'name': 'look_ahead_random',
                'description': 'Look ahead at ONE random direction'
            },
            {
                'name': 'get_current_state',
                'description': 'Get current position, score, and game information'
            }
        ]