import numpy as np
from collections import deque
from dataclasses import dataclass

# Actions: 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT
MOVE_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

@dataclass
class GridConfig:
    size: int = 8
    obstacle_prob: float = 0.20
    max_steps: int = 64
    step_penalty: float = -0.01     # penalty for moving to empty cell
    wall_penalty: float = -0.1      # penalty for moving through wall
    goal_reward: float = 1.0

class GridWorld:
    """
    Minimal gridworld:
      - observation: 3×H×W tensor-like (numpy float32): [walls, agent, goal]
      - rewards: +10 at goal, -1 for empty cell, -5 for wall
      - agent CAN move through walls (with penalty)
      - episode ends at goal or after max_steps
    """

    def __init__(self, cfg: GridConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or GridConfig()
        self.rng = rng or np.random.default_rng()
        self.size = self.cfg.size
        self.t = 0
        self.walls = None
        self.start = None
        self.goal = None
        self.agent = None

    # ---------- public API ----------
    def reset(self):
        self._make_connected_grid()
        self.agent = self.start
        self.t = 0
        return self._obs()

    def step(self, action: int):
        self.t += 1
        ax, ay = self.agent
        dx, dy = MOVE_DELTAS[action]
        nx, ny = ax + dx, ay + dy

        # Check move validity and assign rewards
        if not self._in_bounds(nx, ny):
            # Out of bounds - stay in place with penalty
            self.agent = (ax, ay)
            reward = self.cfg.step_penalty
        elif self.walls[nx, ny]:
            # Move through wall with heavy penalty
            self.agent = (nx, ny)
            reward = self.cfg.wall_penalty
        else:
            # Normal move to empty cell
            self.agent = (nx, ny)
            reward = self.cfg.step_penalty

        done = False

        # Check if reached goal (overrides other rewards)
        if self.agent == self.goal:
            reward = self.cfg.goal_reward
            done = True

        if self.t >= self.cfg.max_steps:
            done = True

        return self._obs(), float(reward), bool(done), {}
    
    def get_valid_actions(self):
        """Return list of valid actions (not out of bounds)."""
        ax, ay = self.agent
        valid = []
        for action, (dx, dy) in enumerate(MOVE_DELTAS):
            nx, ny = ax + dx, ay + dy
            if self._in_bounds(nx, ny):
                valid.append(action)
        return valid

    def render(self):
        """Simple ASCII render for debugging."""
        H = self.size
        grid = np.full((H, H), '.', dtype=str)
        grid[self.walls] = '#'
        gx, gy = self.goal
        grid[gx, gy] = 'G'
        ax, ay = self.agent
        grid[ax, ay] = 'A'
        print("\n".join("".join(row) for row in grid))

    # ---------- helpers ----------
    def _obs(self):
        H = self.size
        walls = self.walls.astype(np.float32)
        agent = np.zeros_like(walls, dtype=np.float32)
        goal = np.zeros_like(walls, dtype=np.float32)
        ax, ay = self.agent
        gx, gy = self.goal
        agent[ax, ay] = 1.0
        goal[gx, gy] = 1.0
        obs = np.stack([walls, agent, goal], axis=0)  # (3, H, H)
        return obs

    def _make_connected_grid(self):
        """Sample random walls/start/goal until start→goal is reachable."""
        H = self.size
        max_retries = 100
        
        for attempt in range(max_retries):
            walls = self.rng.random((H, H)) < self.cfg.obstacle_prob
            
            # Get two different free cells for start and goal
            free_cells = np.argwhere(~walls)
            if len(free_cells) < 2:
                continue
            
            # Sample two distinct cells
            indices = self.rng.choice(len(free_cells), size=2, replace=False)
            start = tuple(free_cells[indices[0]])
            goal = tuple(free_cells[indices[1]])
            
            # Ensure they remain free
            walls[start] = False
            walls[goal] = False
            
            if self._reachable(walls, start, goal):
                self.walls = walls
                self.start = start
                self.goal = goal
                return
        
        raise RuntimeError(f"Could not generate connected grid after {max_retries} attempts. "
                         f"Try reducing obstacle_prob (current: {self.cfg.obstacle_prob})")

    def _reachable(self, walls, start, goal):
        H = walls.shape[0]
        seen = np.zeros_like(walls, dtype=bool)
        q = deque([start])
        seen[start] = True
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in MOVE_DELTAS:
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny) and not walls[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        return False

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size