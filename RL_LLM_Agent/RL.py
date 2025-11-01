# gridworld_rl.py
# Minimal text-based GridWorld + tabular Q-learning demo
# Dependencies: Python 3.9+, numpy

import numpy as np
from dataclasses import dataclass

@dataclass
class StepResult:
    obs: int
    reward: float
    terminated: bool
    info: dict

class GridWorld:
    """
    5x5 grid. S=start, G=goal, X=wall.
    Rewards: +10 on goal, -1 per step, -5 hitting wall (stays in place).
    Actions: 0=up, 1=right, 2=down, 3=left
    Observation = single integer state id in [0, n_states)
    """
    def __init__(self, w=5, h=5, start=(0,0), goal=(4,4), walls={(1,1),(2,3),(3,1)}):
        self.w, self.h = w, h
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.n_actions = 4
        self.state = start

        # map (x,y) -> state_id and back
        self.id_of = {(x, y): y * self.w + x for y in range(self.h) for x in range(self.w)}
        self.xy_of = {v: k for k, v in self.id_of.items()}
        self.n_states = self.w * self.h

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.start
        return self._obs()

    def step(self, action: int) -> StepResult:
        x, y = self.state
        dxdy = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        dx, dy = dxdy[action]
        nx, ny = x + dx, y + dy

        reward = -1.0  # time penalty

        # bounds check
        if not (0 <= nx < self.w and 0 <= ny < self.h) or (nx, ny) in self.walls:
            # hit wall or boundary: stay put, extra penalty
            reward += -5.0
            nx, ny = x, y

        self.state = (nx, ny)
        terminated = (self.state == self.goal)
        if terminated:
            reward += 10.0

        return StepResult(self._obs(), reward, terminated, info={})

    def render(self):
        grid = [['.' for _ in range(self.w)] for _ in range(self.h)]
        for (wx, wy) in self.walls:
            grid[wy][wx] = 'X'
        sx, sy = self.start
        gx, gy = self.goal
        grid[sy][sx] = 'S'
        grid[gy][gx] = 'G'
        ax, ay = self.state
        grid[ay][ax] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()

    def _obs(self) -> int:
        return self.id_of[self.state]


# ---- Q-learning demo -------------------------------------------------------

def q_learning_demo(episodes=400, gamma=0.99, alpha=0.5, epsilon_start=1.0, epsilon_end=0.05):
    env = GridWorld()
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)

    def epsilon_greedy(s, eps):
        if np.random.rand() < eps:
            return np.random.randint(env.n_actions)
        return int(np.argmax(Q[s]))

    for ep in range(episodes):
        s = env.reset()
        eps = epsilon_end + (epsilon_start - epsilon_end) * max(0, (episodes - ep) / episodes)
        for t in range(200):  # safety cap
            a = epsilon_greedy(s, eps)
            step = env.step(a)
            s2, r, done = step.obs, step.reward, step.terminated
            # Q-learning update
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) * (0 if done else 1) - Q[s, a])
            s = s2
            if done:
                break

    return env, Q

def greedy_rollout(env, Q, render=True):
    s = env.reset()
    total = 0.0
    visited = [env.xy_of[s]]
    for _ in range(50):
        a = int(np.argmax(Q[s]))
        step = env.step(a)
        total += step.reward
        s = step.obs
        visited.append(env.xy_of[s])
        if render:
            env.render()
        if step.terminated:
            break
    return total, visited

if __name__ == "__main__":
    env, Q = q_learning_demo(episodes=400)
    print("Greedy rollout after training:")
    total, path = greedy_rollout(env, Q, render=True)
    print("Total reward:", total)
    print("Path:", path)