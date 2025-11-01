import numpy as np
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions=4, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Q-Learning Agent with epsilon-greedy exploration
        
        Args:
            n_actions: Number of possible actions (4 for up/down/left/right)
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, explore=True):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current state (x, y)
            explore: If True, use epsilon-greedy; if False, always choose best
        
        Returns:
            action: 0=up, 1=down, 2=left, 3=right
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # No future rewards if episode is done
            target_q = reward
        else:
            # Bootstrap from best next action
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, n_episodes=5000, max_steps=200, verbose=True):
        """
        Train the Q-learning agent
        
        Args:
            env: GridWorld environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress
        
        Returns:
            rewards: List of total rewards per episode
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Choose action
                action = self.get_action(state, explore=True)
                
                # Take action
                next_state, reward, done = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            
            # Print progress
            if verbose and (episode + 1) % 500 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def save(self, filepath='models/q_table.pkl'):
        """Save Q-table to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert defaultdict to regular dict for pickling
        q_dict = dict(self.q_table)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_dict,
                'epsilon': self.epsilon,
                'lr': self.lr,
                'gamma': self.gamma
            }, f)
        
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath='models/q_table.pkl'):
        """Load Q-table from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No Q-table found at {filepath}. Train the model first!")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.epsilon = data['epsilon']
        self.lr = data['lr']
        self.gamma = data['gamma']
        
        print(f"Q-table loaded from {filepath}")
        print(f"Loaded {len(self.q_table)} states")