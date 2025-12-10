import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from gridworld import GridWorld, GridConfig

# -------------------------
# Tiny CNN Policy (3x8x8 -> 4 actions)
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs):
        # obs: (B, 3, 8, 8) float32
        x = self.features(obs)
        logits = self.head(x)
        return logits

    def act(self, obs, action_mask=None):
        """
        Returns sampled action, log-prob, and per-step entropy.
        obs: (B, 3, 8, 8)
        action_mask: optional (B, num_actions) boolean mask for valid actions
        """
        logits = self.forward(obs)
        
        # Apply action mask if provided (mask invalid actions)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()            # (B,)
        return a, logp, ent


# -------------------------
# Utilities
# -------------------------
def discount_returns(rewards, gamma):
    """
    Monte Carlo returns for a single episode.
    rewards: list[float]
    returns a list[float] same length.
    """
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def generate_grid_set(cfg, rng, num_grids):
    """Generate a fixed set of training grids."""
    print(f"Generating {num_grids} training grids...")
    grids = []
    for i in range(num_grids):
        env = GridWorld(cfg=cfg, rng=rng)
        env.reset()
        grids.append({
            'walls': env.walls.copy(),
            'start': env.start,
            'goal': env.goal
        })
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_grids} grids")
    print(f"Grid generation complete!\n")
    return grids


def load_grid_into_env(env, grid_data):
    """Load a saved grid configuration into an environment."""
    env.walls = grid_data['walls'].copy()
    env.start = grid_data['start']
    env.goal = grid_data['goal']
    env.agent = env.start
    env.t = 0
    return env._obs()


# -------------------------
# Training
# -------------------------
def main():
    # ---- Repro / device ----
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- IO ----
    os.makedirs("models", exist_ok=True)
    save_path = "models/policy.pt"
    grids_path = "models/train_grids.pkl"

    # ---- Env config ----
    cfg = GridConfig(
        size=8,
        obstacle_prob=0.20,
        max_steps=64,
        step_penalty=-0.1,   
        wall_penalty=-0.5,   
        goal_reward=1.0
    )

    # ---- Hyperparameters ----
    gamma = 0.99
    lr = 2.5e-4
    entropy_coef = 0.01
    batch_episodes = 32          # episodes per update
    total_updates = 1000          # total parameter updates
    print_every = 10
    num_train_grids = 400        # fixed set of training grids
    use_action_masking = True    # Prevent out-of-bounds actions (recommended)

    # ---- Generate or load fixed grid set ----
    env_rng = np.random.default_rng(seed)
    
    if os.path.exists(grids_path):
        print(f"Loading existing grids from {grids_path}")
        with open(grids_path, 'rb') as f:
            train_grids = pickle.load(f)
        print(f"Loaded {len(train_grids)} grids\n")
    else:
        train_grids = generate_grid_set(cfg, env_rng, num_train_grids)
        with open(grids_path, 'wb') as f:
            pickle.dump(train_grids, f)
        print(f"Saved grids to {grids_path}\n")

    # ---- Model/opt ----
    policy = PolicyNet(num_actions=4).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ---- Training loop ----
    running_success = []

    for upd in range(1, total_updates + 1):
        all_logps = []
        all_returns = []
        all_ents = []
        batch_success = []

        # Collect a batch of episodes with the current policy
        for _ in range(batch_episodes):
            # Sample a random grid from the fixed training set
            grid_idx = np.random.randint(len(train_grids))
            grid_data = train_grids[grid_idx]
            
            # Load grid into environment
            env = GridWorld(cfg=cfg, rng=env_rng)
            obs_np = load_grid_into_env(env, grid_data)
            done = False

            ep_logps = []
            ep_rewards = []
            ep_ents = []

            # Rollout one episode
            while not done:
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,8,8)
                
                # Optional: Create action mask for valid actions
                if use_action_masking:
                    valid_actions = env.get_valid_actions()
                    action_mask = torch.zeros(1, 4, dtype=torch.bool, device=device)
                    action_mask[0, valid_actions] = True
                    a, logp, ent = policy.act(obs_t, action_mask)
                else:
                    a, logp, ent = policy.act(obs_t)

                a_item = int(a.item())
                next_obs_np, r, done, _ = env.step(a_item)

                ep_logps.append(logp.squeeze(0))     # logp per step
                ep_rewards.append(r)
                ep_ents.append(ent.squeeze(0))       # entropy per step

                obs_np = next_obs_np

            # Success if last reward was positive (reached goal)
            success = 1.0 if (len(ep_rewards) > 0 and ep_rewards[-1] > 0.0) else 0.0
            batch_success.append(success)

            # Monte Carlo returns for this episode
            G = discount_returns(ep_rewards, gamma)           # list len T
            G_t = torch.tensor(G, dtype=torch.float32, device=device)

            # Store episode data for the batch
            all_logps += ep_logps
            all_returns.append(G_t)
            all_ents += ep_ents

        # Concatenate across the batch
        returns_t = torch.cat(all_returns)                    # (sum_T,)
        logps_t = torch.stack(all_logps)                      # (sum_T,)
        ents_t = torch.stack(all_ents)                        # (sum_T,)

        # Standardize returns (helps stability for REINFORCE)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient loss (REINFORCE)
        policy_loss = -(logps_t * returns_t).mean()

        # Entropy bonus (average over all steps in batch)
        entropy_bonus = ents_t.mean()

        loss = policy_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        running_success.extend(batch_success)
        if upd % print_every == 0:
            avg_succ_recent = np.mean(running_success[-10 * batch_episodes:]) if len(running_success) >= 10 * batch_episodes else np.mean(running_success)
            print(
                f"[upd {upd:4d}] "
                f"loss={loss.item():.4f}  "
                f"policy_loss={policy_loss.item():.4f}  "
                f"ent={entropy_bonus.item():.3f}  "
                f"avg_success_recent={avg_succ_recent:.3f}"
            )

    # Save final policy
    torch.save(policy.state_dict(), save_path)
    print(f"\nSaved policy to {save_path}")

if __name__ == "__main__":
    main()