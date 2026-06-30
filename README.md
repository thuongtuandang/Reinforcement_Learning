# Reinforcement Learning Projects

This repository collects small reinforcement learning projects that combine theory, implementation, and experiments in gridworld-style environments and board games.

## Projects

### RL_LLM_Agent

Q-learning theory and experiments with an LLM agent playing GridWorld without direct access to the full solution. The LLM chooses moves through tools such as random moves, valid moves, BFS pathfinding, a trained Q-learning policy, and trap-aware helpers.

See `RL_LLM_Agent/docs/Q_Learning.pdf` for the Q-learning theory and `RL_LLM_Agent/docs/Implementation_Guide.pdf` for the agent/tool design and experiments.

### Policy_Gradient

Policy gradient theory and a REINFORCE implementation for GridWorld. The agent learns a policy over varied start and goal configurations, with training and evaluation on generated grid layouts.

See `Policy_Gradient/docs/Policy_Gradient_Theory.pdf` for the theory and `Policy_Gradient/docs/Policy_Gradient__Implementation.pdf` for implementation details.

### mcts_medium

AlphaZero-style Tic-Tac-Toe agent for a 5x5 board using Monte Carlo Tree Search and a PyTorch policy/value network. The project includes training, evaluation against random and minimax opponents, and human play scripts, including temperature-based play for more exploratory behavior.

See `mcts_medium/README.md` for setup and usage.

## Repository Layout

```text
RL_LLM_Agent/     Q-learning + LLM tool-using GridWorld agent
Policy_Gradient/  Policy gradient theory and GridWorld implementation
mcts_medium/      MCTS-based 5x5 Tic-Tac-Toe agent
```
