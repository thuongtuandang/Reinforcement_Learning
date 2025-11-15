# Reinforcement Learning + LLM Agent in Gridworld

# Can a Large Language Model play a game on its own?

This project explores exactly that — by combining Reinforcement Learning (Q-learning) with an LLM-powered agent in a simple Gridworld game.

# How It Works

The agent is an LLM (e.g. GPT-4o-mini) with a set of tools it can use to make decisions.

Tools include:

random_move, valid_move, bfs_move (pathfinding), rl_move (Q-table), and more.

The environment:

-1 per move, -5 for walls, +10 for reaching the goal.

The LLM doesn't learn via RL — it uses an RL model as one of its decision tools and explains each move in natural language.

# Try it yourself

git clone https://github.com/thuongtuandang/Reinforcement_Learning.git

cd Reinforcement_Learning/RL_LLM_Agent

pip install openai==2.6.1

# Create a .env file:

OPENAI_API_KEY=your_api_key

OPENAI_MODEL=gpt-4o-mini

# Train Q-learning model

python train_rl.py

The RL model is saved at models/

It can fail, and in case it fails, you can run the training script again, or change num_obstacles in train_rl.py (default 45)

# Watch the LLM agent play

python main.py

# Documentation

In docs/Q_Learning.pdf, you will find the theory of Q-learning. The core of Q-learning is a fixed point theorem, and how can we approximate this fixed point to obtain the optimal policy.

You will also find the implementation guide in docs/Implementation_Guide.pdf. It will include the structure of the project (LLM agent, tools and Q-learning) for gridworld game. At the end, I will share my experimental results and raise some questions.