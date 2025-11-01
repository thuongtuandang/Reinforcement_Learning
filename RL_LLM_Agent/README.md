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

Create a .env file:
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini

python train_rl.py   # Train Q-learning model
python main.py       # Watch the LLM agent play