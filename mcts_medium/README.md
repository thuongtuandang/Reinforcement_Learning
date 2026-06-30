# MCTS Tic-Tac-Toe

AlphaZero-style Tic-Tac-Toe agent for a 5x5 board using Monte Carlo Tree Search (MCTS) and a small PyTorch policy/value network. The default training script uses the medium preset: 5x5 with 4 in a row to win. The repository includes a trained checkpoint, and the resulting agent plays very strongly.

## Structure

- `env/` - Tic-Tac-Toe environment, config, and observation encoding
- `train_agent/` - neural network, MCTS, and training scripts
- `evaluation/` - scripts for evaluating trained agents
- `play/` - human-vs-agent play scripts
- `slides/` - seminar slides for MCTS and policy-value networks
- `models/` - trained model checkpoint

## Documentation

The seminar slides are available at `slides/MCTS.pdf`.

## Trained Model

The trained checkpoint is included at `models/trained_agent.pth`. You can play directly against the trained MCTS agent or run the evaluation scripts without retraining from scratch.

## Requirements

Python 3 with:

```bash
pip install numpy torch
```

## Usage

Train the medium agent:

```bash
python train_agent/train.py
```

Play against the trained agent:

```bash
python play/play.py
```

Play with a more creative agent:

```bash
python play/play_with_temperature.py
```

Increase `agent_temperature` in `play/play_with_temperature.py` for more varied moves (`0.0` = strongest/deterministic, `0.3` = balanced, `0.7+` = more exploratory). Training-time exploration can also be adjusted in `env/config.py` with `MCTS_TEMPERATURE`, `MCTS_C_PUCT`, and `MCTS_DIRICHLET_ALPHA`.

Evaluate against random or minimax opponents:

```bash
python evaluation/evaluate_vs_random.py
python evaluation/evaluate_vs_minimax.py
```
