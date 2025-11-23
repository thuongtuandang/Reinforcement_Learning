# Policy Gradient for GridWorld

This project implements **Policy Gradient** methods with theoretical foundations and practical demonstrations.

## Theory

`docs/Policy_Gradient_Theory.pdf` contains a self-contained mathematical treatment including:
- Policy gradient theorem proof (discounted and average reward settings)
- REINFORCE algorithm derivation via trajectory sampling

## Implementation

The code demonstrates these concepts through a **GridWorld navigation task**:
- **Environment**: 8×8 grid with randomly placed obstacles and goals
- **Agent**: CNN-based policy network that learns to navigate to the goal
- **Training**: REINFORCE algorithm with entropy regularization on a fixed set of 400 training grids
- **Evaluation**: Tests generalization to unseen grid configurations

### Architecture
- **Observation**: 3-channel image (walls, agent position, goal position)
- **Policy**: CNN (3×8×8 → 64 features → 4 actions)
- **Actions**: UP, RIGHT, DOWN, LEFT

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_rl.py
```
Trains the policy on 400 randomly generated grids for 500 updates. Saves the model to `models/policy.pt` and displays training statistics including success rate and loss metrics.

### Testing
```bash
python test_rl.py --episodes 5 --delay 0.5
```
Visualizes the trained agent navigating through new, unseen grids with step-by-step ASCII rendering.

**Optional arguments:**
- `--model`: Path to trained model (default: `models/policy.pt`)
- `--episodes`: Number of test episodes (default: 5)
- `--delay`: Delay between visualization steps in seconds (default: 0.5)

## Results

The trained agent achieves high success rates on both training grids and generalizes well to completely new grid configurations.