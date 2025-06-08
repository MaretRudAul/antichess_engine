# Antichess Reinforcement Learning with PPO

This project implements a reinforcement learning agent for the chess variant "Antichess". In Antichess, the goal is to lose all your pieces or have no legal moves. Captures are mandatory.

## Project Structure

```
antichess_engine/
├── antichess/           # Game logic and rules
│   ├── __init__.py
│   ├── board.py
│   └── utils.py
├── envs/                # Gym-compatible environment
│   ├── __init__.py
│   └── antichess_env.py
├── models/              # Custom neural network models
│   ├── __init__.py
│   └── custom_policy.py
├── train/               # Training logic
│   ├── __init__.py
│   └── train_ppo.py
├── evaluate/            # Evaluation scripts
│   ├── __init__.py
│   └── evaluate_policy.py
├── config.py            # Global constants and hyperparameters
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/antichess_engine.git
cd antichess_engine

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training a New Agent

#### Basic Training (Curriculum Learning - Recommended)

```bash
python -m train.train_ppo
```

This starts with random opponents and automatically switches to self-play after 200,000 timesteps.

#### Training Options

**Pure Random Opponents:**

```bash
python -m train.train_ppo --opponent random
```

**Pure Self-Play Training:**

```bash
python -m train.train_ppo --opponent self_play --self-play-prob 0.8
```

**Heuristic Opponents Only:**

```bash
python -m train.train_ppo --opponent heuristic
```

**Custom Curriculum Learning:**

```bash
python -m train.train_ppo --opponent curriculum --self-play-start 100000 --self-play-prob 0.9
```

**Mixed Opponents:**

```bash
python -m train.train_ppo --opponent mixed --random-prob 0.4 --heuristic-prob 0.4
```

#### Advanced Training Options

**Long Training with More Environments:**

```bash
python -m train.train_ppo --total-timesteps 2000000 --num-envs 16
```

**Fresh Training (Ignore Checkpoints):**

```bash
python -m train.train_ppo --no-resume --opponent self_play
```

**Custom Directories and Reproducibility:**

```bash
python -m train.train_ppo --log-dir ./my_experiment --model-dir ./my_models --seed 42 --verbose
```

### Full Command Line Options

```bash
python -m train.train_ppo --help
```

**Training Configuration:**

- `--opponent`: Opponent strategy (`random`, `heuristic`, `self_play`, `curriculum`, `mixed`)
- `--total-timesteps`: Total training timesteps (default: 1,000,000)
- `--num-envs`: Number of parallel environments (default: 8)

**Self-Play Options:**

- `--self-play-start`: When to start self-play in curriculum mode (default: 200,000)
- `--self-play-prob`: Probability of using model vs random in self-play (default: 0.8)

**Mixed Opponent Options:**

- `--random-prob`: Probability of random opponent in mixed mode (default: 0.5)
- `--heuristic-prob`: Probability of heuristic opponent in mixed mode (default: 0.3)

**Training Control:**

- `--no-resume`: Start fresh training, ignore checkpoints
- `--eval-freq`: Evaluation frequency (default: 10,000)
- `--checkpoint-freq`: Checkpoint save frequency (default: 10,000)

**System Options:**

- `--device`: Training device (`auto`, `cpu`, `cuda`)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable detailed output

**Output Options:**

- `--log-dir`: Custom log directory
- `--model-dir`: Custom model directory

### Evaluating a Trained Agent

```bash
python -m evaluate.evaluation_policy --model trained_models/final_model.zip --episodes 100
```

This evaluates the trained model against different opponents and generates performance statistics.

## Training Strategies

### 1. Curriculum Learning (Recommended)

```bash
python -m train.train_ppo --opponent curriculum
```

- **Phase 1 (0-200k steps):** Random opponents to learn basic tactics
- **Phase 2 (200k+ steps):** Self-play for advanced strategy development

### 2. Pure Self-Play

```bash
python -m train.train_ppo --opponent self_play
```

- Generally not recommended due to overfitting against itself. Mixed self-play is preferred:

```bash
python -m train.train_ppo --opponent self_play --self-play-prob 0.8
```

### 3. Mixed Training

```bash
python -m train.train_ppo --opponent mixed --random-prob 0.3 --heuristic-prob 0.5 --self-play-prob 0.2
```

## Approach

This project uses:

1. **Custom Gym Environment**: OpenAI Gym-compatible environment for Antichess with a 13-plane board representation and action masking.

2. **PPO Algorithm**: Proximal Policy Optimization from Stable-Baselines3, effective for complex games with large action spaces.

3. **CNN Architecture**: Custom convolutional neural network inspired by AlphaZero for processing board states.

4. **Action Masking**: Ensures only legal moves are selected, dramatically improving sample efficiency.

5. **Self-Play Training**: Progressive curriculum from random opponents to self-play for advanced strategy development.

6. **Multiple Opponent Strategies**:
   - **Random**: Uniformly random legal moves
   - **Heuristic**: Simple strategy preferring captures and central squares
   - **Self-Play**: Previous versions of the trained model

## Hyperparameters

Key hyperparameters (configurable in `config.py`):

- **Learning rate**: 3e-4
- **Entropy coefficient**: 0.1 (increased for exploration during self-play)
- **GAE lambda**: 0.95
- **Discount factor (gamma)**: 0.99
- **Network architecture**: 256x256 for both policy and value networks
- **CNN features**: 256-dimensional feature extraction

## Antichess Rules

Antichess differs from standard chess:

- **Objective**: Lose all pieces or have no legal moves
- **Mandatory captures**: Must capture when possible
- **No special king rules**: No check, checkmate, or castling
- **Pawn promotion**: Can promote to any piece
- **Win condition**: First player with no pieces or no legal moves wins

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Training Logs

- Progress metrics: `logs/*/progress.csv`
- Model checkpoints: `trained_models/*/`
- Evaluation results: `logs/*/evaluations.npz`

## Example Training Experiments

### Compare Different Strategies

```bash
# Baseline: Random opponents only
python -m train.train_ppo --opponent random --total-timesteps 500000

# Curriculum learning
python -m train.train_ppo --opponent curriculum --total-timesteps 500000

# Pure self-play
python -m train.train_ppo --opponent self_play --total-timesteps 500000
```

### Self-Play Strength Experiments

```bash
# Conservative self-play (50% model, 50% random)
python -m train.train_ppo --opponent self_play --self-play-prob 0.5

# Aggressive self-play (90% model, 10% random)
python -m train.train_ppo --opponent self_play --self-play-prob 0.9
```

## Future Work

- ~~Implement self-play training~~
- ~~Add comprehensive command-line interface~~
- Train thoroughly to achieve expert-level play
- Hyperparameter optimization with Optuna
- Integration with chess GUIs for human vs. AI play
- MCTS integration for hybrid approach
- Use Ray framework for distributed RL
- Update self-play to use updating opponent?

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License
