### Antichess Reinforcement Learning with PPO

This project implements a reinforcement learning agent for the chess variant "Antichess". In Antichess, the goal is to lose all your pieces or have no legal moves. Captures are mandatory.

### Project Structure

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
├── config.py            # Global constants or hyperparameters
├── requirements.txt
└── README.md
```

### Installation

```python3
# Clone the repository
git clone https://github.com/username/antichess_engine.git
cd antichess_engine

# Install dependencies
python3 -m venv venv
pip install -r requirements.txt
```

### Usage

Training a New Agent

```python3
python -m train.train_ppo
```

This will train a PPO agent using the settings defined in config.py. The training logs and model checkpoints will be saved in the logs/ and models directories respectively.

Evaluating a Trained Agent

```python3
python -m evaluate.evaluate_policy --model models/final_model.zip --episodes 100
```

This will evaluate the trained model against different opponents and generate performance statistics.

### Approach

This project uses:

1. **Custom Gym Environment**: An OpenAI Gym-compatible environment for Antichess with a 13-plane board representation.

2. **PPO Algorithm**: The Proximal Policy Optimization algorithm from Stable-Baselines3, which has been shown to be effective for complex games.

3. **CNN Architecture**: A custom convolutional neural network inspired by AlphaZero that processes the board state.

4. **Action Masking**: Ensures the agent only selects legal moves, greatly improving sample efficiency.

### Hyperparameters

The key hyperparameters can be adjusted in config.py:

- Learning rate: 3e-4
- Entropy coefficient: 0.01
- GAE lambda: 0.95
- Discount factor (gamma): 0.99

### Antichess Rules

Antichess differs from standard chess in several ways:

- The goal is to lose all pieces or have no legal moves
- Captures are mandatory when available
- Kings have no special status (no check or checkmate)
- There is no castling
- Pawns can promote to any piece

### Future Work

- Implement self-play training
- Add hyperparameter optimization
- Train thoroughly to attain real results
- Integrate with a chess GUI for human vs. AI play
