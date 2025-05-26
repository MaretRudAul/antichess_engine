# File: config.py

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters
PPO_PARAMS = {
    "learning_rate": 3e-4,  # Learning rate
    "n_steps": 2048,  # Number of steps to collect before updating
    "batch_size": 64,  # Minibatch size for updates
    "n_epochs": 10,  # Number of policy update epochs per update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda parameter
    "clip_range": 0.2,  # PPO clipping parameter
    "clip_range_vf": 0.2,  # Value function clipping parameter
    "ent_coef": 0.01,  # Entropy coefficient (higher = more exploration)
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Gradient clipping for numerical stability
    "net_arch": [256, 256],  # Policy/value network architecture (shared layers)
}

# Training Configuration
TRAINING_PARAMS = {
    "total_timesteps": 10_000_000,  # Total timesteps to train for
    "num_envs": 8,  # Number of parallel environments
    "opponent": "random",  # Opponent strategy during training
    "eval_freq": 10000,  # How often to evaluate the agent
    "n_eval_episodes": 50,  # Number of episodes for evaluation
    "checkpoint_freq": 10000,  # How often to save model checkpoints
}

# Evaluation Configuration
EVALUATION_PARAMS = {
    "n_episodes": 100,  # Number of episodes for evaluation
    "opponents": ["random", "heuristic"],  # List of opponents to evaluate against
    "render": False,  # Whether to render the evaluation games
}

# Environment Parameters
ENV_PARAMS = {
    "board_size": 8,  # Standard 8x8 chess board
    "action_space_size": 4096,  # 64x64 possible from-to moves
}

# Model Architecture Parameters
MODEL_PARAMS = {
    "cnn_filters": 64,  # Number of filters in CNN layers
    "features_dim": 256,  # Dimension of extracted features
}