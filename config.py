from models.custom_policy import ChessCNN

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "ent_coef": 0.1,  # Increased for more exploration during self-play
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 256], vf=[256, 256])
    }
}

# Training Configuration
TRAINING_PARAMS = {
    "total_timesteps": 1_000_000,
    "num_envs": 8,
    "opponent": "random",  # Starting opponent (will switch to self_play)
    "eval_freq": 10000,
    "n_eval_episodes": 50,
    "checkpoint_freq": 10000,
    # Self-play specific parameters
    "self_play_start_step": 200_000,  # When to start self-play
    "self_play_probability": 0.8,     # Probability of using model vs random
}

# Evaluation Configuration
EVALUATION_PARAMS = {
    "n_episodes": 100,
    "opponents": ["random", "heuristic", "self_play"],  # Added self_play
    "render": False,
}

# Environment Parameters
ENV_PARAMS = {
    "board_size": 8,
    "action_space_size": 4096,
}

# Model Architecture Parameters
MODEL_PARAMS = {
    "cnn_filters": 64,
    "features_dim": 256,
}