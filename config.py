from models.custom_policy import ChessCNN

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters
PPO_PARAMS = {
    "learning_rate": 1e-4,  # Reduced from 3e-4 for stability
    "n_steps": 4096,        # Doubled for better sample efficiency
    "batch_size": 128,      # Doubled for more stable updates
    "n_epochs": 4,          # Reduced from 10 to prevent overfitting
    "gamma": 0.995,         # Increased for longer-term planning
    "gae_lambda": 0.95,
    "clip_range": 0.15,     # More conservative updates
    "clip_range_vf": 0.15,
    "ent_coef": 0.08,       # Slightly reduced but still exploratory
    "vf_coef": 0.3,         # Reduced to focus more on policy
    "max_grad_norm": 0.8,
    "policy_kwargs": {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[384, 256], vf=[384, 256])  # Modest size increase
    }
}

# Training Configuration
TRAINING_PARAMS = {
    "total_timesteps": 1_500_000,
    "num_envs": 10,              # Increased from 8 for more parallel experience
    "opponent": "random",
    "eval_freq": 25000,          # Less frequent to save CPU time
    "n_eval_episodes": 20,       # Reduced from 50 for speed
    "checkpoint_freq": 25000,    # Less frequent saves
    # Self-play specific parameters
    "self_play_start_step": 500_000,  # Later start (40% through training)
    "self_play_probability": 0.65,    # More diversity than 0.8
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