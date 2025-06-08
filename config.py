from models.custom_policy import ChessCNN
import numpy as np

from schedules.schedules import CombinedSchedule

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters - CORRECT FUNCTIONS FOR SB3 v2.6.0
PPO_PARAMS = {
    "learning_rate": CombinedSchedule(1e-4, 1e-6, linear_pct=0.6),  # Custom picklable linear schedule
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": dict(pi=[512, 256, 128], vf=[512, 256, 128])
    }
}

# Training Configuration
TRAINING_PARAMS = {
    "total_timesteps": 2_000_000,
    "num_envs": 8,
    "opponent": "curriculum",
    "eval_freq": 25000,
    "n_eval_episodes": 20,
    "checkpoint_freq": 50000,
    "self_play_start_step": 400_000,
    "self_play_probability": 0.8,
}

# Curriculum stages for mixed training
CURRICULUM_CONFIG = {
    # Global learning rate parameters
    "lr_initial": 1e-4,
    "lr_final": 1e-6,
    
    # Phase definitions
    "phase_1": {
        "timesteps": int(0.3 * 2_000_000),  # 30% of training
        "opponent_mix": {"random": 1.0}
    },
    "phase_2": {
        "timesteps": int(0.6 * 2_000_000),  # 30% of training
        "opponent_mix": {"random": 0.7, "heuristic": 0.3}
    },
    "phase_3": {
        "timesteps": float('inf'),  # Remaining 40% of training
        "opponent_mix": {"random": 0.3, "heuristic": 0.3, "self_play": 0.4}
    }
}