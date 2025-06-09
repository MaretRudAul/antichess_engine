from models.custom_policy import ChessCNN
import numpy as np
import os

from schedules.schedules import CombinedSchedule

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.

Hyperparameters can be loaded from:
1. Optimized values from hyperparameter optimization results
2. Default hardcoded values (fallback)
"""

# Check for optimized hyperparameters
_HYPEROPT_RESULTS_PATH = os.environ.get("ANTICHESS_HYPEROPT_PATH")
if not _HYPEROPT_RESULTS_PATH:
    # Try to find the latest hyperopt results automatically
    from hyperopt.config_loader import find_latest_hyperopt_results
    _HYPEROPT_RESULTS_PATH = find_latest_hyperopt_results()

# Load hyperparameters (optimized or default)
if _HYPEROPT_RESULTS_PATH and os.path.exists(_HYPEROPT_RESULTS_PATH):
    print(f"Loading optimized hyperparameters from: {_HYPEROPT_RESULTS_PATH}")
    from hyperopt.config_loader import get_ppo_params
    PPO_PARAMS = get_ppo_params(_HYPEROPT_RESULTS_PATH)
else:
    # Default PPO Algorithm Hyperparameters
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
    "checkpoint_freq": 400000,
    "self_play_start_step": 400_000,
    "self_play_probability": 0.8,
    "self_play_update_freq": 50_000,
}

# Curriculum stages for mixed training
CURRICULUM_CONFIG = {
    # Global learning rate parameters
    "lr_initial": 5e-5,
    "lr_final": 1e-5,
    
    # Phase definitions
    "phase_1": {
        "timesteps": int(0.15 * 2_000_000),  # 15% of training
        "opponent_mix": {"random": 1.0}
    },
    "phase_2": {
        "timesteps": int(0.35 * 2_000_000),  # 20% of training
        "opponent_mix": {"random": 0.4, "heuristic": 0.6}
    },
    "phase_3": {
        "timesteps": int(0.60 * 2_000_000),  # 25% of training
        "opponent_mix": {"random": 0.25, "heuristic": 0.25, "self_play": 0.5}
    },
    "phase_4": {
        "timesteps": float('inf'),  # Remaining 40% of training
        "opponent_mix": {"random": 0.1, "heuristic": 0.1, "self_play": 0.8}
    }
}