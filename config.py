from models.custom_policy import ChessCNN
from stable_baselines3.common.utils import get_linear_fn, constant_fn
import numpy as np


class LinearSchedule:
    """
    Picklable linear learning rate schedule.
    Compatible with Stable-Baselines3 and multiprocessing.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6):
        """
        Initialize linear schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
        """
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        # Linear interpolation from initial to final value
        return self.final_value + (self.initial_value - self.final_value) * progress_remaining

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters - CORRECT FUNCTIONS FOR SB3 v2.6.0
PPO_PARAMS = {
    "learning_rate": LinearSchedule(3e-4, 1e-6),  # Custom picklable linear schedule
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
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
    "phase_1": {
        "timesteps": 200_000,
        "opponent_mix": {"random": 1.0}
    },
    "phase_2": {
        "timesteps": 400_000,
        "opponent_mix": {"random": 0.7, "heuristic": 0.3}
    },
    "phase_3": {
        "timesteps": float('inf'),
        "opponent_mix": {"random": 0.3, "heuristic": 0.3, "self_play": 0.4}
    }
}