from models.custom_policy import ChessCNN
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

class CosineSchedule:
    """
    Picklable cosine annealing learning rate schedule.
    Compatible with Stable-Baselines3 and multiprocessing.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6):
        """
        Initialize cosine schedule.
        
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
        import math
        # Cosine decay from initial to final value
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1 - progress_remaining)))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay


class CombinedSchedule:
    """
    Combined linear and cosine schedule with phase transition.
    Linear decay is used for the first portion of training,
    then transitions to cosine annealing for fine-tuning.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6, 
                 linear_pct: float = 0.6):
        """
        Initialize combined schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
            linear_pct: Percentage of training that uses linear schedule
        """
        self.linear_schedule = LinearSchedule(initial_value, final_value)
        self.cosine_schedule = CosineSchedule(initial_value, final_value)
        self.linear_pct = linear_pct
        
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        # Determine which phase we're in
        training_progress = 1.0 - progress_remaining
        
        if training_progress < self.linear_pct:
            # In linear phase: rescale progress_remaining for this phase
            phase_progress_remaining = 1.0 - (training_progress / self.linear_pct)
            return self.linear_schedule(phase_progress_remaining)
        else:
            # In cosine phase: rescale progress_remaining for this phase
            cosine_phase_progress = (training_progress - self.linear_pct) / (1.0 - self.linear_pct)
            phase_progress_remaining = 1.0 - cosine_phase_progress
            return self.cosine_schedule(phase_progress_remaining)

"""
Global configuration parameters for the Antichess PPO training.
This file centralizes all hyperparameters and constants.
"""

# PPO Algorithm Hyperparameters - CORRECT FUNCTIONS FOR SB3 v2.6.0
PPO_PARAMS = {
    "learning_rate": CombinedSchedule(3e-4, 1e-6, linear_pct=0.6),  # Custom picklable linear schedule
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