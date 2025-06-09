"""
Configuration loader for Antichess RL training.

This module provides functions to load configuration from different sources:
- Default hardcoded values
- Optimized hyperparameters from JSON files
- Environment variables for overrides

This separation allows the main config.py to remain static while enabling
dynamic loading of optimized hyperparameters.
"""

import json
import os
from typing import Dict, Any, Optional

from models.custom_policy import ChessCNN
from schedules.schedules import LinearSchedule, CombinedSchedule


def load_optimized_hyperparameters(filepath: str) -> Dict[str, Any]:
    """
    Load optimized hyperparameters from a JSON file.
    
    Args:
        filepath: Path to the hyperparameter optimization results JSON file
        
    Returns:
        Dictionary of hyperparameters suitable for PPO training
        
    Raises:
        FileNotFoundError: If the results file doesn't exist
        ValueError: If the results file is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hyperparameter results file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in hyperparameter results file: {e}")
    
    if "best_params" not in results:
        raise ValueError("Hyperparameter results file missing 'best_params' key")
    
    params = results["best_params"].copy()
    
    # Convert learning rate configuration back to schedule object
    if "learning_rate" in params and isinstance(params["learning_rate"], dict):
        lr_config = params["learning_rate"]
        if lr_config["type"] == "LinearSchedule":
            params["learning_rate"] = LinearSchedule(
                lr_config["initial_value"],
                lr_config["final_value"]
            )
        else:
            raise ValueError(f"Unknown learning rate schedule type: {lr_config['type']}")
    
    # Reconstruct policy_kwargs from flattened parameters
    policy_kwargs = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 512},  # Default
        "net_arch": dict(pi=[512, 256, 128], vf=[512, 256, 128])  # Default
    }
    
    # Extract features_dim if present
    if "features_dim" in params:
        features_dim = params.pop("features_dim")
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": features_dim}
    
    # Reconstruct network architecture
    pi_layers = []
    vf_layers = []
    for i in [1, 2, 3]:
        pi_key = f"pi_layer{i}"
        vf_key = f"vf_layer{i}"
        if pi_key in params:
            pi_layers.append(params.pop(pi_key))
        if vf_key in params:
            vf_layers.append(params.pop(vf_key))
    
    if pi_layers and vf_layers:
        policy_kwargs["net_arch"] = dict(pi=pi_layers, vf=vf_layers)
    
    # Add reconstructed policy_kwargs if not already present
    if "policy_kwargs" not in params:
        params["policy_kwargs"] = policy_kwargs
    else:
        # Merge with existing policy_kwargs
        params["policy_kwargs"].update(policy_kwargs)
    
    return params


def get_default_ppo_params() -> Dict[str, Any]:
    """
    Get default PPO hyperparameters.
    
    Returns:
        Dictionary of default PPO hyperparameters
    """
    return {
        "learning_rate": CombinedSchedule(1e-4, 1e-6, linear_pct=0.6),
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


def get_ppo_params(hyperopt_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get PPO hyperparameters from various sources.
    
    Priority order:
    1. Optimized hyperparameters from file (if provided and exists)
    2. Default hyperparameters
    
    Args:
        hyperopt_path: Optional path to hyperparameter optimization results
        
    Returns:
        Dictionary of PPO hyperparameters
    """
    # Try to load optimized hyperparameters
    if hyperopt_path and os.path.exists(hyperopt_path):
        try:
            print(f"Loading optimized hyperparameters from: {hyperopt_path}")
            return load_optimized_hyperparameters(hyperopt_path)
        except Exception as e:
            print(f"Warning: Failed to load optimized hyperparameters: {e}")
            print("Falling back to default hyperparameters")
    
    # Fall back to default hyperparameters
    return get_default_ppo_params()


def get_default_training_params() -> Dict[str, Any]:
    """
    Get default training configuration parameters.
    
    Returns:
        Dictionary of training parameters
    """
    return {
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


def get_default_curriculum_config() -> Dict[str, Any]:
    """
    Get default curriculum learning configuration.
    
    Returns:
        Dictionary of curriculum configuration
    """
    return {
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


def get_config(
    hyperopt_path: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None,
    curriculum_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get complete configuration for training.
    
    Args:
        hyperopt_path: Optional path to hyperparameter optimization results
        training_params: Optional override for training parameters
        curriculum_config: Optional override for curriculum configuration
        
    Returns:
        Dictionary containing all configuration sections
    """
    config = {
        "ppo_params": get_ppo_params(hyperopt_path),
        "training_params": training_params or get_default_training_params(),
        "curriculum_config": curriculum_config or get_default_curriculum_config()
    }
    
    return config


def find_latest_hyperopt_results(results_dir: str = "hyperopt_results") -> Optional[str]:
    """
    Find the most recent hyperparameter optimization results file.
    
    Args:
        results_dir: Directory containing hyperopt results
        
    Returns:
        Path to the latest results file, or None if no results found
    """
    if not os.path.exists(results_dir):
        return None
    
    # Find all JSON files that look like hyperopt results
    result_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("hyperopt_results_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                result_files.append(filepath)
    
    if not result_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file
