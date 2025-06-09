"""
Comprehensive configuration system for Antichess RL training.

This module provides all configuration loading and management functionality:
- Loading default configurations from JSON
- Loading optimized hyperparameters from optimization results
- Dynamic configuration based on environment variables
- Centralized access to all training parameters

Usage:
    from config import get_ppo_params, get_training_params, get_curriculum_config
    
    ppo_params = get_ppo_params()  # Uses defaults or optimized values
    training_params = get_training_params() 
    curriculum_config = get_curriculum_config()
"""

import json
import os
import glob
from typing import Dict, Any, Optional, Union

# Import required classes for dynamic configuration
from models.custom_policy import ChessCNN
from schedules.schedules import LinearSchedule, CombinedSchedule


def _load_defaults() -> Dict[str, Any]:
    """Load default configuration from JSON file."""
    # Get the directory where this config.py file is located
    config_dir = os.path.dirname(os.path.abspath(__file__))
    defaults_path = os.path.join(config_dir, "hyperopt", "defaults.json")
    
    try:
        with open(defaults_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load default configuration from {defaults_path}: {e}")


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
        if lr_config["type"] == "combined":
            params["learning_rate"] = CombinedSchedule(
                lr_config["initial"],
                lr_config["final"], 
                lr_config["linear_pct"]
            )
        elif lr_config["type"] == "linear":
            params["learning_rate"] = LinearSchedule(
                lr_config["initial"],
                lr_config["final"]
            )
        else:
            # Fallback to constant learning rate
            params["learning_rate"] = lr_config.get("initial", 3e-4)
    
    # Convert policy kwargs to proper format
    if "policy_kwargs" in params:
        policy_kwargs = params["policy_kwargs"]
        
        # Set features extractor class
        policy_kwargs["features_extractor_class"] = ChessCNN
        
        # Ensure net_arch is properly formatted
        if "net_arch" in policy_kwargs and isinstance(policy_kwargs["net_arch"], dict):
            # Convert from flat dict format to proper net_arch format
            net_arch = policy_kwargs["net_arch"]
            if "pi" in net_arch and "vf" in net_arch:
                policy_kwargs["net_arch"] = dict(
                    pi=net_arch["pi"],
                    vf=net_arch["vf"]
                )
    
    return params


def find_latest_hyperopt_results(results_dir: str = None) -> Optional[str]:
    """
    Find the most recent hyperparameter optimization results file.
    
    Args:
        results_dir: Directory to search for results files. 
                    Defaults to 'hyperopt_results' in project root.
    
    Returns:
        Path to the most recent results file, or None if no files found
    """
    if results_dir is None:
        # Default to hyperopt_results directory in project root
        project_root = os.path.dirname(__file__)
        results_dir = os.path.join(project_root, "hyperopt_results")
    
    if not os.path.exists(results_dir):
        return None
    
    # Look for JSON files in the results directory
    pattern = os.path.join(results_dir, "*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file


def get_ppo_params(hyperopt_path: str = None) -> Dict[str, Any]:
    """
    Get PPO parameters from defaults or optimized results.
    
    Args:
        hyperopt_path: Path to hyperparameter optimization results file.
                      If None, checks environment variable or auto-finds latest.
    
    Returns:
        Dictionary of PPO parameters ready for model creation
    """
    # Load defaults first
    defaults = _load_defaults()
    ppo_params = defaults["ppo_params"].copy()
    
    # Convert JSON format to actual objects
    ppo_params["learning_rate"] = CombinedSchedule(1e-4, 1e-6, linear_pct=0.6)
    ppo_params["policy_kwargs"] = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": dict(pi=[512, 256, 128], vf=[512, 256, 128])
    }
    
    # Try to load optimized hyperparameters
    if hyperopt_path is None:
        # Check environment variable
        hyperopt_path = os.environ.get("ANTICHESS_HYPEROPT_PATH")
        
        # If still None, try to find latest results
        if hyperopt_path is None:
            hyperopt_path = find_latest_hyperopt_results()
    
    # Load optimized parameters if available
    if hyperopt_path and os.path.exists(hyperopt_path):
        try:
            print(f"Loading optimized hyperparameters from: {hyperopt_path}")
            optimized_params = load_optimized_hyperparameters(hyperopt_path)
            ppo_params.update(optimized_params)
            print("Successfully loaded optimized hyperparameters!")
        except Exception as e:
            print(f"Warning: Failed to load optimized hyperparameters: {e}")
            print("Using default hyperparameters...")
    
    return ppo_params


def get_training_params() -> Dict[str, Any]:
    """Get training parameters from defaults."""
    defaults = _load_defaults()
    return defaults["training_params"].copy()


def get_curriculum_config() -> Dict[str, Any]:
    """Get curriculum configuration from defaults."""
    defaults = _load_defaults()
    return defaults["curriculum_config"].copy()


# Legacy compatibility - these will be imported by existing code
def get_default_ppo_params() -> Dict[str, Any]:
    """Legacy function - use get_ppo_params() instead."""
    return get_ppo_params()


def get_default_training_params() -> Dict[str, Any]:
    """Legacy function - use get_training_params() instead."""
    return get_training_params()


def get_default_curriculum_config() -> Dict[str, Any]:
    """Legacy function - use get_curriculum_config() instead."""
    return get_curriculum_config()


# Provide direct access to configurations for convenience
PPO_PARAMS = get_ppo_params()
TRAINING_PARAMS = get_training_params()
CURRICULUM_CONFIG = get_curriculum_config()