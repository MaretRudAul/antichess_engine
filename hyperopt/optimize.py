"""
Hyperparameter optimization module for Antichess RL using Optuna.

This module provides a comprehensive framework for optimizing PPO hyperparameters
using Bayesian optimization. It includes:
- Intelligent hyperparameter search spaces
- Early stopping for poor configurations
- Multi-objective optimization (performance + training stability)
- Automated result saving and loading
"""

import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.antichess_env import AntichessEnv
from models.custom_policy import ChessCNN, MaskedActorCriticPolicy
from schedules.schedules import LinearSchedule


class HyperparameterOptimizer:
    """
    Manages hyperparameter optimization for Antichess RL training.
    
    This class handles:
    - Defining search spaces for hyperparameters
    - Running optimization trials with early stopping
    - Evaluating trial performance
    - Saving and loading optimization results
    """
    def __init__(
        self,
        study_name: str = "antichess_hyperopt",
        storage: Optional[str] = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        load_if_exists: bool = True
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            study_name: Name for the Optuna study
            storage: Database URL for study persistence (None for in-memory)
            n_trials: Number of optimization trials to run
            n_jobs: Number of parallel optimization jobs
            sampler: Optuna sampler (defaults to TPESampler)
            pruner: Optuna pruner (defaults to MedianPruner)
        """        
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.load_if_exists = load_if_exists
          # Set default sampler and pruner optimized for limited budget
        self.sampler = sampler or TPESampler(n_startup_trials=5, n_ei_candidates=50)
        self.pruner = pruner or MedianPruner(n_startup_trials=5, n_warmup_steps=15)
        
        # Training configuration for optimization
        self.optimization_config = {
            "total_timesteps": 2_000_000,  # 2M timesteps for thorough evaluation
            "num_envs": 4,  # Balanced for GPU memory
            "eval_freq": 40_000,  # More frequent evaluation for better pruning
            "n_eval_episodes": 12,  # Reliable evaluation
            "opponent": "random",  # Simple opponent for initial optimization
        }
        
        # Results storage
        self.results_dir = "hyperopt_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space for hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # Learning rate optimization
        lr_initial = trial.suggest_float("lr_initial", 5e-6, 5e-3, log=True)
        lr_final = trial.suggest_float("lr_final", 1e-8, lr_initial * 0.05, log=True)
        
        # PPO-specific hyperparameters
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8192])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        
        # Ensure batch_size doesn't exceed n_steps * num_envs
        max_batch_size = n_steps * self.optimization_config["num_envs"]
        if batch_size > max_batch_size:
            batch_size = max_batch_size
        
        gamma = trial.suggest_float("gamma", 0.90, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.80, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.5)
        
        # Regularization parameters
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 2.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 2.0)
          # Network architecture - comprehensive search space
        features_dim = trial.suggest_categorical("features_dim", [128, 256, 384, 512, 768, 1024, 1536])
        
        # Policy network architecture - maximum diversity
        pi_layer1 = trial.suggest_categorical("pi_layer1", [128, 256, 384, 512, 768, 1024, 1536])
        pi_layer2 = trial.suggest_categorical("pi_layer2", [64, 128, 192, 256, 384, 512, 768])
        pi_layer3 = trial.suggest_categorical("pi_layer3", [32, 64, 96, 128, 192, 256, 384])
        
        # Value network architecture - independent optimization
        vf_layer1 = trial.suggest_categorical("vf_layer1", [128, 256, 384, 512, 768, 1024, 1536])
        vf_layer2 = trial.suggest_categorical("vf_layer2", [64, 128, 192, 256, 384, 512, 768])
        vf_layer3 = trial.suggest_categorical("vf_layer3", [32, 64, 96, 128, 192, 256, 384])
        
        return {
            "learning_rate": LinearSchedule(lr_initial, lr_final),
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "features_extractor_class": ChessCNN,
                "features_extractor_kwargs": {"features_dim": features_dim},
                "net_arch": dict(
                    pi=[pi_layer1, pi_layer2, pi_layer3],
                    vf=[vf_layer1, vf_layer2, vf_layer3]
                )
            }
        }    
    def make_env(self, rank: int, seed: int = 0):
        """Create a single environment for training."""
        def _init():
            env = AntichessEnv(opponent=self.optimization_config["opponent"])
            env.seed(seed + rank)
            return env
        return _init
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (mean evaluation reward)
        """
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        
        try:
            # Create temporary directory for this trial
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create training environment
                train_env = SubprocVecEnv([
                    self.make_env(i, seed=trial.number) 
                    for i in range(self.optimization_config["num_envs"])
                ])
                train_env = VecMonitor(train_env)                # Create evaluation environment (vectorized to match training env)
                def make_eval_env():
                    env = AntichessEnv(opponent=self.optimization_config["opponent"])
                    env.seed(trial.number + 1000)
                    # Don't wrap with Monitor here - VecMonitor will handle monitoring
                    return env
                
                eval_env = DummyVecEnv([make_eval_env])
                eval_env = VecMonitor(eval_env)
                
                # Create model with suggested hyperparameters
                model = PPO(
                    MaskedActorCriticPolicy,
                    train_env,
                    verbose=0,
                    tensorboard_log=None,  # Disable tensorboard for optimization
                    **hyperparams
                )
                
                # Create evaluation callback with pruning
                eval_callback = OptunaPruningCallback(
                    trial=trial,
                    eval_env=eval_env,
                    eval_freq=self.optimization_config["eval_freq"],
                    n_eval_episodes=self.optimization_config["n_eval_episodes"],
                    deterministic=True
                )
                
                # Train the model
                model.learn(
                    total_timesteps=self.optimization_config["total_timesteps"],
                    callback=eval_callback
                )
                
                # Get final evaluation score
                final_reward = eval_callback.get_best_mean_reward()
                
                # Clean up
                train_env.close()
                eval_env.close()
                
                return final_reward
                
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a very low score for failed trials
            return -1000.0
    
    def optimize(self) -> optuna.Study:
        """
        Run the hyperparameter optimization.
        
        Returns:
            Completed Optuna study
        """
        # Create or load study based on load_if_exists parameter
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=self.load_if_exists
        )
        
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        print(f"Study name: {self.study_name}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        return study
    
    def save_results(self, study: optuna.Study, filename: Optional[str] = None) -> str:
        """
        Save optimization results to JSON file.
        
        Args:
            study: Completed Optuna study
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperopt_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Extract best parameters and convert to serializable format
        best_params = study.best_params.copy()
          # Convert LinearSchedule to serializable format
        if "lr_initial" in best_params and "lr_final" in best_params:
            best_params["learning_rate"] = {
                "type": "linear",
                "initial": best_params.pop("lr_initial"),
                "final": best_params.pop("lr_final")
            }
        
        # Create results dictionary
        results = {
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "optimization_config": self.optimization_config,
            "timestamp": datetime.now().isoformat(),
            "trial_history": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    @staticmethod
    def load_best_hyperparameters(filepath: str) -> Dict[str, Any]:
        """
        Load best hyperparameters from saved results.
        
        Args:
            filepath: Path to saved results JSON file
            
        Returns:
            Dictionary of best hyperparameters in format suitable for config.py
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        params = results["best_params"].copy()
          # Convert learning rate back to schedule object
        if "learning_rate" in params and isinstance(params["learning_rate"], dict):
            lr_config = params["learning_rate"]
            if lr_config["type"] == "linear":
                from schedules.schedules import LinearSchedule
                params["learning_rate"] = LinearSchedule(
                    lr_config["initial"],
                    lr_config["final"]
                )
        
        # Ensure policy_kwargs structure is correct
        if "features_dim" in params:
            features_dim = params.pop("features_dim")
            if "policy_kwargs" not in params:
                params["policy_kwargs"] = {}
            params["policy_kwargs"]["features_extractor_kwargs"] = {"features_dim": features_dim}
            params["policy_kwargs"]["features_extractor_class"] = ChessCNN
        
        # Build net_arch from individual layer parameters
        pi_layers = []
        vf_layers = []
        for i in [1, 2, 3]:
            if f"pi_layer{i}" in params:
                pi_layers.append(params.pop(f"pi_layer{i}"))
            if f"vf_layer{i}" in params:
                vf_layers.append(params.pop(f"vf_layer{i}"))
        
        if pi_layers and vf_layers:
            if "policy_kwargs" not in params:
                params["policy_kwargs"] = {}
            params["policy_kwargs"]["net_arch"] = dict(pi=pi_layers, vf=vf_layers)
        
        return params


class OptunaPruningCallback(EvalCallback):
    """
    Custom evaluation callback that integrates with Optuna pruning.
    
    This callback reports intermediate results to Optuna and allows
    for early termination of unpromising trials.
    """
    
    def __init__(self, trial: optuna.Trial, **kwargs):
        """
        Initialize the pruning callback.
        
        Args:
            trial: Optuna trial object
            **kwargs: Arguments passed to EvalCallback
        """
        super().__init__(**kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """
        Called at each evaluation step.
        
        Returns:
            True if training should continue, False if pruned
        """
        # Call parent evaluation
        continue_training = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Get current mean reward
            if hasattr(self, 'last_mean_reward'):
                current_reward = self.last_mean_reward
                
                # Update best reward
                if current_reward > self.best_mean_reward:
                    self.best_mean_reward = current_reward
                
                # Report to Optuna
                self.trial.report(current_reward, step=self.eval_idx)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    print(f"Trial {self.trial.number} pruned at step {self.eval_idx}")
                    return False
                
                self.eval_idx += 1
        
        return continue_training
    
    def get_best_mean_reward(self) -> float:
        """Get the best mean reward achieved during evaluation."""
        return self.best_mean_reward


def parse_args():
    """Parse command line arguments for hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Antichess RL"
    )
    
    parser.add_argument(
        "--study-name", 
        type=str, 
        default="antichess_hyperopt",
        help="Name for the Optuna study"
    )
    
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=35,
        help="Number of optimization trials to run"
    )
    
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=1,
        help="Number of parallel optimization jobs"
    )
    
    parser.add_argument(
        "--storage", 
        type=str, 
        default=None,
        help="Database URL for study persistence (e.g., sqlite:///study.db)"
    )
    
    parser.add_argument(
        "--training-timesteps", 
        type=int, 
        default=2_000_000,
        help="Timesteps for each optimization trial"
    )
    
    parser.add_argument(
        "--num-envs", 
        type=int, 
        default=8,
        help="Number of parallel environments for each trial"
    )
    
    parser.add_argument(
        "--save-results", 
        type=str, 
        default=None,
        help="Filename to save optimization results (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--load-study", 
        action="store_true",
        help="Load existing study if it exists"
    )
    
    parser.add_argument(
        "--show-results", 
        action="store_true",
        help="Show optimization results and exit"
    )
    
    return parser.parse_args()


def main():
    """Main function for hyperparameter optimization."""
    args = parse_args()
      # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name=args.study_name,
        storage=args.storage,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        load_if_exists=args.load_study
    )
    
    # Update optimization config with command line arguments
    optimizer.optimization_config.update({
        "total_timesteps": args.training_timesteps,
        "num_envs": args.num_envs
    })
    
    if args.show_results:
        # Just show existing results
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=args.storage
            )
            print(f"\nStudy: {study.study_name}")
            print(f"Number of trials: {len(study.trials)}")
            print(f"Best value: {study.best_value:.4f}")
            print(f"Best parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Could not load study: {e}")
        return
    
    # Run optimization
    study = optimizer.optimize()
    
    # Display results
    print(f"\nOptimization completed!")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_path = optimizer.save_results(study, args.save_results)
    
    print(f"\nTo use these hyperparameters in training:")
    print(f"1. Use the saved results file: {results_path}")
    print(f"2. Update your config.py to load from this file")
    print(f"3. Run training with the optimized hyperparameters")


if __name__ == "__main__":
    main()
