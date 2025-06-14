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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.antichess_env import AntichessEnv
from models.custom_policy import ChessCNN, MaskedActorCriticPolicy
from schedules.schedules import LinearSchedule, CombinedSchedule, CurriculumAwareSchedule
from config import get_curriculum_config


class TimestepLimitCallback(BaseCallback):
    """
    Callback to enforce a strict timestep limit during training.
    Stops training when the specified number of timesteps is reached.
    """
    
    def __init__(self, max_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_timesteps = max_timesteps
        
    def _on_step(self) -> bool:
        """Check if we should stop training due to timestep limit."""
        if self.num_timesteps >= self.max_timesteps:
            if self.verbose > 0:
                print(f"   🛑 TIMESTEP LIMIT REACHED: {self.num_timesteps:,} >= {self.max_timesteps:,}")
            return False  # Stop training
        return True


class HyperOptProgressCallback(BaseCallback):
    """
    Custom callback to show training progress during hyperparameter optimization.
    Provides detailed progress information similar to regular training.
    """
    
    def __init__(self, trial_number: int, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.trial_number = trial_number
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_update_time = None
        self.last_timesteps = 0
        # Fixed update interval - let TimestepLimitCallback handle early stopping
        self.update_interval = 5000
        
    def _on_training_start(self) -> None:
        """Initialize progress tracking when training starts."""
        import time
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_timesteps = 0
        if self.verbose > 0:
            print(f"\nTRIAL {self.trial_number + 1} TRAINING STARTED")
            print(f"   Target: {self.total_timesteps:,} timesteps")
            print(f"   {'='*50}")
    
    def _on_step(self) -> bool:
        """Update progress information periodically."""
        # Just show progress - let TimestepLimitCallback handle stopping
        if self.verbose > 0 and self.num_timesteps > 0 and self.num_timesteps % self.update_interval == 0:
            self._update_progress()
        return True
    
    def _on_training_end(self) -> None:
        """Show final training statistics."""
        if self.verbose > 0:
            self._update_progress(final=True)
            import time
            total_time = time.time() - self.start_time
            avg_fps = self.num_timesteps / total_time if total_time > 0 else 0
            print(f"   Training completed in {total_time:.1f}s")
            print(f"   Average: {avg_fps:.1f} FPS")
            print(f"   {'='*50}")
    
    def _update_progress(self, final: bool = False):
        """Update and display current progress."""
        import time
        current_time = time.time()
        
        # Calculate progress
        progress_pct = (self.num_timesteps / self.total_timesteps) * 100
        
        # Calculate FPS since last update
        time_diff = current_time - self.last_update_time
        timesteps_diff = self.num_timesteps - self.last_timesteps
        current_fps = timesteps_diff / time_diff if time_diff > 0 else 0
        
        # Calculate ETA
        if self.num_timesteps > 0 and not final:
            elapsed_time = current_time - self.start_time
            avg_fps = self.num_timesteps / elapsed_time
            remaining_timesteps = self.total_timesteps - self.num_timesteps
            eta_seconds = remaining_timesteps / avg_fps if avg_fps > 0 else 0
            eta_str = f"ETA: {eta_seconds/60:.1f}m"
        else:
            eta_str = "Complete" if final else "Calculating..."
        
        # Create progress bar
        bar_width = 30
        filled_width = int(bar_width * min(progress_pct, 100) / 100)  # Cap at 100%
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Print progress line  
        status = "FINAL" if final else "PROGRESS"
        progress_display = min(progress_pct, 100)  # Cap display at 100%
        print(f"   {status}: [{bar}] {progress_display:5.1f}% | "
              f"{self.num_timesteps:,}/{self.total_timesteps:,} | "
              f"{current_fps:.1f} FPS | {eta_str}")
        
        # Update tracking variables
        self.last_update_time = current_time
        self.last_timesteps = self.num_timesteps


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
        load_if_exists: bool = True,
        device: str = "auto"
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
            device: Device to use ('auto', 'cuda', 'cpu')
        """        
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.load_if_exists = load_if_exists
        
        # Set up device with automatic detection
        self.device = self._setup_device(device)
          # Set default sampler and pruner optimized for limited budget
        self.sampler = sampler or TPESampler(n_startup_trials=5, n_ei_candidates=50)
        self.pruner = pruner or MedianPruner(n_startup_trials=5, n_warmup_steps=15)
        
        # Training configuration for optimization
        self.optimization_config = {
            "total_timesteps": 2_000_000,  # 2M timesteps for thorough evaluation
            "num_envs": 4,  # Balanced for GPU memory
            "eval_freq": 40_000,  # More frequent evaluation for better pruning
            "n_eval_episodes": 12,  # Reliable evaluation
            "opponent": "curriculum",  # Use curriculum learning for better optimization
            "use_enhanced_curriculum": True,  # Enable enhanced curriculum
        }
        
        # Results storage
        self.results_dir = "hyperopt_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _setup_device(self, device: str) -> str:
        """
        Set up the training device with automatic CUDA detection and fallback.
        
        Args:
            device: Device preference ('auto', 'cuda', 'cpu')
            
        Returns:
            Device string to use for training
        """
        import torch
        
        if device == "auto":
            # Try CUDA first, fallback to CPU
            if torch.cuda.is_available():
                selected_device = "cuda"
                print(f"🔥 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                selected_device = "cpu"
                print("⚠️  CUDA not available, using CPU")
        elif device == "cuda":
            if torch.cuda.is_available():
                selected_device = "cuda"
                print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("❌ CUDA requested but not available! Falling back to CPU")
                selected_device = "cpu"
        else:
            selected_device = "cpu"
            print("💻 Using CPU for training")
        
        return selected_device
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space for hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # Learning rate optimization - test different schedule types
        schedule_type = trial.suggest_categorical("schedule_type", ["linear", "combined", "curriculum"])
        lr_initial = trial.suggest_float("lr_initial", 1e-5, 1e-3, log=True)  # Wider range: 1e-5 to 1e-3
        lr_final = trial.suggest_float("lr_final", 1e-7, lr_initial * 0.1, log=True)  # Allow higher final rates
        
        # Create the appropriate schedule
        if schedule_type == "linear":
            learning_rate = LinearSchedule(lr_initial, lr_final)
        elif schedule_type == "combined":
            # For combined schedule, also optimize the linear percentage
            linear_pct = trial.suggest_float("linear_pct", 0.4, 0.8)
            learning_rate = CombinedSchedule(lr_initial, lr_final, linear_pct=linear_pct)
        else:  # curriculum
            # Use curriculum-aware schedule that adapts to training phases
            curriculum_config = get_curriculum_config()
            learning_rate = CurriculumAwareSchedule(
                lr_initial, lr_final, curriculum_config, self.optimization_config["total_timesteps"]
            )
        
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
            "learning_rate": learning_rate,  # Use the dynamically created schedule
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
        print(f"\n{'='*60}")
        print(f"STARTING TRIAL {trial.number + 1}")
        print(f"{'='*60}")
        
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        
        # Print the hyperparameters being tested
        print("Hyperparameters for this trial:")
        for key, value in hyperparams.items():
            if key == "policy_kwargs":
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if subkey == "net_arch":
                        print(f"    {subkey}: pi={subvalue['pi']}, vf={subvalue['vf']}")
                    else:
                        print(f"    {subkey}: {subvalue}")
            elif hasattr(value, '__class__') and 'Schedule' in value.__class__.__name__:
                print(f"  {key}: {value.__class__.__name__}")
            else:
                print(f"  {key}: {value}")
        print()
        
        try:
            # Create temporary directory for this trial
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create training environment - start with random opponents for curriculum
                if self.optimization_config.get("use_enhanced_curriculum", False):
                    # Enhanced curriculum starts with random opponents
                    def make_curriculum_env(rank):
                        def _init():
                            env = AntichessEnv(opponent="random")
                            env.seed(trial.number + rank)
                            return env
                        return _init
                    
                    train_env = SubprocVecEnv([
                        make_curriculum_env(i) 
                        for i in range(self.optimization_config["num_envs"])
                    ])
                else:
                    # Use the specified opponent directly
                    train_env = SubprocVecEnv([
                        self.make_env(i, seed=trial.number) 
                        for i in range(self.optimization_config["num_envs"])
                    ])
                train_env = VecMonitor(train_env)
                
                # Create evaluation environment (always use same opponent as training baseline)
                eval_opponent = "heuristic"  # Use heuristic for consistent evaluation
                def make_eval_env():
                    env = AntichessEnv(opponent=eval_opponent)
                    env.seed(trial.number + 1000)
                    return env
                
                eval_env = DummyVecEnv([make_eval_env])
                eval_env = VecMonitor(eval_env)
                
                # Create model with suggested hyperparameters
                print("Creating PPO model with suggested hyperparameters...")
                print(f"Using device: {self.device}")
                model = PPO(
                    MaskedActorCriticPolicy,
                    train_env,
                    verbose=1,  # Enable verbose output for training progress
                    tensorboard_log=temp_dir,  # Enable tensorboard for this trial
                    device=self.device,  # Explicitly set device
                    **hyperparams
                )
                
                # Create evaluation callback with pruning
                print("Setting up evaluation callback with pruning...")
                eval_callback = OptunaPruningCallback(
                    trial=trial,
                    total_timesteps=self.optimization_config["total_timesteps"],
                    eval_env=eval_env,
                    eval_freq=self.optimization_config["eval_freq"],
                    n_eval_episodes=self.optimization_config["n_eval_episodes"],
                    deterministic=True
                )
                
                # Create timestep limit callback
                timestep_limit_callback = TimestepLimitCallback(
                    max_timesteps=self.optimization_config["total_timesteps"],
                    verbose=1
                )
                
                # Create plateau detection callback for intelligent early stopping
                plateau_callback = PlateauDetectionCallback(
                    plateau_rollouts=10,
                    min_improvement=0.001,
                    use_enhanced_curriculum=self.optimization_config.get("use_enhanced_curriculum", False),
                    total_timesteps=self.optimization_config["total_timesteps"],
                    verbose=1
                )
                
                # Create progress tracking callback
                progress_callback = HyperOptProgressCallback(
                    trial_number=trial.number,
                    total_timesteps=self.optimization_config["total_timesteps"],
                    verbose=1
                )
                
                # Create enhanced curriculum callback if needed
                curriculum_callbacks = []
                if self.optimization_config.get("use_enhanced_curriculum", False):
                    from callbacks.callbacks import EnhancedCurriculumCallback
                    curriculum_config = get_curriculum_config()
                    enhanced_curriculum_callback = EnhancedCurriculumCallback(
                        curriculum_config=curriculum_config,
                        model_dir=temp_dir,  # Use temp dir for this trial
                        total_timesteps=self.optimization_config["total_timesteps"],  # Pass total timesteps
                        verbose=1
                    )
                    curriculum_callbacks.append(enhanced_curriculum_callback)
                
                # Combine callbacks - order matters!
                callbacks = [timestep_limit_callback, plateau_callback, progress_callback] + curriculum_callbacks + [eval_callback]
                
                # Train the model
                print(f"Starting training for {self.optimization_config['total_timesteps']:,} timesteps...")
                print(f"    Evaluation every {self.optimization_config['eval_freq']:,} timesteps")
                print(f"    Using {self.optimization_config['num_envs']} parallel environments")
                
                model.learn(
                    total_timesteps=self.optimization_config["total_timesteps"],
                    callback=callbacks,
                    progress_bar=False  # We'll handle our own progress reporting
                )
                
                # Get final evaluation score
                final_reward = eval_callback.get_best_mean_reward()
                
                # Clean up
                train_env.close()
                eval_env.close()
                
                # Report trial results
                print(f"TRIAL {trial.number + 1} COMPLETED")
                print(f"    Final reward: {final_reward:.4f}")
                print(f"    Training completed successfully!")
                print(f"{'='*60}\n")
                
                return final_reward
                
        except Exception as e:
            print(f"TRIAL {trial.number + 1} FAILED")
            print(f"    Error: {e}")
            print(f"    Returning penalty score...")
            print(f"{'='*60}\n")
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
        print(f"Training timesteps per trial: {self.optimization_config['total_timesteps']:,}")
        print(f"Evaluation frequency: {self.optimization_config['eval_freq']:,}")
        print(f"Parallel environments: {self.optimization_config['num_envs']}")
        print()
        
        # Define a callback to show progress between trials
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                print(f"\n🎯 OPTIMIZATION PROGRESS: {completed_trials}/{self.n_trials} trials completed")
                if study.best_value is not None:
                    print(f"    Best reward so far: {study.best_value:.4f} (Trial #{study.best_trial.number + 1})")
                print(f"    {'='*60}")
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f"    ✂️ Trial #{trial.number + 1} was pruned")
            elif trial.state == optuna.trial.TrialState.FAIL:
                print(f"    ❌ Trial #{trial.number + 1} failed")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,  # Disable default progress bar to reduce clutter
            callbacks=[trial_callback]
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
        # Convert schedule to serializable format
        if "lr_initial" in best_params and "lr_final" in best_params:
            schedule_type = best_params.pop("schedule_type", "linear")
            lr_config = {
                "type": schedule_type,
                "initial": best_params.pop("lr_initial"),
                "final": best_params.pop("lr_final")
            }
            if schedule_type == "combined" and "linear_pct" in best_params:
                lr_config["linear_pct"] = best_params.pop("linear_pct")
            # Note: curriculum schedule doesn't need extra params as it uses config
            best_params["learning_rate"] = lr_config
        
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
        
        params = results["best_params"].copy()          # Convert learning rate back to schedule object
        if "learning_rate" in params and isinstance(params["learning_rate"], dict):
            lr_config = params["learning_rate"]
            if lr_config["type"] == "linear":
                from schedules.schedules import LinearSchedule
                params["learning_rate"] = LinearSchedule(
                    lr_config["initial"],
                    lr_config["final"]
                )
            elif lr_config["type"] == "combined":
                from schedules.schedules import CombinedSchedule
                params["learning_rate"] = CombinedSchedule(
                    lr_config["initial"],
                    lr_config["final"],
                    linear_pct=lr_config.get("linear_pct", 0.6)
                )
            elif lr_config["type"] == "curriculum":
                from schedules.schedules import CurriculumAwareSchedule
                from config import get_curriculum_config
                curriculum_config = get_curriculum_config()
                # Note: total_timesteps should be provided by the caller
                params["learning_rate"] = CurriculumAwareSchedule(
                    lr_config["initial"],
                    lr_config["final"],
                    curriculum_config,
                    2_000_000  # Default total timesteps
                )
            elif lr_config["type"] == "combined":
                from schedules.schedules import CombinedSchedule
                linear_pct = lr_config.get("linear_pct", 0.6)  # Default to 0.6 if not specified
                params["learning_rate"] = CombinedSchedule(
                    lr_config["initial"],
                    lr_config["final"],
                    linear_pct=linear_pct
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
    
    def __init__(self, trial: optuna.Trial, total_timesteps: int = 2_000_000, **kwargs):
        """
        Initialize the pruning callback.
        
        Args:
            trial: Optuna trial object
            total_timesteps: Total training timesteps for progress calculation
            **kwargs: Arguments passed to EvalCallback
        """
        super().__init__(**kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.best_mean_reward = -100.0  # Start with a reasonable default instead of -inf
        self.total_timesteps = total_timesteps
        self.evaluations_completed = 0
        
        print(f"    📊 Evaluation callback initialized:")
        print(f"       Eval frequency: {self.eval_freq:,} steps (per environment)")
        print(f"       Episodes per eval: {self.n_eval_episodes}")
        print(f"       Total timesteps: {total_timesteps:,}")
        print(f"       Expected evaluations: ~{total_timesteps // (self.eval_freq * self.eval_env.num_envs)}")
    
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
                self.evaluations_completed += 1
                
                # Update best reward
                if current_reward > self.best_mean_reward:
                    self.best_mean_reward = current_reward
                    print(f"New best mean reward!")
                
                # Progress reporting
                progress_pct = (self.num_timesteps / self.total_timesteps) * 100
                print(f"    📊 EVALUATION {self.eval_idx + 1} at {self.num_timesteps:,} steps ({progress_pct:.1f}%)")
                print(f"       Current reward: {current_reward:.4f} | Best so far: {self.best_mean_reward:.4f}")
                
                # Report to Optuna
                self.trial.report(current_reward, step=self.eval_idx)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    print(f"    ✂️ TRIAL {self.trial.number + 1} PRUNED at evaluation {self.eval_idx + 1}")
                    print(f"       Reason: Performance below median of previous trials")
                    print(f"       Final reward: {current_reward:.4f}")
                    return False
                
                self.eval_idx += 1
        
        return continue_training
    
    def get_best_mean_reward(self) -> float:
        """Get the best mean reward achieved during evaluation."""
        if self.evaluations_completed == 0:
            print(f"    ⚠️  WARNING: No evaluations completed! Using default reward of 0.0")
            return 0.0  # Return neutral reward instead of -inf if no evaluations happened
        return self.best_mean_reward


class PlateauDetectionCallback(BaseCallback):
    """
    Callback to detect training plateaus and enable early stopping.
    Only stops training if we're in the final curriculum phase or not using curriculum.
    """
    
    def __init__(self, 
                 plateau_rollouts: int = 10, 
                 min_improvement: float = 0.001,
                 use_enhanced_curriculum: bool = True,
                 total_timesteps: int = 2_000_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.plateau_rollouts = plateau_rollouts
        self.min_improvement = min_improvement
        self.use_enhanced_curriculum = use_enhanced_curriculum
        self.total_timesteps = total_timesteps
        
        # Track recent performance
        self.recent_rewards = []
        self.rollout_count = 0
        self.last_rollout_timestep = 0
        
        # Load curriculum config to determine phases
        if use_enhanced_curriculum:
            from config import get_curriculum_config
            self.curriculum_config = get_curriculum_config()
            self.final_phase_start = self._calculate_final_phase_start()
        else:
            self.curriculum_config = None
            self.final_phase_start = 0  # Always allow early stopping if no curriculum
    
    def _calculate_final_phase_start(self) -> int:
        """Calculate when the final curriculum phase starts."""
        if not self.curriculum_config:
            return 0
        
        # Enhanced curriculum has 4 phases, we want the start of phase 4
        phase_1_pct = self.curriculum_config.get("phase_1", {}).get("duration_pct", 0.15)
        phase_2_pct = self.curriculum_config.get("phase_2", {}).get("duration_pct", 0.20)
        phase_3_pct = self.curriculum_config.get("phase_3", {}).get("duration_pct", 0.25)
        
        final_phase_pct = phase_1_pct + phase_2_pct + phase_3_pct
        return int(self.total_timesteps * final_phase_pct)
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout to track performance."""
        self.rollout_count += 1
        
        # Try to get mean reward from various sources
        mean_reward = None
        
        # Method 1: Try to get from PPO's episode info buffer (most reliable)
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            # Get recent episode rewards from PPO's internal buffer
            recent_episodes = list(self.model.ep_info_buffer)[-20:]  # Last 20 episodes
            if recent_episodes:
                episode_rewards = [ep_info['r'] for ep_info in recent_episodes if 'r' in ep_info]
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards)
                    if self.verbose > 2:
                        print(f"    Method 1 success: {len(episode_rewards)} episodes, mean={mean_reward:.4f}")
        
        # Method 2: Try to access from callback locals/globals (set by SB3 during rollout)
        if mean_reward is None:
            try:
                # Check if we're in a callback context with locals/globals set
                if hasattr(self, 'locals') and self.locals and 'ep_infos' in self.locals:
                    ep_infos = self.locals['ep_infos']
                    if ep_infos:
                        episode_rewards = [info['r'] for info in ep_infos if 'r' in info]
                        if episode_rewards:
                            mean_reward = np.mean(episode_rewards)
                            if self.verbose > 2:
                                print(f"    Method 2 success: {len(episode_rewards)} episodes, mean={mean_reward:.4f}")
            except Exception as e:
                if self.verbose > 2:
                    print(f"    Method 2 failed: {e}")
        
        # Method 3: Try to access the logger's most recent values
        if mean_reward is None and hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            try:
                # Try different possible keys for episode reward mean
                possible_keys = ['rollout/ep_rew_mean', 'train/ep_rew_mean', 'episode_reward_mean']
                for key in possible_keys:
                    if key in self.model.logger.name_to_value:
                        mean_reward = self.model.logger.name_to_value[key]
                        if self.verbose > 2:
                            print(f"    Method 3 success: key='{key}', mean={mean_reward:.4f}")
                        break
            except Exception as e:
                if self.verbose > 2:
                    print(f"    Method 3 failed: {e}")
        
        # Method 4: Try to get from training environment's monitor wrapper
        if mean_reward is None:
            try:
                env = self.model.get_env()
                if hasattr(env, 'get_attr'):
                    # Try to get recent episode rewards from Monitor wrappers
                    try:
                        # Get episode rewards from each environment's monitor
                        all_rewards = []
                        episode_rewards = env.get_attr('episode_rewards')
                        for env_rewards in episode_rewards:
                            if env_rewards and len(env_rewards) > 0:
                                all_rewards.extend(env_rewards[-3:])  # Last 3 from each env
                        if all_rewards:
                            mean_reward = np.mean(all_rewards)
                            if self.verbose > 2:
                                print(f"    Method 4 success: {len(all_rewards)} episodes, mean={mean_reward:.4f}")
                    except Exception as sub_e:
                        if self.verbose > 2:
                            print(f"    Method 4 sub-failed: {sub_e}")
            except Exception as e:
                if self.verbose > 2:
                    print(f"    Method 4 failed: {e}")
        
        # If still no reward found, provide detailed debugging info
        if mean_reward is None:
            if self.verbose > 1:
                print(f"    ⚠️  WARNING: Could not extract reward for rollout {self.rollout_count}")
                if self.verbose > 2:
                    print(f"    Debug info:")
                    if hasattr(self.model, 'ep_info_buffer'):
                        buffer_size = len(self.model.ep_info_buffer) if self.model.ep_info_buffer else 0
                        print(f"      ep_info_buffer size: {buffer_size}")
                        if buffer_size > 0:
                            sample_info = list(self.model.ep_info_buffer)[-1]
                            print(f"      latest ep_info keys: {list(sample_info.keys())}")
                    if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                        available_keys = list(self.model.logger.name_to_value.keys())
                        reward_keys = [k for k in available_keys if 'reward' in k.lower() or 'rew' in k.lower() or 'ep_' in k.lower()]
                        print(f"      Available episode/reward keys: {reward_keys[:10]}")  # Show first 10
                    
                    # Also check locals if available
                    if hasattr(self, 'locals') and self.locals:
                        local_keys = list(self.locals.keys())
                        info_keys = [k for k in local_keys if 'info' in k.lower() or 'ep' in k.lower() or 'reward' in k.lower()]
                        print(f"      Available locals keys: {info_keys}")
            
            # Don't add None/invalid rewards to our tracking - just skip this rollout
            return
        
        # Successfully extracted a reward
        self.recent_rewards.append(mean_reward)
        
        # Keep only recent rewards for plateau detection
        if len(self.recent_rewards) > self.plateau_rollouts:
            self.recent_rewards.pop(0)
        
        if self.verbose > 1:  # Only show with high verbosity
            print(f"    Rollout {self.rollout_count}: Mean reward = {mean_reward:.4f} (tracking {len(self.recent_rewards)} rollouts)")
    
    def _is_in_final_phase(self) -> bool:
        """Check if we're in the final curriculum phase or not using curriculum."""
        if not self.use_enhanced_curriculum:
            return True  # Always allow early stopping if no curriculum
        
        return self.num_timesteps >= self.final_phase_start
    
    def _detect_plateau(self) -> bool:
        """Detect if training has plateaued."""
        if len(self.recent_rewards) < self.plateau_rollouts:
            return False
        
        # Check if there's been minimal improvement over the last N rollouts
        early_rewards = self.recent_rewards[:self.plateau_rollouts//2]
        recent_rewards = self.recent_rewards[self.plateau_rollouts//2:]
        
        # Ensure we have valid rewards to compare
        if not early_rewards or not recent_rewards:
            return False
        
        early_mean = np.mean(early_rewards)
        recent_mean = np.mean(recent_rewards)
        
        improvement = recent_mean - early_mean
        
        if self.verbose > 2:
            print(f"    Plateau check: Early mean={early_mean:.4f}, Recent mean={recent_mean:.4f}, Improvement={improvement:.4f}")
        
        return improvement < self.min_improvement
    
    def _on_step(self) -> bool:
        """Check for early stopping conditions."""
        # Check if we completed a rollout (detect significant timestep jumps)
        try:
            if hasattr(self.model, 'n_steps') and hasattr(self.model, 'get_env'):
                env = self.model.get_env()
                if hasattr(env, 'num_envs') and env.num_envs is not None:
                    rollout_size = self.model.n_steps * env.num_envs
                    if rollout_size > 0:  # Additional safety check
                        current_rollout = self.num_timesteps // rollout_size
                        
                        # Check if we've completed new rollouts
                        if current_rollout > self.rollout_count:
                            # We may have completed multiple rollouts - call end for each
                            rollouts_completed = current_rollout - self.rollout_count
                            for _ in range(rollouts_completed):
                                self._on_rollout_end()
                elif hasattr(env, 'num_envs'):
                    # If num_envs is None, fall back to simple timestep-based detection
                    n_steps = getattr(self.model, 'n_steps', 2048)  # Default PPO rollout size
                    if self.num_timesteps > 0 and self.num_timesteps % n_steps == 0:
                        # Likely completed a rollout
                        self._on_rollout_end()
        except Exception as e:
            # If we can't determine rollout boundaries, just continue
            if self.verbose > 1:
                print(f"    Warning: Could not detect rollout boundary: {e}")
        
        # Only consider early stopping if we're in the final phase
        if not self._is_in_final_phase():
            if self.verbose > 1 and self.rollout_count % 5 == 0:  # Log occasionally
                phase_info = f"curriculum phase {self._get_current_phase()}" if self.use_enhanced_curriculum else "non-curriculum training"
                print(f"    Rollout {self.rollout_count}: In {phase_info}, not checking for plateau")
            return True
        
        # Check for plateau only if we have enough data
        if len(self.recent_rewards) >= self.plateau_rollouts and self._detect_plateau():
            if self.verbose > 0:
                current_phase = "final curriculum phase" if self.use_enhanced_curriculum else "training"
                print(f"    🛑 PLATEAU DETECTED: No improvement for {self.plateau_rollouts} rollouts in {current_phase}")
                print(f"       Recent rewards: {[f'{r:.4f}' for r in self.recent_rewards[-5:]]}")
                print(f"       Early mean: {np.mean(self.recent_rewards[:self.plateau_rollouts//2]):.4f}")
                print(f"       Late mean: {np.mean(self.recent_rewards[self.plateau_rollouts//2:]):.4f}")
                print(f"       Stopping training early to save time...")
            return False
        
        return True
    
    def _get_current_phase(self) -> int:
        """Get the current curriculum phase (1-4)."""
        if not self.use_enhanced_curriculum:
            return 1
        
        progress = self.num_timesteps / self.total_timesteps
        if progress < 0.15:
            return 1
        elif progress < 0.35:
            return 2
        elif progress < 0.60:
            return 3
        else:
            return 4
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
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training ('auto', 'cuda', 'cpu')"
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
        load_if_exists=args.load_study,
        device=args.device  # Use device from command line
    )
    
    # Update optimization config with command line arguments
    optimizer.optimization_config.update({
        "total_timesteps": args.training_timesteps,
        "num_envs": args.num_envs
    })
    
    # Adjust evaluation frequency for short runs
    if args.training_timesteps <= 200_000:
        # For short runs, evaluate much more frequently
        # eval_freq should be based on steps per environment, not total timesteps
        steps_per_env = args.training_timesteps // args.num_envs
        eval_freq = max(1000, min(steps_per_env // 2, 10_000))  # Evaluate at least twice during training
        optimizer.optimization_config["eval_freq"] = eval_freq
        optimizer.optimization_config["n_eval_episodes"] = 5  # Fewer episodes for speed
        print(f"Adjusted evaluation frequency to {eval_freq:,} steps (per env) for {args.training_timesteps:,} timestep run")
        print(f"   This means ~{args.training_timesteps // eval_freq} evaluations during training")
    
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
