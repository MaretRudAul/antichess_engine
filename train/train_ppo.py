import argparse
import os
import sys
import zipfile
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.antichess_env import AntichessEnv
from models.custom_policy import ChessCNN, MaskedActorCriticPolicy
from config import get_ppo_params, get_training_params, get_curriculum_config, load_optimized_hyperparameters
from schedules.schedules import CurriculumAwareSchedule

from callbacks.callbacks import (
    MaskedEvalCallback,
    SelfPlayCallback,
    ImmediateSelfPlayCallback,
    EnhancedCurriculumCallback,
    GradientMonitorCallback,
    UpdateSelfPlayModelCallback
)


def make_env(rank, opponent="random", seed=0):
    def _init():
        env = AntichessEnv(opponent=opponent)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

def parse_args():
    # Load configuration parameters
    training_params = get_training_params()
    
    parser = argparse.ArgumentParser(description="Train an Antichess agent using PPO")
    
    # Training configuration
    parser.add_argument("--opponent", type=str, default="curriculum", 
                       choices=["random", "heuristic", "self_play", "curriculum", "mixed"],
                       help="Opponent strategy: random, heuristic, self_play, curriculum (random->self_play), or mixed")
    
    parser.add_argument("--total-timesteps", type=int, default=training_params["total_timesteps"],
                       help="Total number of timesteps to train")
    
    parser.add_argument("--num-envs", type=int, default=training_params["num_envs"],
                       help="Number of parallel environments")
    
    # Self-play specific arguments
    parser.add_argument("--self-play-start", type=int, default=training_params["self_play_start_step"],
                       help="Timestep to start self-play (only for curriculum mode)")
    
    parser.add_argument("--self-play-prob", type=float, default=training_params["self_play_probability"],
                       help="Probability of using model vs random in self-play mode (0.0-1.0)")
    
    # Mixed opponent arguments
    parser.add_argument("--random-prob", type=float, default=0.5,
                       help="Probability of random opponent in mixed mode (0.0-1.0)")
    
    parser.add_argument("--heuristic-prob", type=float, default=0.3,
                       help="Probability of heuristic opponent in mixed mode (0.0-1.0)")
    
    # Training options
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from checkpoint, start fresh training")
    
    parser.add_argument("--eval-freq", type=int, default=training_params["eval_freq"],
                       help="Evaluate every N timesteps")
    
    parser.add_argument("--checkpoint-freq", type=int, default=training_params["checkpoint_freq"],
                       help="Save checkpoint every N timesteps")
      # Curriculum options
    parser.add_argument("--use-enhanced-curriculum", action="store_true",
                       help="Use the enhanced multi-phase curriculum from config.py")
    
    # Hyperparameter optimization options
    parser.add_argument("--hyperopt-path", type=str, default=None,
                       help="Path to hyperparameter optimization results file")
    
    # Hardware options
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Output options
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Custom log directory (default: auto-generated)")
    
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Custom model directory (default: auto-generated)")

    parser.add_argument("--resume-from", type=str, default=None,
                   help="Path to a specific checkpoint to resume training from")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    parser.add_argument("--self-play-update-freq", type=int, default=training_params["self_play_update_freq"],
                   help="Update self-play model every N timesteps")
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments."""
    # Validate probabilities
    if not 0.0 <= args.self_play_prob <= 1.0:
        raise ValueError("self-play-prob must be between 0.0 and 1.0")
    
    if not 0.0 <= args.random_prob <= 1.0:
        raise ValueError("random-prob must be between 0.0 and 1.0")
    
    if not 0.0 <= args.heuristic_prob <= 1.0:
        raise ValueError("heuristic-prob must be between 0.0 and 1.0")
    
    # For mixed mode, probabilities should sum to reasonable values
    if args.opponent == "mixed":
        total_prob = args.random_prob + args.heuristic_prob
        if total_prob > 1.0:
            print(f"Warning: Random + Heuristic probabilities sum to {total_prob:.2f} > 1.0")
            print(f"Remaining probability ({1.0 - total_prob:.2f}) will be used for self-play")
      # Validate timesteps
    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive")
    
    # Only validate self-play-start for simple curriculum mode
    if args.opponent == "curriculum" and not args.use_enhanced_curriculum:
        if args.self_play_start >= args.total_timesteps:
            raise ValueError("self-play-start must be less than total-timesteps")
    
    # Validate environment count
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")

def create_mixed_env(random_prob=0.5, heuristic_prob=0.3):
    """Create an environment that randomly selects opponent strategy."""
    class MixedOpponentEnv(AntichessEnv):
        def __init__(self):
            # Randomly select initial opponent
            rand = np.random.random()
            if rand < random_prob:
                opponent = "random"
            elif rand < random_prob + heuristic_prob:
                opponent = "heuristic"
            else:
                opponent = "self_play"
            
            super().__init__(opponent=opponent)
            self.random_prob = random_prob
            self.heuristic_prob = heuristic_prob
        
        def reset(self, **kwargs):
            # Randomly select opponent for each episode
            rand = np.random.random()
            if rand < self.random_prob:
                self.opponent = "random"
            elif rand < self.random_prob + self.heuristic_prob:
                self.opponent = "heuristic"
            else:
                self.opponent = "self_play"
            
            return super().reset(**kwargs)
    
    return MixedOpponentEnv()

def extract_steps_from_checkpoint(checkpoint_path: str) -> int:
    """Extract the number of timesteps from a checkpoint filename."""
    try:
        filename = os.path.basename(checkpoint_path)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1].endswith('.zip'):
            steps = int(parts[-1].replace('.zip', ''))
            return steps
    except:
        pass
    
    # Return a reasonable default rather than accessing global config
    training_params = get_training_params()
    return training_params["checkpoint_freq"]

def create_new_model(env, log_dir, device="auto", lr_schedule=None, custom_hyperparams=None):
    """Create a new PPO model with the configured parameters"""
    print("Creating PPO agent with learning rate scheduling...")
    
    # Start with default PPO parameters
    ppo_params = get_ppo_params(verbose=False)  # Disable verbose to avoid duplicate messages
    
    # Override with custom hyperparameters if provided
    if custom_hyperparams is not None:
        print("Using custom hyperparameters from optimization results...")
        ppo_params.update(custom_hyperparams)
    
    # Override learning rate if provided
    if lr_schedule is not None:
        ppo_params["learning_rate"] = lr_schedule
    
    model = PPO(
        MaskedActorCriticPolicy,
        env,
        **ppo_params,
        tensorboard_log=log_dir,
        device=device,
        verbose=1
    )

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    return model

def main():
    """Train a PPO agent to play Antichess with configurable opponents."""
    args = parse_args()
    validate_args(args)
    
    # Load configuration parameters
    training_params = get_training_params()
    curriculum_config = get_curriculum_config() 
    
    # Load custom hyperparameters if specified
    custom_hyperparams = None
    if args.hyperopt_path:
        try:
            print(f"Loading optimized hyperparameters from: {args.hyperopt_path}")
            custom_hyperparams = load_optimized_hyperparameters(args.hyperopt_path)
            print("Successfully loaded custom hyperparameters!")
        except Exception as e:
            print(f"Warning: Failed to load hyperparameters from {args.hyperopt_path}: {e}")
            print("Continuing with default hyperparameters...")
    
    print("Starting Antichess PPO Training with Learning Rate Scheduling")
    print("=" * 60)
    
    # Print configuration
    print(f"Training Configuration:")
    print(f"   Opponent Strategy: {args.opponent}")
    print(f"   Total Timesteps: {args.total_timesteps:,}")
    print(f"   Parallel Environments: {args.num_envs}")
    print(f"   Learning Rate: Combined linear+cosine schedule (3e-4 → 1e-6)")
    
    if args.opponent == "curriculum":
        if args.use_enhanced_curriculum:
            print(f"   Curriculum Type: Enhanced (multi-phase)")
        else:
            print(f"   Curriculum Type: Simple (random → self-play)")
            print(f"   Self-play starts at: {args.self_play_start:,} timesteps")
            print(f"   Self-play probability: {args.self_play_prob:.1%}")
    elif args.opponent == "self_play":
        print(f"   Self-play probability: {args.self_play_prob:.1%}")
    elif args.opponent == "mixed":
        print(f"   Random opponent: {args.random_prob:.1%}")
        print(f"   Heuristic opponent: {args.heuristic_prob:.1%}")
        print(f"   Self-play opponent: {1.0 - args.random_prob - args.heuristic_prob:.1%}")
    
    curriculum_lr_schedule = None
    if args.opponent == "curriculum" and args.use_enhanced_curriculum:
        print("   Using curriculum-aware learning rate schedule:")
        print("     Phase 1 (Random): Linear schedule")
        print("     Phase 2 (Mixed: random, heuristic): Linear schedule")
        print("     Phase 3 (Mixed: random, heuristic, self-play): Cosine annealing")
        print("     Phase 4 (Mixed: majority self-play): Cosine annealing")
        curriculum_lr_schedule = CurriculumAwareSchedule(
            curriculum_config["lr_initial"], 
            curriculum_config["lr_final"], 
            curriculum_config, 
            args.total_timesteps
        )
    else:
        print("   Using combined linear+cosine schedule (60% linear, 40% cosine)")

    if args.opponent == "curriculum" or args.opponent == "self_play":
        print(f"   Self-play model update frequency: {args.self_play_update_freq:,}")
    
    # Set device with detailed CUDA information
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️  CUDA not available, using CPU")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ CUDA requested but not available! Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU for training")
    
    print(f"   Device: {device}")
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"   Random Seed: {args.seed}")
    
    print("=" * 60)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = f"logs/antichess_{args.opponent}_{timestamp}"
    
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = f"trained_models/antichess_{args.opponent}_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environments based on opponent strategy
    print(f"Creating {args.num_envs} training environments...")
    
    if args.opponent == "random":
        env = make_vec_env(
            lambda: AntichessEnv(opponent="random"),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=log_dir
        )
        
    elif args.opponent == "heuristic":
        env = make_vec_env(
            lambda: AntichessEnv(opponent="heuristic"),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=log_dir
        )
        
    elif args.opponent == "self_play":
        env = make_vec_env(
            lambda: AntichessEnv(opponent="self_play"),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=log_dir
        )
        
    elif args.opponent == "curriculum":
        # Start with random opponents, will switch to self-play later via callback
        env = make_vec_env(
            lambda: AntichessEnv(opponent="random"),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=log_dir
        )
        
        if args.use_enhanced_curriculum:
            print(f"Enhanced Curriculum Schedule:")
            for phase_name, phase_config in curriculum_config.items():
                print(f"   {phase_name}: {phase_config}")
        else:
            print(f"Simple Curriculum Schedule:")
            print(f"   Phase 1 (0-{args.self_play_start:,}): Random opponents")
            print(f"   Phase 2 ({args.self_play_start:,}+): Self-play ({args.self_play_prob:.1%} model, {1-args.self_play_prob:.1%} random)")
        
    elif args.opponent == "mixed":
        env = make_vec_env(
            lambda: create_mixed_env(args.random_prob, args.heuristic_prob),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv,
            monitor_dir=log_dir
        )
    
    else:
        raise ValueError(f"Unknown opponent strategy: {args.opponent}")
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: AntichessEnv(opponent="heuristic"),  # Always use heuristic for evaluation
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=os.path.join(log_dir, "eval")
    )
    
    # Replace the callback initialization section with this improved version:

    # Initialize callbacks list AFTER environment creation
    callbacks = []

    # First callback: Gradient monitoring (this is fine to keep first)
    gradient_monitor = GradientMonitorCallback(warning_threshold=100.0, verbose=1 if args.verbose else 0)
    callbacks.append(gradient_monitor)

    # Second: Add curriculum/opponent setup callbacks BEFORE evaluation
    if args.opponent == "curriculum":
        if args.use_enhanced_curriculum:
            # Use the enhanced multi-phase curriculum
            enhanced_curriculum_callback = EnhancedCurriculumCallback(
                curriculum_config=curriculum_config,
                verbose=1 if args.verbose else 0,
                model_dir=model_dir  # Pass model directory
            )
            callbacks.append(enhanced_curriculum_callback)
        else:
            # Use the simple two-phase curriculum
            self_play_callback = SelfPlayCallback(
                switch_timestep=args.self_play_start,
                self_play_prob=args.self_play_prob,
                verbose=1 if args.verbose else 0,
                model_dir=model_dir  # Pass model directory
            )
            callbacks.append(self_play_callback)
        
    elif args.opponent == "self_play":
        # Add callback to set up self-play immediately when training starts
        immediate_self_play_callback = ImmediateSelfPlayCallback(
            self_play_prob=args.self_play_prob,
            verbose=1 if args.verbose else 0,
            model_dir=model_dir  # Pass model directory
        )
        callbacks.append(immediate_self_play_callback)

    # Third: Add self-play model updates for ALL modes that might use self-play
    if args.opponent in ["self_play", "curriculum", "mixed"]:
        update_self_play_callback = UpdateSelfPlayModelCallback(
            update_freq=args.self_play_update_freq,
            model_dir=model_dir,
            verbose=1 if args.verbose else 0
        )
        callbacks.append(update_self_play_callback)

    # Fourth: Add evaluation callback AFTER environment setup is complete
    eval_callback = MaskedEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=training_params["n_eval_episodes"],
        best_model_save_path=model_dir,
        log_path=log_dir,
        verbose=1
    )
    callbacks.append(eval_callback)

    # Fifth: Add checkpoint callback (order doesn't matter as much here)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix="antichess_model",
        verbose=1 if args.verbose else 0
    )
    callbacks.append(checkpoint_callback)
    
    if args.resume_from is not None:
        checkpoint_path = args.resume_from
        if os.path.exists(checkpoint_path) and checkpoint_path.endswith(".zip"):
            print(f"Resuming training from specified checkpoint: {checkpoint_path}")
            try:
                model = PPO.load(
                    checkpoint_path,
                    env=env,
                    custom_objects={"policy_class": MaskedActorCriticPolicy},
                    tensorboard_log=log_dir
                )
                # Configure logging for resumed model
                new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
                model.set_logger(new_logger)
                
                # For manual resume, assume we want to continue training
                print(f"Resuming training for {args.total_timesteps:,} more steps.")
                  # Handle resumed training for different opponent types
                if args.opponent == "curriculum" and args.use_enhanced_curriculum:
                    # Re-detect which curriculum phase we should be in
                    print("Applying correct curriculum phase based on current timesteps...")
                    # Let the curriculum callback handle this during on_training_start
                
                elif args.opponent == "self_play":
                    print("Setting up self-play environments for resumed training...")
                    try:
                        env.env_method('set_opponent_model', model)
                        env.env_method('set_self_play_probability', args.self_play_prob)
                        print("Self-play configured for all environments")
                    except Exception as e:
                        print(f"Failed to setup self-play on resume: {e}")
                        print("Continuing with random opponents...")
                        
            except (zipfile.BadZipFile, ValueError, EOFError) as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training instead.")
                model = create_new_model(env, log_dir, device=device, lr_schedule=curriculum_lr_schedule, custom_hyperparams=custom_hyperparams)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Starting fresh training...")
            model = create_new_model(env, log_dir, device=device, lr_schedule=curriculum_lr_schedule, custom_hyperparams=custom_hyperparams)
    else:
        print("No checkpoint specified. Starting fresh training...")
        model = create_new_model(env, log_dir, device=device, lr_schedule=curriculum_lr_schedule, custom_hyperparams=custom_hyperparams)

    # Set remaining_steps to full training duration (don't try to calculate remaining)
    remaining_steps = args.total_timesteps

    if hasattr(model, 'num_timesteps'):
        completed_steps = model.num_timesteps
        remaining_steps = max(0, args.total_timesteps - completed_steps)
        print(f"Already trained for {completed_steps:,} steps. Continuing for {remaining_steps:,} more steps.")
    else:
        # If we can't determine steps, use the full amount
        remaining_steps = args.total_timesteps
        print(f"Could not determine completed steps. Training for full {remaining_steps:,} steps.")
    
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model state...")
        interrupted_model_path = os.path.join(model_dir, f"antichess_{args.opponent}_{timestamp}_interrupted")
        model.save(interrupted_model_path)
        print(f"Model saved to {interrupted_model_path}")
        return
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"   Total Timesteps: {args.total_timesteps:,}")
    print(f"   Opponent Strategy: {args.opponent}")
    if args.opponent == "curriculum" and args.use_enhanced_curriculum:
        print(f"   Learning Rate: Curriculum-aligned (linear→linear→cosine)")
    else:
        print(f"   Learning Rate: Combined linear+cosine schedule")
    print(f"   Final Model: {final_model_path}")
    print(f"   Training Logs: {log_dir}")
    print(f"   TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()