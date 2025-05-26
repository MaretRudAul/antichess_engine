import os
import zipfile
import numpy as np
import gymnasium as gym
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.antichess_env import AntichessEnv
from models.custom_policy import ChessCNN, MaskedActorCriticPolicy
from config import PPO_PARAMS, TRAINING_PARAMS

def make_env(rank, opponent="random", seed=0):
    """
    Helper function to create and seed an Antichess environment.
    
    Args:
        rank: Worker rank for running multiple environments
        opponent: Strategy for the opponent
        seed: Random seed
        
    Returns:
        A function that creates and initializes an environment
    """
    def _init():
        env = AntichessEnv(opponent=opponent)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

def main():
    """Train a PPO agent to play Antichess."""
    print("Starting Antichess PPO training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/antichess_ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    model_dir = f"models/antichess_ppo_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create vectorized environments
    num_envs = TRAINING_PARAMS["num_envs"]
    print(f"Creating {num_envs} environments...")
    
    env = make_vec_env(
        lambda: AntichessEnv(opponent=TRAINING_PARAMS["opponent"]),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=log_dir
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        lambda: AntichessEnv(opponent="heuristic"),  # Evaluate against stronger opponent
        n_envs=1
    )
    
    # Define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=TRAINING_PARAMS["eval_freq"],
        n_eval_episodes=TRAINING_PARAMS["n_eval_episodes"],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_PARAMS["checkpoint_freq"],
        save_path=model_dir,
        name_prefix="antichess_model"
    )
    
    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print("Resuming training from checkpoint...")
        try:
            model = PPO.load(
                latest_checkpoint,
                env=env,
                custom_objects={"policy_class": MaskedActorCriticPolicy},
                tensorboard_log=log_dir
            )
            # Extract the total_timesteps already trained
            trained_steps = extract_steps_from_checkpoint(latest_checkpoint)
            remaining_steps = max(0, TRAINING_PARAMS["total_timesteps"] - trained_steps)
            print(f"Already trained for {trained_steps} steps. Continuing for {remaining_steps} more steps.")
        except (zipfile.BadZipFile, ValueError, EOFError) as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training instead.")
            model = create_new_model(env, log_dir)
            remaining_steps = TRAINING_PARAMS["total_timesteps"]
    else:
        print("No checkpoint found. Starting new training...")
        model = create_new_model(env, log_dir)
        remaining_steps = TRAINING_PARAMS["total_timesteps"]
    
    print(f"Training PPO agent for {remaining_steps} steps...")

    callbacks = [
        checkpoint_callback,
        eval_callback
    ]
    
    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False  # Important to continue counting from where we left off
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

def find_latest_checkpoint() -> str:
    """
    Find the most recent valid checkpoint file.
    
    Returns:
        Path to the latest valid checkpoint file or None if not found
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    # Look for all checkpoint files
    checkpoint_paths = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".zip") and "antichess_model" in file:
                full_path = os.path.join(root, file)
                # Verify this is a valid zip file
                try:
                    with zipfile.ZipFile(full_path, 'r') as zip_ref:
                        # Just checking if it's a valid zip - no need to extract
                        pass
                    checkpoint_paths.append(full_path)
                except zipfile.BadZipFile:
                    print(f"Warning: Found corrupted checkpoint file {full_path}, skipping")
                    continue
    
    if not checkpoint_paths:
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_paths[0]

def extract_steps_from_checkpoint(checkpoint_path: str) -> int:
    """
    Extract the number of timesteps from a checkpoint filename.
    If can't extract, return a conservative estimate.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Number of timesteps trained
    """
    try:
        # Try to extract the timestep from the filename (format: antichess_model_XXXXXXX_steps.zip)
        filename = os.path.basename(checkpoint_path)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1].endswith('.zip'):
            steps = int(parts[-1].replace('.zip', ''))
            return steps
    except:
        pass
    
    # If extraction fails, return a conservative estimate based on checkpoint frequency
    return TRAINING_PARAMS["checkpoint_freq"]

def create_new_model(env, log_dir):
    """Create a new PPO model with the configured parameters"""
    print("Creating PPO agent...")
    
    policy_kwargs = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": dict(features_dim=256),
        "net_arch": PPO_PARAMS["net_arch"]
    }
    
    return PPO(
        policy=MaskedActorCriticPolicy,
        env=env,
        learning_rate=PPO_PARAMS["learning_rate"],
        n_steps=PPO_PARAMS["n_steps"],
        batch_size=PPO_PARAMS["batch_size"],
        n_epochs=PPO_PARAMS["n_epochs"],
        gamma=PPO_PARAMS["gamma"],
        gae_lambda=PPO_PARAMS["gae_lambda"],
        clip_range=PPO_PARAMS["clip_range"],
        clip_range_vf=PPO_PARAMS["clip_range_vf"],
        ent_coef=PPO_PARAMS["ent_coef"],
        vf_coef=PPO_PARAMS["vf_coef"],
        max_grad_norm=PPO_PARAMS["max_grad_norm"],
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir
    )

if __name__ == "__main__":
    main()