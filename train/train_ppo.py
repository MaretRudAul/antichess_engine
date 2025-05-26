import os
import numpy as np
import gymnasium as gym
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
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
    
    # Create and train the PPO agent
    print("Creating PPO agent...")
    
    policy_kwargs = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": dict(features_dim=256),
        "net_arch": PPO_PARAMS["net_arch"]
    }
    
    model = PPO(
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
    
    print(f"Training PPO agent for {TRAINING_PARAMS['total_timesteps']} steps...")
    
    model.learn(
        total_timesteps=TRAINING_PARAMS["total_timesteps"],
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()