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
from config import PPO_PARAMS, TRAINING_PARAMS, CURRICULUM_CONFIG

class SelfPlayCallback(BaseCallback):
    """
    Callback to introduce self-play during training.
    """
    
    def __init__(self, switch_timestep=100_000, self_play_prob=0.8, verbose=0):
        super(SelfPlayCallback, self).__init__(verbose)
        self.switch_timestep = switch_timestep
        self.self_play_prob = self_play_prob
        self.switched = False
        
    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.switch_timestep:
            print(f"\n🔄 CURRICULUM TRANSITION at timestep {self.num_timesteps:,}")
            print(f"   Switching from random opponents to self-play")
            
            try:
                # First set the opponent type to self_play
                self.training_env.env_method('__setattr__', 'opponent', 'self_play')
                
                # Save the model to a temporary file
                import os
                temp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_self_play_model.zip")
                self.model.save(temp_model_path)
                
                # Share the model path instead of the model object
                self.training_env.env_method('set_opponent_model_path', temp_model_path)
                self.training_env.env_method('set_self_play_probability', self.self_play_prob)
                
                print(f"   All environments now using self-play with {self.self_play_prob:.1%} model probability")
                print(f"   🎯 Self-play phase activated!")
                self.switched = True
                
            except Exception as e:
                print(f"   ❌ Failed to enable self-play: {e}")
                print(f"   Continuing with random opponents...")
                    
        return True

class ImmediateSelfPlayCallback(BaseCallback):
    """
    Callback to set up self-play immediately for pure self-play mode.
    """
    
    def __init__(self, self_play_prob=0.8, verbose=0):
        super(ImmediateSelfPlayCallback, self).__init__(verbose)
        self.self_play_prob = self_play_prob
        self.setup_complete = False
        
    def _on_training_start(self) -> None:
        """Set up self-play environments when training starts."""
        if not self.setup_complete:
            print("🤖 Setting up self-play environments...")
            try:
                # Save the model to a temporary file
                import os
                temp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_self_play_model.zip")
                self.model.save(temp_model_path)
                
                # Share the model path instead of the model object
                self.training_env.env_method('set_opponent_model_path', temp_model_path)
                self.training_env.env_method('set_self_play_probability', self.self_play_prob)
                
                print(f"   Self-play configured for all environments ({self.self_play_prob:.1%} model)")
                self.setup_complete = True
            except Exception as e:
                print(f"   ❌ Failed to setup self-play: {e}")
                print(f"   Falling back to random opponents...")
        
    def _on_step(self) -> bool:
        return True

class EnhancedCurriculumCallback(BaseCallback):
    """
    Enhanced curriculum learning callback with multiple phases.
    Uses the config-defined curriculum schedule.
    """
    
    def __init__(self, curriculum_config=None, verbose=0):
        super(EnhancedCurriculumCallback, self).__init__(verbose)
        self.curriculum_config = curriculum_config or CURRICULUM_CONFIG
        self.current_phase = 0
        self.phase_keys = list(self.curriculum_config.keys())
        self.phase_start_timestep = 0
        self.model_set_for_self_play = False
        
    def _on_training_start(self) -> None:
        """Initialize curriculum at training start."""
        if self.verbose > 0:
            print("\n" + "="*50)
            print("🎓 ENHANCED CURRICULUM LEARNING INITIALIZED")
            print("="*50)
            self._print_curriculum_schedule()
            print("="*50)
            
        # Set initial phase
        self._update_phase_if_needed()
        
    def _on_step(self) -> bool:
        """Check if we need to advance to the next curriculum phase."""
        self._update_phase_if_needed()
        return True
    
    def _update_phase_if_needed(self):
        """Update the training phase if we've reached the next milestone."""
        if self.current_phase >= len(self.phase_keys):
            return  # Already at final phase
            
        current_phase_key = self.phase_keys[self.current_phase]
        current_phase_config = self.curriculum_config[current_phase_key]
        
        # Check if we should advance to next phase
        timesteps_in_phase = self.num_timesteps - self.phase_start_timestep
        phase_duration = current_phase_config["timesteps"]
        
        if timesteps_in_phase >= phase_duration and self.current_phase < len(self.phase_keys) - 1:
            # Advance to next phase
            self.current_phase += 1
            self.phase_start_timestep = self.num_timesteps
            
            if self.verbose > 0:
                next_phase_key = self.phase_keys[self.current_phase]
                print(f"\n🎯 CURRICULUM PHASE TRANSITION at timestep {self.num_timesteps:,}")
                print(f"   Transitioning from {current_phase_key} to {next_phase_key}")
            
            self._apply_phase_configuration()
    
    def _apply_phase_configuration(self):
        """Apply the configuration for the current phase."""
        if self.current_phase >= len(self.phase_keys):
            return
            
        phase_key = self.phase_keys[self.current_phase]
        phase_config = self.curriculum_config[phase_key]
        opponent_mix = phase_config["opponent_mix"]
        
        if self.verbose > 0:
            print(f"   Applying {phase_key} configuration:")
            for opponent, prob in opponent_mix.items():
                print(f"     {opponent}: {prob:.1%}")
        
        # Update environments based on opponent mix
        if "self_play" in opponent_mix and opponent_mix["self_play"] > 0:
            # Enable self-play
            if not self.model_set_for_self_play:
                if self.verbose > 0:
                    print(f"   🤖 Setting up self-play with model...")
                
                try:
                    self.training_env.env_method('set_opponent_model', self.model)
                    self.training_env.env_method('__setattr__', 'opponent', 'self_play')
                    self.training_env.env_method('set_self_play_probability', opponent_mix["self_play"])
                    self.model_set_for_self_play = True
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   ❌ Failed to enable self-play: {e}")
                        print(f"   Continuing with previous opponent type...")
            else:
                # Update self-play probability
                try:
                    self.training_env.env_method('set_self_play_probability', opponent_mix["self_play"])
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   ❌ Failed to update self-play probability: {e}")
        
        if self.verbose > 0:
            print(f"   ✅ Phase {phase_key} activated")
    
    def _print_curriculum_schedule(self):
        """Print the full curriculum schedule."""
        print("Curriculum Schedule:")
        cumulative_timesteps = 0
        
        for i, phase_key in enumerate(self.phase_keys):
            phase_config = self.curriculum_config[phase_key]
            duration = phase_config["timesteps"]
            opponent_mix = phase_config["opponent_mix"]
            
            if duration == float('inf'):
                print(f"  Phase {i+1} ({phase_key}): {cumulative_timesteps:,}+ timesteps")
            else:
                end_timesteps = cumulative_timesteps + duration
                print(f"  Phase {i+1} ({phase_key}): {cumulative_timesteps:,} - {end_timesteps:,} timesteps")
                cumulative_timesteps = end_timesteps
            
            for opponent, prob in opponent_mix.items():
                print(f"    {opponent}: {prob:.1%}")

class MaskedEvalCallback(BaseCallback):
    """
    Custom evaluation callback that properly handles action masking.
    """
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=50, 
                 best_model_save_path=None, log_path=None, verbose=0):
        super(MaskedEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate_policy()
        return True
    
    def _evaluate_policy(self):
        """Evaluate the current policy."""
        episode_rewards = []
        episode_lengths = []
        illegal_actions = 0
        
        print(f"\n{'='*50}")
        print(f"🎯 EVALUATION AT TIMESTEP {self.num_timesteps:,}")
        print(f"{'='*50}")
        
        for episode in range(self.n_eval_episodes):
            obs_vec = self.eval_env.reset()
            
            # Handle Dict observation space properly
            if isinstance(obs_vec, tuple):
                obs_vec = obs_vec[0]  # Get observation from (obs, info) tuple
            
            # For vectorized environments, obs_vec is a list/array of observations
            if isinstance(obs_vec, list):
                obs = obs_vec[0]  # Get first environment's observation
            elif isinstance(obs_vec, dict):
                obs = obs_vec  # Single environment case
            else:
                # Handle numpy array case for vectorized environments
                obs = {key: obs_vec[key][0] for key in obs_vec.keys()} if hasattr(obs_vec, 'keys') else obs_vec[0]
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Use the model to predict action with proper masking
                action = self._predict_with_mask(obs)
                
                # Take step in environment
                step_result = self.eval_env.step([action])
                
                # Handle both old and new gym API
                if len(step_result) == 4:
                    obs_vec, reward, done_vec, info = step_result
                elif len(step_result) == 5:
                    obs_vec, reward, terminated_vec, truncated_vec, info = step_result
                    done_vec = [t or tr for t, tr in zip(terminated_vec, truncated_vec)]
                else:
                    raise ValueError(f"Unexpected step result length: {len(step_result)}")
                
                # Extract values for single environment
                if isinstance(obs_vec, list):
                    obs = obs_vec[0]
                elif isinstance(obs_vec, dict):
                    obs = obs_vec
                else:
                    obs = {key: obs_vec[key][0] for key in obs_vec.keys()} if hasattr(obs_vec, 'keys') else obs_vec[0]
                
                episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                episode_length += 1
                done = done_vec[0] if isinstance(done_vec, (list, np.ndarray)) else done_vec
                
                # Check for illegal actions
                info_single = info[0] if isinstance(info, list) else info
                if isinstance(info_single, dict) and info_single.get("illegal_move", False):
                    illegal_actions += 1
                
                # Safety check to prevent infinite episodes
                if episode_length > 300:
                    done = True
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        illegal_rate = illegal_actions / self.n_eval_episodes
        
        # Always print results (regardless of verbose setting)
        print(f"Evaluation Results ({self.n_eval_episodes} episodes):")
        print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"  Mean Episode Length: {mean_length:.1f} ± {std_length:.1f}")
        print(f"  Illegal Action Rate: {illegal_rate:.1%}")
        print(f"  Best Reward So Far: {self.best_mean_reward:.3f}")
        
        # Log to tensorboard/csv (shows up in training output)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        self.logger.record("eval/std_ep_length", std_length)
        self.logger.record("eval/illegal_rate", illegal_rate)
        self.logger.record("eval/n_episodes", self.n_eval_episodes)
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                best_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(best_path)
                print(f"  🎉 NEW BEST MODEL! Saved to {best_path}")
        
        print(f"{'='*50}\n")
    
    def _predict_with_mask(self, obs):
        """Make a prediction using the model while respecting action masks."""
        if not isinstance(obs, dict) or 'action_mask' not in obs:
            # Fallback to standard prediction if no mask available
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        
        # Convert observations to tensors
        device = next(self.model.policy.parameters()).device
        
        # Handle single observation (not batched)
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension if needed
                if value.ndim == len(self.model.observation_space[key].shape):
                    value = np.expand_dims(value, axis=0)
                obs_tensor[key] = torch.from_numpy(value).float().to(device)
            else:
                obs_tensor[key] = value
        
        # Use the policy's forward method directly
        with torch.no_grad():
            actions, values, log_probs = self.model.policy.forward(obs_tensor, deterministic=True)
        
        # Convert back to numpy and remove batch dimension
        action = actions.cpu().numpy()
        if action.ndim > 0:
            action = action[0]
        
        return int(action)

def make_env(rank, opponent="random", seed=0):
    def _init():
        env = AntichessEnv(opponent=opponent)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Antichess agent using PPO")
    
    # Training configuration
    parser.add_argument("--opponent", type=str, default="curriculum", 
                       choices=["random", "heuristic", "self_play", "curriculum", "mixed"],
                       help="Opponent strategy: random, heuristic, self_play, curriculum (random->self_play), or mixed")
    
    parser.add_argument("--total-timesteps", type=int, default=TRAINING_PARAMS["total_timesteps"],
                       help="Total number of timesteps to train")
    
    parser.add_argument("--num-envs", type=int, default=TRAINING_PARAMS["num_envs"],
                       help="Number of parallel environments")
    
    # Self-play specific arguments
    parser.add_argument("--self-play-start", type=int, default=TRAINING_PARAMS["self_play_start_step"],
                       help="Timestep to start self-play (only for curriculum mode)")
    
    parser.add_argument("--self-play-prob", type=float, default=TRAINING_PARAMS["self_play_probability"],
                       help="Probability of using model vs random in self-play mode (0.0-1.0)")
    
    # Mixed opponent arguments
    parser.add_argument("--random-prob", type=float, default=0.5,
                       help="Probability of random opponent in mixed mode (0.0-1.0)")
    
    parser.add_argument("--heuristic-prob", type=float, default=0.3,
                       help="Probability of heuristic opponent in mixed mode (0.0-1.0)")
    
    # Training options
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from checkpoint, start fresh training")
    
    parser.add_argument("--eval-freq", type=int, default=TRAINING_PARAMS["eval_freq"],
                       help="Evaluate every N timesteps")
    
    parser.add_argument("--checkpoint-freq", type=int, default=TRAINING_PARAMS["checkpoint_freq"],
                       help="Save checkpoint every N timesteps")
    
    # Curriculum options
    parser.add_argument("--use-enhanced-curriculum", action="store_true",
                       help="Use the enhanced multi-phase curriculum from config.py")
    
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
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
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

def find_latest_checkpoint() -> str:
    """Find the most recent valid checkpoint file."""
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        return None
    
    checkpoint_paths = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".zip") and "antichess_model" in file:
                checkpoint_paths.append(os.path.join(root, file))
    
    if not checkpoint_paths:
        return None
    
    checkpoint_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_paths[0]

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
    
    return TRAINING_PARAMS["checkpoint_freq"]

def create_new_model(env, log_dir, device="auto"):
    """Create a new PPO model with the configured parameters"""
    print("Creating PPO agent with learning rate scheduling...")
    
    model = PPO(
        MaskedActorCriticPolicy,
        env,
        **PPO_PARAMS,
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
    
    print("Starting Antichess PPO Training with Learning Rate Scheduling")
    print("=" * 60)
    
    # Print configuration
    print(f"Training Configuration:")
    print(f"   Opponent Strategy: {args.opponent}")
    print(f"   Total Timesteps: {args.total_timesteps:,}")
    print(f"   Parallel Environments: {args.num_envs}")
    print(f"   Learning Rate: Scheduled (linear decay from 3e-4)")
    
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
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
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
            for phase_name, phase_config in CURRICULUM_CONFIG.items():
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
    
    # Initialize callbacks list AFTER environment creation
    callbacks = []
    
    # Add evaluation callback
    eval_callback = MaskedEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=TRAINING_PARAMS["n_eval_episodes"],
        best_model_save_path=model_dir,
        log_path=log_dir,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix="antichess_model",
        verbose=1 if args.verbose else 0
    )
    callbacks.append(checkpoint_callback)
    
    # Add opponent-specific callbacks
    if args.opponent == "curriculum":
        if args.use_enhanced_curriculum:
            # Use the enhanced multi-phase curriculum
            enhanced_curriculum_callback = EnhancedCurriculumCallback(
                curriculum_config=CURRICULUM_CONFIG,
                verbose=1 if args.verbose else 0
            )
            callbacks.append(enhanced_curriculum_callback)
        else:
            # Use the simple two-phase curriculum
            self_play_callback = SelfPlayCallback(
                switch_timestep=args.self_play_start,
                self_play_prob=args.self_play_prob,
                verbose=1 if args.verbose else 0
            )
            callbacks.append(self_play_callback)
        
    elif args.opponent == "self_play":
        # Add callback to set up self-play immediately when training starts
        immediate_self_play_callback = ImmediateSelfPlayCallback(
            self_play_prob=args.self_play_prob,
            verbose=1 if args.verbose else 0
        )
        callbacks.append(immediate_self_play_callback)
    
    # Check for existing checkpoints (unless --no-resume is specified)
    if not args.no_resume:
        latest_checkpoint = find_latest_checkpoint()
    else:
        latest_checkpoint = None
        print("Checkpoint resume disabled, starting fresh training")
    
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
            # Configure logging for resumed model
            new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            
            trained_steps = extract_steps_from_checkpoint(latest_checkpoint)
            remaining_steps = max(0, args.total_timesteps - trained_steps)
            print(f"Already trained for {trained_steps:,} steps. Continuing for {remaining_steps:,} more steps.")
            
            # Handle resumed training for different opponent types
            if args.opponent == "curriculum" and trained_steps >= args.self_play_start:
                print("Checkpoint already past self-play threshold. Enabling self-play...")
                try:
                    env.env_method('__setattr__', 'opponent', 'self_play')
                    env.env_method('set_opponent_model', model)
                    env.env_method('set_self_play_probability', args.self_play_prob)
                    print("Self-play enabled for all environments")
                except Exception as e:
                    print(f"Failed to enable self-play on resume: {e}")
                    print("Continuing with current opponent type...")
            
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
            model = create_new_model(env, log_dir, device=device)
            remaining_steps = args.total_timesteps
    else:
        print("No checkpoint found. Starting new training...")
        model = create_new_model(env, log_dir, device=device)
        remaining_steps = args.total_timesteps
    
    print(f"Starting training for {remaining_steps:,} timesteps...")
    
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
    print(f"   Learning Rate: Scheduled (linear decay)")
    print(f"   Final Model: {final_model_path}")
    print(f"   Training Logs: {log_dir}")
    print(f"   TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()