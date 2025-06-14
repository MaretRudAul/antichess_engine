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
from config import get_ppo_params, get_training_params, get_curriculum_config
from schedules.schedules import CurriculumAwareSchedule

import torch.multiprocessing as mp
from copy import deepcopy

class SelfPlayCallback(BaseCallback):
    """
    Callback to introduce self-play during training.
    """
    
    def __init__(self, switch_timestep=100_000, self_play_prob=0.8, model_dir=None, verbose=0):
        super(SelfPlayCallback, self).__init__(verbose)
        self.switch_timestep = switch_timestep
        self.self_play_prob = self_play_prob
        self.switched = False
        self.model_dir = model_dir
        
    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.switch_timestep:
            print(f"\nCURRICULUM TRANSITION at timestep {self.num_timesteps:,}")
            print(f"   Switching from random opponents to self-play")
            
            try:
                # First set the opponent type to self_play
                self.training_env.env_method('__setattr__', 'opponent', 'self_play')
                
                # Save the model to training-specific directory
                temp_model_path = os.path.join(self.model_dir, "temp_self_play_model.zip")
                self.model.save(temp_model_path)
                
                # Share the model path instead of the model object
                self.training_env.env_method('set_opponent_model_path', temp_model_path)
                self.training_env.env_method('set_self_play_probability', self.self_play_prob)
                
                print(f"   All environments now using self-play with {self.self_play_prob:.1%} model probability")
                print(f"   Self-play phase activated!")
                self.switched = True
                
            except Exception as e:
                print(f"   Failed to enable self-play: {e}")
                print(f"   Continuing with random opponents...")
                    
        return True

class ImmediateSelfPlayCallback(BaseCallback):
    """
    Callback to set up self-play immediately for pure self-play mode.
    """
    
    def __init__(self, self_play_prob=0.8, model_dir=None, verbose=0):
        super(ImmediateSelfPlayCallback, self).__init__(verbose)
        self.self_play_prob = self_play_prob
        self.setup_complete = False
        self.model_dir = model_dir  # Store model directory
        
    def _on_training_start(self) -> None:
        """Set up self-play environments when training starts."""
        if not self.setup_complete:
            print("Setting up self-play environments...")
            try:
                # Save the model to training-specific directory
                temp_model_path = os.path.join(self.model_dir, "temp_self_play_model.zip")
                self.model.save(temp_model_path)
                
                # Share the model path instead of the model object
                self.training_env.env_method('set_opponent_model_path', temp_model_path)
                self.training_env.env_method('set_self_play_probability', self.self_play_prob)
                
                print(f"   Self-play configured for all environments ({self.self_play_prob:.1%} model)")
                self.setup_complete = True
            except Exception as e:
                print(f"   Failed to setup self-play: {e}")
                print(f"   Falling back to random opponents...")
        
    def _on_step(self) -> bool:
        return True

class EnhancedCurriculumCallback(BaseCallback):
    """
    Enhanced curriculum learning callback with multiple phases.
    Uses the config-defined curriculum schedule.
    """
    
    def __init__(self, curriculum_config=None, model_dir=None, total_timesteps=None, verbose=0):
        super(EnhancedCurriculumCallback, self).__init__(verbose)
        self.curriculum_config = curriculum_config or get_curriculum_config()
        self.current_phase = 0
        self.phase_keys = [key for key in self.curriculum_config.keys() if key.startswith("phase_")]
        self.phase_start_timestep = 0
        self.model_set_for_self_play = False
        self.model_dir = model_dir  # Store model directory
        self.total_timesteps = total_timesteps
        
        # Scale curriculum phases based on actual training length
        if self.total_timesteps:
            self._scale_curriculum_phases()
        
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
                    print(f"   Setting up self-play with model...")
                
                try:
                    # Save model to training-specific directory
                    temp_model_path = os.path.join(self.model_dir, "temp_self_play_model.zip")
                    self.model.save(temp_model_path)
                    
                    # Use file-based model sharing instead of direct model passing
                    self.training_env.env_method('__setattr__', 'opponent', 'self_play')
                    self.training_env.env_method('set_opponent_model_path', temp_model_path)
                    self.training_env.env_method('set_self_play_probability', opponent_mix["self_play"])
                    self.model_set_for_self_play = True
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   Failed to enable self-play: {e}")
                        print(f"   Continuing with previous opponent type...")
            else:
                # Update self-play probability
                try:
                    self.training_env.env_method('set_self_play_probability', opponent_mix["self_play"])
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   Failed to update self-play probability: {e}")
        
        if self.verbose > 0:
            print(f"   Phase {phase_key} activated")
    
    def _update_phase_if_needed(self):
        """Check if we need to advance to the next curriculum phase and update if needed."""
        if self.current_phase >= len(self.phase_keys):
            return
            
        # Get current phase configuration
        current_phase_key = self.phase_keys[self.current_phase]
        current_phase = self.curriculum_config[current_phase_key]
        
        # Check if we need to advance to the next phase
        phase_duration = current_phase["timesteps"]
        
        # Skip phases with 0 timesteps (disabled phases)
        if phase_duration == 0:
            if self.current_phase + 1 < len(self.phase_keys):
                self.current_phase += 1
                self.phase_start_timestep = self.num_timesteps
                self._update_phase_if_needed()  # Check the next phase immediately
            return
        
        # Only advance if phase has a finite duration and we've exceeded it
        # Also check that we haven't finished training (avoid end-of-training transitions)
        if (phase_duration is not None and 
            phase_duration != float('inf') and 
            self.num_timesteps >= self.phase_start_timestep + phase_duration and
            self.total_timesteps is not None and 
            self.num_timesteps < self.total_timesteps):  # Don't transition at the very end
            
            # Time to advance to next phase
            if self.current_phase + 1 < len(self.phase_keys):
                # Check if the next phase is also disabled (0 timesteps)
                next_phase_key = self.phase_keys[self.current_phase + 1]
                next_phase_duration = self.curriculum_config[next_phase_key]["timesteps"]
                
                if next_phase_duration > 0:  # Only transition to active phases
                    if self.verbose > 0:
                        print(f"\nCURRICULUM PHASE TRANSITION at timestep {self.num_timesteps:,}")
                        print(f"   Transitioning from {current_phase_key} to {next_phase_key}")
                    
                    # Update phase and starting timestep
                    self.current_phase += 1
                    self.phase_start_timestep = self.num_timesteps
                    
                    # Reset value function to help adapt to new phase
                    self._reset_value_function()
                    
                    # Apply the new phase configuration
                    self._apply_phase_configuration()
        elif self.current_phase == 0 and self.num_timesteps == 0:
            # Initial phase setup
            self._apply_phase_configuration()
    
    def _scale_curriculum_phases(self):
        """Scale curriculum phases based on actual training timesteps."""
        # Get the original total timesteps from the config (sum of all phases)
        original_total = 0
        for phase_key in self.phase_keys:
            phase_timesteps = self.curriculum_config[phase_key]["timesteps"]
            if phase_timesteps is not None:
                original_total += phase_timesteps
        
        if original_total == 0:
            return  # Can't scale if we don't have original totals
        
        # Calculate scaling factor
        scale_factor = self.total_timesteps / original_total
        
        # Scale each phase duration
        for phase_key in self.phase_keys:
            if self.curriculum_config[phase_key]["timesteps"] is not None:
                original_duration = self.curriculum_config[phase_key]["timesteps"]
                scaled_duration = int(original_duration * scale_factor)
                # Ensure minimum phase duration of at least 1000 timesteps
                scaled_duration = max(scaled_duration, 1000)
                self.curriculum_config[phase_key]["timesteps"] = scaled_duration
        
        # For short training runs, we might want to limit to fewer phases
        if self.total_timesteps < 50_000:
            # For very short runs, only use phase 1 (random opponents)
            for i, phase_key in enumerate(self.phase_keys):
                if i == 0:
                    # First phase gets all the timesteps
                    self.curriculum_config[phase_key]["timesteps"] = self.total_timesteps
                else:
                    # Disable other phases by setting timesteps to 0
                    self.curriculum_config[phase_key]["timesteps"] = 0
        
        if self.verbose > 1:
            print(f"   Curriculum scaled for {self.total_timesteps:,} timesteps (factor: {scale_factor:.3f})")

    def _print_curriculum_schedule(self):
        """Print the full curriculum schedule."""
        print("Curriculum Schedule:")
        cumulative_timesteps = 0
        
        for i, phase_key in enumerate(self.phase_keys):
            phase_config = self.curriculum_config[phase_key]
            duration = phase_config["timesteps"]
            opponent_mix = phase_config["opponent_mix"]
            
            if duration is None or duration == float('inf'):
                print(f"  Phase {i+1} ({phase_key}): {cumulative_timesteps:,}+ timesteps")
            elif duration == 0:
                print(f"  Phase {i+1} ({phase_key}): DISABLED (0 timesteps)")
            else:
                end_timesteps = cumulative_timesteps + duration
                print(f"  Phase {i+1} ({phase_key}): {cumulative_timesteps:,} - {end_timesteps:,} timesteps")
                cumulative_timesteps = end_timesteps
            
            # Only show opponent mix for active phases
            if duration != 0:
                for opponent, prob in opponent_mix.items():
                    print(f"    {opponent}: {prob:.1%}")

    
    def _reset_value_function(self):
        """Reset the value function network to help adapt to new curriculum phases."""
        if self.verbose > 0:
            print("   🔄 Resetting value function for new curriculum phase")
        
        if hasattr(self.model, 'policy'):
            # Access the value network
            if hasattr(self.model.policy, 'value_net'):
                # Store the current weights for potential recovery
                old_weights = {name: param.clone() for name, param in 
                            self.model.policy.value_net.named_parameters()}
                
                # Reinitialize value network with new weights
                for layer_name, param in self.model.policy.value_net.named_parameters():
                    if 'weight' in layer_name:
                        # Initialize with small random values
                        param.data.normal_(0, 0.01)
                    elif 'bias' in layer_name:
                        param.data.zero_()
                
                if self.verbose > 0:
                    print("   ✅ Value function reset complete")

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
        print(f"EVALUATION AT TIMESTEP {self.num_timesteps:,}")
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
                print(f"  NEW BEST MODEL! Saved to {best_path}")
        
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
    
class GradientMonitorCallback(BaseCallback):
    """
    Callback for monitoring gradient norms during training.
    Helps detect gradient explosions early.
    """
    
    def __init__(self, warning_threshold=100.0, verbose=0):
        super(GradientMonitorCallback, self).__init__(verbose)
        self.warning_threshold = warning_threshold
        
    def _on_step(self) -> bool:
        # Only check periodically to avoid overhead
        if self.n_calls % 10 == 0:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                # Extract gradient norms
                total_norm = 0
                for param_group in self.model.policy.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2).item()
                            total_norm += param_norm ** 2
                            
                            # Check individual gradient norms for extreme values
                            if param_norm > self.warning_threshold:
                                print(f"WARNING: Parameter gradient norm explosion: {param_norm:.2f}")
                                
                total_norm = total_norm ** 0.5
                
                # Log to Tensorboard
                self.logger.record("train/gradient_norm", total_norm)
                
                # Alert on high gradient norms
                if total_norm > self.warning_threshold:
                    print(f"\nGRADIENT EXPLOSION DETECTED: norm = {total_norm:.2f}")
                    print(f"   Consider reducing the learning rate or adding gradient clipping")
                    
                    # If gradient norm is extremely high, you might want to take action
                    if total_norm > self.warning_threshold * 10:
                        print(f"EXTREME GRADIENT EXPLOSION: {total_norm:.2f}")
                        print(f"   Taking protective action: Saving emergency checkpoint")
                        emergency_path = os.path.join(
                            os.path.dirname(self.model.logger.dir), 
                            f"emergency_gradient_explosion_{self.num_timesteps}"
                        )
                        self.model.save(emergency_path)
        
        return True
    
class UpdateSelfPlayModelCallback(BaseCallback):
    """Efficient self-play model updates using shared memory."""
    
    def __init__(self, update_freq=50_000, model_dir=None, verbose=0):
        super(UpdateSelfPlayModelCallback, self).__init__(verbose)
        self.update_freq = update_freq
        self.model_dir = model_dir
        self.last_update = 0
        self.shared_weights = None
        
    def _on_training_start(self):
        """Initialize shared dictionary for weights."""
        # Create shared weights at the beginning
        self.shared_weights = {}
        for name, param in self.model.policy.state_dict().items():
            # Move to CPU for sharing and make shared
            self.shared_weights[name] = param.cpu().clone().share_memory_()
        
        # Initialize environments with the shared weights structure
        try:
            self.training_env.env_method('initialize_shared_weights', self.shared_weights)
            print("Self-play environments initialized with shared weights")
        except Exception as e:
            print(f"Failed to initialize shared weights: {e}")
        
    def _on_step(self) -> bool:
        # Check if it's time to update the model
        if (self.num_timesteps - self.last_update) >= self.update_freq:
            print(f"\nUPDATING SELF-PLAY MODEL at timestep {self.num_timesteps:,}")
            
            try:
                # Update the shared weights with current model weights
                with torch.no_grad():
                    for name, param in self.model.policy.state_dict().items():
                        if name in self.shared_weights:
                            self.shared_weights[name].copy_(param.cpu())
                
                # Signal environments that weights were updated
                self.training_env.env_method('notify_weights_updated')
                print(f"   Self-play model updated at step {self.num_timesteps:,}")
                self.last_update = self.num_timesteps
                
            except Exception as e:
                print(f"   Failed to update self-play model: {e}")
            
        return True