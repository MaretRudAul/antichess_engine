import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union, Any

class ChessCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Antichess.
    
    Architecture inspired by AlphaZero's residual network, but simplified:
    1. Initial convolutional layer
    2. Several residual blocks
    3. Policy and value heads
    """
    
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 256):
        """
        Initialize the CNN.
        
        Args:
            observation_space: The observation space
            features_dim: Dimension of the extracted features
        """
        super(ChessCNN, self).__init__(observation_space, features_dim)
        
        # Handle Dict observation space
        if isinstance(observation_space, gym.spaces.Dict):
            # Extract the Box space with the actual observation
            actual_observation_space = observation_space["observation"]
        else:
            actual_observation_space = observation_space
        
        # Input is a 13x8x8 tensor (12 piece planes + 1 turn plane)
        n_input_channels = actual_observation_space.shape[0]
        
        # CNN implementation
        self.cnn = nn.Sequential(
            # Initial convolutional layer
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Residual block 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Residual block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final layers
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of the flattened features
        # 8x8 board with 64 filters
        cnn_output_size = 64 * 8 * 8
        
        # Project to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Batch of observations (can be dict or tensor)
            
        Returns:
            Tensor of extracted features
        """
        # If observations is a dict, extract the actual observation tensor
        if isinstance(observations, dict):
            observations = observations["observation"]
            
        features = self.cnn(observations)
        return self.fc(features)

class CustomAntichessPolicy(ActorCriticPolicy):
    """
    Custom policy for Antichess that handles action masking.
    This ensures that only legal moves can be selected during training and evaluation.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        *args,
        **kwargs,
    ):
        """Initialize the policy."""
        # Use our custom feature extractor
        kwargs["features_extractor_class"] = ChessCNN
        kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        super(CustomAntichessPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        
        Args:
            observation: The input observation
            deterministic: Whether to use a deterministic action
            
        Returns:
            The selected action
        """
        # Get action mask from the observation (last part of the observation)
        if isinstance(observation, dict) and "action_mask" in observation:
            action_mask = observation["action_mask"]
        else:
            # If no mask is provided, assume all actions are valid
            action_mask = torch.ones(self.action_space.n)
            
        # Get features
        latent_pi, latent_vf = self._get_latent(observation)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Apply action mask by setting probabilities of invalid actions to near-zero
        if hasattr(distribution, "distribution") and hasattr(distribution.distribution, "probs"):
            # For categorical distributions (typical for discrete action spaces)
            masked_probs = distribution.distribution.probs * action_mask
            masked_probs = masked_probs / (masked_probs.sum() + 1e-10)  # Normalize
            distribution.distribution.probs = masked_probs
            
        # Choose action
        if deterministic:
            if hasattr(distribution, "mode"):
                actions = distribution.mode()
            else:
                # For categorical, use argmax of masked probs
                actions = torch.argmax(masked_probs, dim=1)
        else:
            actions = distribution.sample()
            
        return actions

class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy that supports action masking.
    
    This extends the standard SB3 ActorCriticPolicy to handle action masks
    provided with the observations.
    """
    
    def _get_latent(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get latent features from the observation.
        
        Args:
            obs: The observation tensor or dictionary
            
        Returns:
            Tuple of (policy_features, value_features)
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.policy_net(features)
        latent_vf = self.mlp_extractor.value_net(features)
        return latent_pi, latent_vf
    
    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            obs: The observation, which can be a tensor or a dictionary with 
                 an 'action_mask' key
            deterministic: Whether to use a deterministic action
            
        Returns:
            actions, values, log_probs
        """
        # Extract observation and action mask
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # Convert action_mask to tensor if it's numpy
            if action_mask is not None and isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).float()
        else:
            action_mask = None
            
        # Get latent features
        latent_pi, latent_vf = self._get_latent(obs)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Make sure action_mask is on the same device as the logits
            if hasattr(distribution, "distribution") and hasattr(distribution.distribution, "logits"):
                device = distribution.distribution.logits.device
                action_mask = action_mask.to(device)
                
                # Apply mask by setting invalid actions to very negative values
                masked_logits = distribution.distribution.logits.clone()
                # Where action_mask is 0, set logits to very negative values
                masked_logits = torch.where(action_mask > 0, masked_logits, torch.full_like(masked_logits, -1e8))
                distribution.distribution.logits = masked_logits
        
        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        log_probs = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs