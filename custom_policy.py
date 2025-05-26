import gym
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
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initialize the CNN.
        
        Args:
            observation_space: The observation space
            features_dim: Dimension of the extracted features
        """
        super(ChessCNN, self).__init__(observation_space, features_dim)
        
        # Input is a 13x8x8 tensor (12 piece planes + 1 turn plane)
        n_input_channels = observation_space.shape[0]
        
        # Define the CNN feature extractor
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
            observations: Batch of observations
            
        Returns:
            Tensor of extracted features
        """
        features = self.cnn(observations)
        return self.fc(features)

