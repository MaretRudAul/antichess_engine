"""
AlphaZero-style Neural Network for Antichess

This module implements a deep convolutional neural network following the AlphaZero
architecture, specifically adapted for antichess. The network combines policy and
value estimation in a shared representation learning framework.

Architecture:
- Input: 8x8x19 board representation tensor
- Backbone: ResNet-style convolutional layers with residual connections
- Policy Head: Outputs move probabilities for all 4096 possible moves
- Value Head: Outputs position evaluation (-1 to +1)

Key adaptations for antichess:
- Optimized for forced capture scenarios
- Enhanced position evaluation for piece sacrifice objectives
- Robust handling of unusual material imbalances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block for deep CNN with batch normalization and ReLU activation.
    
    Implements the standard ResNet building block:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """
    
    def __init__(self, channels: int):
        """
        Initialize residual block.
        
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channels, 8, 8)
            
        Returns:
            Output tensor of same shape with residual connection
        """
        residual = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        
        # Final activation
        return F.relu(out)


class AntichessNet(nn.Module):
    """
    AlphaZero-style CNN for antichess position evaluation and move prediction.
    
    This network follows the dual-head architecture of AlphaZero with adaptations
    for antichess gameplay. It processes board positions and outputs both move
    probabilities and position values.
    
    Architecture:
    1. Input convolution layer
    2. Stack of residual blocks
    3. Policy head for move prediction
    4. Value head for position evaluation
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize antichess neural network.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__()
        
        self.config = config
        
        # Input convolution - transforms 19-channel input to filter channels
        self.input_conv = nn.Conv2d(
            config.input_channels, 
            config.filters, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.filters)
        
        # Residual tower - stack of residual blocks for deep feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.filters) 
            for _ in range(config.residual_blocks)
        ])
        
        # Policy head - predicts move probabilities
        self.policy_conv = nn.Conv2d(config.filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_head_size)
        self.policy_output = nn.Linear(config.policy_head_size, 4096)  # 64x64 move encoding
        
        # Value head - predicts position evaluation
        self.value_conv = nn.Conv2d(config.filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, config.value_head_size)
        self.value_fc2 = nn.Linear(config.value_head_size, 1)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized AntichessNet with {self._count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value estimate.
        
        Args:
            x: Board representation tensor of shape (batch, 19, 8, 8)
            
        Returns:
            Tuple of (policy_logits, value) where:
            - policy_logits: Raw logits for 4096 possible moves (batch, 4096)
            - value: Position evaluation in range (-1, 1) (batch, 1)
        """
        batch_size = x.size(0)
        
        # Input convolution with batch norm and activation
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Pass through residual tower
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Policy head computation
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        
        # Flatten and pass through policy fully connected layers
        policy = policy.view(batch_size, -1)  # Shape: (batch, 32*8*8)
        policy = self.policy_fc(policy)
        policy = F.relu(policy)
        policy = self.dropout(policy)  # Apply dropout before final layer
        policy_logits = self.policy_output(policy)  # Shape: (batch, 4096)
        
        # Value head computation
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        
        # Flatten and pass through value fully connected layers
        value = value.view(batch_size, -1)  # Shape: (batch, 8*8)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.dropout(value)  # Apply dropout before final layer
        value = self.value_fc2(value)  # Shape: (batch, 1)
        value = torch.tanh(value)  # Squash to (-1, 1) range
        
        return policy_logits, value
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction interface for inference (no gradients).
        
        Args:
            x: Board representation tensor
            
        Returns:
            Tuple of (policy_probs, value) with softmax applied to policy
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
            return policy_probs, value
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                # Standard batch norm initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get detailed model information for logging and analysis."""
        # Count parameters by type
        conv_params = sum(p.numel() for m in self.modules() 
                         if isinstance(m, nn.Conv2d) for p in m.parameters())
        bn_params = sum(p.numel() for m in self.modules() 
                       if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
        fc_params = sum(p.numel() for m in self.modules() 
                       if isinstance(m, nn.Linear) for p in m.parameters())
        
        return {
            'total_parameters': self._count_parameters(),
            'conv_parameters': conv_params,
            'bn_parameters': bn_params,
            'fc_parameters': fc_params,
            'residual_blocks': self.config.residual_blocks,
            'filters': self.config.filters,
            'input_channels': self.config.input_channels,
            'policy_head_size': self.config.policy_head_size,
            'value_head_size': self.config.value_head_size,
            'dropout_rate': self.config.dropout_rate
        }
    
    def save_checkpoint(self, filepath: str, optimizer_state: dict = None, 
                       training_info: dict = None):
        """
        Save model checkpoint with training state.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optimizer state dict (optional)
            training_info: Additional training information (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__,
            'model_info': self.get_model_info()
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if training_info:
            checkpoint['training_info'] = training_info
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Tuple of (model, optimizer_state, training_info)
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model config from checkpoint
        config = ModelConfig(**checkpoint['model_config'])
        
        # Create and load model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        optimizer_state = checkpoint.get('optimizer_state_dict')
        training_info = checkpoint.get('training_info', {})
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return model, optimizer_state, training_info


def create_antichess_model(config: ModelConfig = None, device: str = None) -> AntichessNet:
    """
    Factory function to create antichess neural network.
    
    Args:
        config: Model configuration (uses default if None)
        device: Device to place model on (auto-detects if None)
        
    Returns:
        Initialized AntichessNet model
    """
    if config is None:
        config = ModelConfig()
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AntichessNet(config)
    model.to(device)
    
    logger.info(f"Created antichess model on {device}")
    return model


# Export the main classes
__all__ = ['AntichessNet', 'ResidualBlock', 'create_antichess_model']
