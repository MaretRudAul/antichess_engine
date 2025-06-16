from dataclasses import dataclass
from typing import Dict, Any
import os
import torch

@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    input_channels: int = 19
    residual_blocks: int = 12
    filters: int = 256
    policy_head_size: int = 4096
    value_head_size: int = 256
    dropout_rate: float = 0.1

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10
    games_per_iteration: int = 1000
    evaluation_games: int = 100
    
@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search parameters"""
    simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 1.0
    temperature_threshold: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

@dataclass
class SystemConfig:
    """System and infrastructure settings"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    redis_host: str = "localhost"
    redis_port: int = 6379
    mongodb_uri: str = "mongodb://localhost:27017/"
    wandb_project: str = "antichess-rl"
    checkpoint_interval: int = 100
