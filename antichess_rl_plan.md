# Antichess Reinforcement Learning Project - Complete Implementation Plan

## Project Overview

**Objective**: Build a professional-grade reinforcement learning system that masters antichess (losing chess) using self-play and modern deep learning techniques.

**Target Performance**: Achieve superhuman performance in antichess through AlphaZero-style training with custom adaptations for antichess rule variants.

## Tech Stack

### Core Framework
- **Python 3.9+**
- **PyTorch 2.0+** with CUDA support
- **PyTorch Lightning** for structured training
- **Weights & Biases** for experiment tracking
- **Ray RLlib** for distributed training
- **python-chess** for game engine foundation

### Infrastructure
- **Docker** for containerization
- **Redis** for caching and message passing
- **MongoDB** for game storage
- **pytest** for testing
- **Black + isort** for code formatting
- **mypy** for type checking

### ML/RL Libraries
- **NumPy/PyTorch** for tensor operations
- **OpenAI Gym** for environment interface
- **MCTS** (custom implementation)
- **Stable-Baselines3** for baseline comparisons

## Project Structure

```
antichess-rl/
├── src/
│   ├── __init__.py
│   ├── game/
│   │   ├── __init__.py
│   │   ├── antichess_rules.py
│   │   ├── board_representation.py
│   │   └── move_generation.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── antichess_env.py
│   │   └── gym_wrapper.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   ├── mcts.py
│   │   └── alphazero_agent.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── self_play.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── tests/
├── scripts/
├── data/
├── models/
├── logs/
├── docker/
├── requirements.txt
├── setup.py
├── README.md
└── .github/workflows/
```

## Component Specifications

### 1. Game Engine (`src/game/`)
- **Antichess Rules**: Implement forced captures, win by losing all pieces
- **Board Representation**: 8x8x19 tensor format for neural network input
- **Move Generation**: Legal move generation with forced capture priority
- **Game State Management**: Position evaluation, game termination detection

### 2. Environment (`src/environment/`)
- **Gym Interface**: Standard RL environment API
- **State Representation**: Normalized board tensors
- **Action Space**: Move encoding/decoding
- **Reward Function**: Antichess-specific reward shaping

### 3. Neural Network (`src/models/`)
- **Architecture**: ResNet-style CNN with policy/value heads
- **Input**: 8x8x19 board representation
- **Output**: Policy (move probabilities) + Value (position evaluation)
- **Loss Functions**: Cross-entropy (policy) + MSE (value)

### 4. MCTS (`src/models/mcts.py`)
- **Tree Search**: Upper Confidence Bound applied to trees
- **Node Expansion**: Neural network guided expansion
- **Backup**: Value propagation with antichess adaptations
- **Move Selection**: Temperature-based sampling

### 5. Training System (`src/training/`)
- **Self-Play**: Distributed game generation
- **Experience Buffer**: Efficient storage and sampling
- **Training Loop**: Batch updates with experience replay
- **Evaluation**: Tournament-style assessment

## Detailed Implementation Steps

### Phase 1: Foundation Setup (Steps 1-10)

#### Step 1: Project Initialization
```bash
# Copilot Task: Set up project structure
mkdir antichess-rl && cd antichess-rl
mkdir -p src/{game,environment,models,training,utils,config}
mkdir -p tests/{unit,integration}
mkdir -p {scripts,data,models,logs,docker}
touch requirements.txt setup.py README.md
```

#### Step 2: Dependencies Configuration
```python
# File: requirements.txt
# Copilot Task: Create comprehensive requirements file
torch>=2.0.0
pytorch-lightning>=2.0.0
wandb>=0.15.0
ray[rllib]>=2.5.0
python-chess>=1.999
gymnasium>=0.28.0
numpy>=1.24.0
pandas>=2.0.0
pymongo>=4.0.0
redis>=4.5.0
pytest>=7.0.0
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0
tensorboard>=2.12.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

#### Step 3: Configuration Management
```python
# File: src/config/settings.py
# Copilot Task: Create comprehensive configuration system
from dataclasses import dataclass
from typing import Dict, Any
import os

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
```

#### Step 4: Game Rules Implementation
```python
# File: src/game/antichess_rules.py
# Copilot Task: Implement complete antichess rule system
import chess
from typing import List, Optional, Tuple
from enum import Enum

class GameResult(Enum):
    ONGOING = 0
    WHITE_WINS = 1  # White has no pieces left or is stalemated
    BLACK_WINS = 2  # Black has no pieces left or is stalemated
    DRAW = 3        # Rare in antichess

class AntichessGame:
    """
    Complete antichess game implementation with forced capture rules
    """
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self.position_history: List[str] = []
    
    def get_legal_moves(self) -> List[chess.Move]:
        """
        Get legal moves with forced capture rule enforcement
        Returns: List of legal moves (captures only if captures available)
        """
        # Implementation details for forced captures
        pass
    
    def make_move(self, move: chess.Move) -> bool:
        """
        Execute a move with antichess rule validation
        Returns: True if move was legal and executed
        """
        pass
    
    def is_game_over(self) -> Tuple[bool, GameResult]:
        """
        Check if game is finished according to antichess rules
        Returns: (is_over, result)
        """
        pass
    
    def get_result(self) -> GameResult:
        """Get current game result"""
        pass
    
    def clone(self) -> 'AntichessGame':
        """Create deep copy of game state"""
        pass
```

#### Step 5: Board Representation
```python
# File: src/game/board_representation.py
# Copilot Task: Implement neural network input representation
import numpy as np
import chess
from typing import np.ndarray

class BoardEncoder:
    """
    Convert chess board to neural network input tensor
    Format: 8x8x19 (12 piece planes + 7 metadata planes)
    """
    
    def __init__(self):
        self.piece_to_plane = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
    
    def encode_board(self, board: chess.Board) -> np.ndarray:
        """
        Encode board position as 8x8x19 tensor
        Planes 0-5: White pieces (P,R,N,B,Q,K)
        Planes 6-11: Black pieces (P,R,N,B,Q,K)
        Plane 12: Turn (1 for white, 0 for black)
        Plane 13: Castling rights (white kingside)
        Plane 14: Castling rights (white queenside)
        Plane 15: Castling rights (black kingside)
        Plane 16: Castling rights (black queenside)
        Plane 17: En passant target square
        Plane 18: Move count (normalized)
        """
        pass
    
    def encode_moves(self, moves: List[chess.Move]) -> np.ndarray:
        """Encode legal moves as action mask"""
        pass
    
    def decode_move(self, move_index: int, board: chess.Board) -> chess.Move:
        """Convert neural network output to chess move"""
        pass
```

#### Step 6: Environment Implementation
```python
# File: src/environment/antichess_env.py
# Copilot Task: Create gymnasium-compatible environment
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from ..game.antichess_rules import AntichessGame, GameResult
from ..game.board_representation import BoardEncoder

class AntichessEnv(gym.Env):
    """
    Antichess environment following Gymnasium interface
    """
    
    def __init__(self, opponent=None):
        super().__init__()
        
        # Action space: all possible moves (4096 for 64x64 from-to encoding)
        self.action_space = spaces.Discrete(4096)
        
        # Observation space: 8x8x19 board representation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 19), dtype=np.float32
        )
        
        self.game = AntichessGame()
        self.encoder = BoardEncoder()
        self.opponent = opponent
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        pass
    
    def render(self, mode='human'):
        """Render current game state"""
        pass
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of legal actions"""
        pass
    
    def _calculate_reward(self, result: GameResult) -> float:
        """Calculate reward based on antichess objectives"""
        pass
```

#### Step 7: Neural Network Architecture
```python
# File: src/models/neural_network.py
# Copilot Task: Implement AlphaZero-style CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..config.settings import ModelConfig

class ResidualBlock(nn.Module):
    """Residual block for deep CNN"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AntichessNet(nn.Module):
    """
    AlphaZero-style CNN for antichess
    Architecture: Conv -> ResBlocks -> Policy/Value heads
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Input convolution
        self.input_conv = nn.Conv2d(
            config.input_channels, config.filters, 3, padding=1
        )
        self.input_bn = nn.BatchNorm2d(config.filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.filters) 
            for _ in range(config.residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(config.filters, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_head_size)
        self.policy_output = nn.Linear(config.policy_head_size, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(config.filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, config.value_head_size)
        self.value_fc2 = nn.Linear(config.value_head_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy and value
        Args: x - board representation (batch, 19, 8, 8)
        Returns: (policy_logits, value)
        """
        pass
```

#### Step 8: MCTS Implementation
```python
# File: src/models/mcts.py
# Copilot Task: Implement Monte Carlo Tree Search
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import chess
from ..game.antichess_rules import AntichessGame
from ..config.settings import MCTSConfig

class MCTSNode:
    """Node in MCTS tree"""
    
    def __init__(self, state: AntichessGame, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False
    
    def value(self) -> float:
        """Average value of node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Upper Confidence Bound score"""
        pass
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select best child using UCB"""
        pass
    
    def expand(self, policy_probs: np.ndarray, legal_moves: List[chess.Move]):
        """Expand node with children"""
        pass
    
    def backup(self, value: float):
        """Backup value through tree"""
        pass

class MCTS:
    """Monte Carlo Tree Search for antichess"""
    
    def __init__(self, neural_net, config: MCTSConfig):
        self.neural_net = neural_net
        self.config = config
    
    def search(self, root_state: AntichessGame) -> Dict[chess.Move, float]:
        """
        Run MCTS simulations and return move probabilities
        Returns: Dictionary mapping moves to visit probabilities
        """
        pass
    
    def _simulate(self, node: MCTSNode) -> float:
        """Single MCTS simulation"""
        pass
    
    def get_move_probs(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """Convert visit counts to move probabilities"""
        pass
```

#### Step 9: AlphaZero Agent
```python
# File: src/models/alphazero_agent.py
# Copilot Task: Implement complete AlphaZero agent
import torch
import numpy as np
from typing import Tuple, Dict, List
import chess
from .neural_network import AntichessNet
from .mcts import MCTS
from ..game.antichess_rules import AntichessGame
from ..game.board_representation import BoardEncoder
from ..config.settings import ModelConfig, MCTSConfig

class AlphaZeroAgent:
    """
    Complete AlphaZero agent for antichess
    Combines neural network with MCTS for move selection
    """
    
    def __init__(self, model_config: ModelConfig, mcts_config: MCTSConfig):
        self.model_config = model_config
        self.mcts_config = mcts_config
        
        self.neural_net = AntichessNet(model_config)
        self.mcts = MCTS(self.neural_net, mcts_config)
        self.encoder = BoardEncoder()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_net.to(self.device)
    
    def select_move(self, game: AntichessGame, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        """
        Select best move using MCTS
        Returns: (chosen_move, move_probabilities)
        """
        pass
    
    def evaluate_position(self, game: AntichessGame) -> Tuple[np.ndarray, float]:
        """
        Evaluate position with neural network
        Returns: (policy_probs, value_estimate)
        """
        pass
    
    def self_play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play complete self-play game
        Returns: List of (state, policy, value) training examples
        """
        pass
    
    def load_checkpoint(self, filepath: str):
        """Load model weights from checkpoint"""
        pass
    
    def save_checkpoint(self, filepath: str):
        """Save model weights to checkpoint"""
        pass
```

#### Step 10: Testing Framework
```python
# File: tests/test_game_rules.py
# Copilot Task: Comprehensive test suite for game logic
import pytest
import chess
from src.game.antichess_rules import AntichessGame, GameResult
from src.game.board_representation import BoardEncoder

class TestAntichessRules:
    """Test antichess rule implementation"""
    
    def test_forced_captures(self):
        """Test that captures are forced when available"""
        pass
    
    def test_win_conditions(self):
        """Test antichess win/loss conditions"""
        pass
    
    def test_move_generation(self):
        """Test legal move generation"""
        pass
    
    def test_game_termination(self):
        """Test game over detection"""
        pass

class TestBoardRepresentation:
    """Test board encoding/decoding"""
    
    def test_board_encoding(self):
        """Test board to tensor conversion"""
        pass
    
    def test_move_encoding(self):
        """Test move to action conversion"""
        pass
    
    def test_symmetry_preservation(self):
        """Test that encoding preserves game state"""
        pass

# File: tests/test_neural_network.py
class TestNeuralNetwork:
    """Test neural network architecture"""
    
    def test_forward_pass(self):
        """Test network forward pass"""
        pass
    
    def test_output_shapes(self):
        """Test policy and value output shapes"""
        pass
    
    def test_gradient_flow(self):
        """Test gradient computation"""
        pass
```

### Phase 2: Training Infrastructure (Steps 11-20)

#### Step 11: Self-Play System
```python
# File: src/training/self_play.py
# Copilot Task: Implement distributed self-play generation
import multiprocessing as mp
from typing import List, Tuple
import numpy as np
from ..models.alphazero_agent import AlphaZeroAgent
from ..config.settings import TrainingConfig
import wandb

class SelfPlayWorker:
    """Worker process for self-play game generation"""
    
    def __init__(self, agent: AlphaZeroAgent, worker_id: int):
        self.agent = agent
        self.worker_id = worker_id
    
    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate multiple self-play games"""
        pass
    
    def play_single_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Play one complete self-play game"""
        pass

class SelfPlayManager:
    """Manages distributed self-play generation"""
    
    def __init__(self, agent: AlphaZeroAgent, config: TrainingConfig, num_workers: int = 4):
        self.agent = agent
        self.config = config
        self.num_workers = num_workers
    
    def generate_training_data(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate training data using multiprocessing
        Returns: List of (state, policy_target, value_target) tuples
        """
        pass
    
    def _worker_process(self, games_per_worker: int, result_queue: mp.Queue):
        """Worker process function"""
        pass
```

#### Step 12: Training Loop
```python
# File: src/training/trainer.py
# Copilot Task: Implement complete training system
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Dict
import wandb
from ..models.alphazero_agent import AlphaZeroAgent
from ..config.settings import TrainingConfig, ModelConfig, MCTSConfig
from .self_play import SelfPlayManager
from .evaluator import ModelEvaluator

class AntichessTrainer:
    """Main training class for antichess RL agent"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, mcts_config: MCTSConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.mcts_config = mcts_config
        
        # Initialize agent and training components
        self.agent = AlphaZeroAgent(model_config, mcts_config)
        self.self_play_manager = SelfPlayManager(self.agent, training_config)
        self.evaluator = ModelEvaluator()
        
        # Training setup
        self.optimizer = optim.Adam(
            self.agent.neural_net.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Tracking
        self.iteration = 0
        self.best_model_path = None
        
    def train(self, num_iterations: int):
        """
        Main training loop
        Each iteration: self-play -> training -> evaluation
        """
        wandb.init(project="antichess-rl")
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Phase 1: Generate self-play data
            print(f"Iteration {iteration}: Generating self-play data...")
            training_data = self.self_play_manager.generate_training_data(
                self.training_config.games_per_iteration
            )
            
            # Phase 2: Train neural network
            print(f"Iteration {iteration}: Training neural network...")
            train_metrics = self._train_network(training_data)
            
            # Phase 3: Evaluate model
            print(f"Iteration {iteration}: Evaluating model...")
            eval_metrics = self.evaluator.evaluate_model(self.agent)
            
            # Phase 4: Log and save
            self._log_metrics(train_metrics, eval_metrics)
            self._save_checkpoint(iteration)
            
            # Update learning rate
            self.scheduler.step()
    
    def _train_network(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> Dict:
        """Train neural network on self-play data"""
        pass
    
    def _create_dataloader(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> DataLoader:
        """Create PyTorch DataLoader from training data"""
        pass
    
    def _compute_loss(self, batch) -> Tuple[torch.Tensor, Dict]:
        """Compute policy and value losses"""
        pass
    
    def _log_metrics(self, train_metrics: Dict, eval_metrics: Dict):
        """Log training metrics to Weights & Biases"""
        pass
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        pass
```

#### Step 13: Model Evaluation
```python
# File: src/training/evaluator.py
# Copilot Task: Implement model evaluation system
import numpy as np
from typing import Dict, List, Tuple
import chess
from ..models.alphazero_agent import AlphaZeroAgent
from ..game.antichess_rules import AntichessGame, GameResult
import wandb

class ModelEvaluator:
    """Evaluate model performance through tournaments and metrics"""
    
    def __init__(self):
        self.baseline_agents = {}
        self.evaluation_history = []
    
    def evaluate_model(self, agent: AlphaZeroAgent, num_games: int = 100) -> Dict:
        """
        Comprehensive model evaluation
        Returns: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Self-play evaluation
        metrics.update(self._evaluate_self_play(agent, num_games // 2))
        
        # Baseline comparisons
        metrics.update(self._evaluate_against_baselines(agent, num_games // 2))
        
        # Position analysis
        metrics.update(self._evaluate_position_understanding(agent))
        
        return metrics
    
    def _evaluate_self_play(self, agent: AlphaZeroAgent, num_games: int) -> Dict:
        """Evaluate through self-play games"""
        pass
    
    def _evaluate_against_baselines(self, agent: AlphaZeroAgent, num_games: int) -> Dict:
        """Evaluate against baseline agents (random, greedy, etc.)"""
        pass
    
    def _evaluate_position_understanding(self, agent: AlphaZeroAgent) -> Dict:
        """Evaluate position evaluation accuracy"""
        pass
    
    def play_tournament_game(self, agent1: AlphaZeroAgent, agent2: AlphaZeroAgent) -> GameResult:
        """Play single tournament game between two agents"""
        pass
    
    def calculate_elo_rating(self, wins: int, losses: int, draws: int, opponent_elo: float = 1500) -> float:
        """Calculate ELO rating based on results"""
        pass

class BaselineAgent:
    """Simple baseline agents for comparison"""
    
    @staticmethod
    def random_agent(game: AntichessGame) -> chess.Move:
        """Select random legal move"""
        pass
    
    @staticmethod
    def greedy_capture_agent(game: AntichessGame) -> chess.Move:
        """Prefer captures, then random"""
        pass
    
    @staticmethod
    def piece_sacrifice_agent(game: AntichessGame) -> chess.Move:
        """Try to sacrifice pieces quickly"""
        pass
```

#### Step 14: Data Management
```python
# File: src/utils/data_utils.py
# Copilot Task: Implement data storage and management
import pickle
import json
import numpy as np
from typing import List, Tuple, Dict, Any
import pymongo
import redis
from pathlib import Path
import chess
from ..game.antichess_rules import AntichessGame

class GameDatabase:
    """MongoDB interface for storing games and training data"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client.antichess_rl
        self.games_collection = self.db.games
        self.training_data_collection = self.db.training_data
    
    def store_game(self, game_data: Dict[str, Any]) -> str:
        """Store complete game with metadata"""
        pass
    
    def store_training_examples(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[str]:
        """Store training examples from self-play"""
        pass
    
    def get_recent_games(self, limit: int = 1000) -> List[Dict]:
        """Retrieve recent games for analysis"""
        pass
    
    def get_training_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Get random batch of training examples"""
        pass
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about stored games"""
        pass

class ExperienceBuffer:
    """Redis-based circular buffer for training data"""
    
    def __init__(self, max_size: int = 1000000, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.max_size = max_size
        self.current_size = 0
        self.buffer_key = "experience_buffer"
    
    def add_experience(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add single experience to buffer"""
        pass
    
    def add_batch(self, experiences: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add batch of experiences to buffer"""
        pass
    
    def sample_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Sample random batch from buffer"""
        pass
    
    def clear_buffer(self):
        """Clear all experiences from buffer"""
        pass
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        pass

class DataProcessor:
    """Utilities for processing and augmenting training data"""
    
    @staticmethod
    def augment_data(states: List[np.ndarray], policies: List[np.ndarray], values: List[float]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Apply data augmentation (rotations, flips) to training data"""
        pass
    
    @staticmethod
    def normalize_states(states: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize state representations"""
        pass
    
    @staticmethod
    def balance_dataset(data: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Balance dataset by game outcome"""
        pass
    
    @staticmethod
    def save_training_data(data: List[Tuple[np.ndarray, np.ndarray, float]], filepath: str):
        """Save training data to disk"""
        pass
    
    @staticmethod
    def load_training_data(filepath: str) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Load training data from disk"""
        pass
```

#### Step 15: Visualization and Analysis
```python
# File: src/utils/visualization.py
# Copilot Task: Implement comprehensive visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import chess
import chess.svg
from ..game.antichess_rules import AntichessGame
from ..models.alphazero_agent import AlphaZeroAgent
import wandb

class TrainingVisualizer:
    """Visualize training progress and metrics"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_curves(self, metrics_history: Dict[str, List[float]]):
        """Plot training loss and evaluation metrics over time"""
        pass
    
    def plot_elo_progression(self, elo_history: List[float]):
        """Plot ELO rating progression"""
        pass
    
    def plot_game_length_distribution(self, game_lengths: List[int]):
        """Plot distribution of game lengths"""
        pass
    
    def plot_move_frequency_heatmap(self, move_counts: Dict[str, int]):
        """Plot heatmap of most frequent moves"""
        pass
    
    def plot_position_evaluation_comparison(self, positions: List[str], human_evals: List[float], ai_evals: List[float]):
        """Compare position evaluations between human and AI"""
        pass

class GameAnalyzer:
    """Analyze individual games and positions"""
    
    def __init__(self, agent: AlphaZeroAgent):
        self.agent = agent
    
    def analyze_game(self, game: AntichessGame) -> Dict[str, Any]:
        """Complete game analysis with move annotations"""
        pass
    
    def visualize_position(self, game: AntichessGame) -> str:
        """Generate SVG visualization of position"""
        pass
    
    def get_move_probabilities(self, game: AntichessGame) -> Dict[str, float]:
        """Get neural network move probabilities for position"""
        pass
    
    def analyze_critical_positions(self, games: List[AntichessGame]) -> List[Dict]:
        """Identify and analyze critical positions from games"""
        pass
    
    def generate_opening_tree(self, games: List[AntichessGame], depth: int = 10) -> Dict:
        """Generate opening tree from game collection"""
        pass

class PerformanceProfiler:
    """Profile system performance and bottlenecks"""
    
    @staticmethod
    def profile_mcts_speed(agent: AlphaZeroAgent, num_positions: int = 100) -> Dict[str, float]:
        """Profile MCTS search speed"""
        pass
    
    @staticmethod
    def profile_neural_network_inference(agent: AlphaZeroAgent, batch_sizes: List[int]) -> Dict[int, float]:
        """Profile neural network inference speed"""
        pass
    
    @staticmethod
    def profile_self_play_generation(agent: AlphaZeroAgent, num_games: int = 10) -> Dict[str, float]:
        """Profile self-play game generation speed"""
        pass
```

#### Step 16: Metrics and Logging
```python
# File: src/utils/metrics.py
# Copilot Task: Implement comprehensive metrics tracking
import numpy as np
from typing import Dict, List, Any, Tuple
import time
from collections import defaultdict
import chess
from ..game.antichess_rules import AntichessGame, GameResult
import wandb

class MetricsTracker:
    """Track and compute various training and evaluation metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
    
    def log_training_metrics(self, epoch: int, policy_loss: float, value_loss: float, total_loss: float, learning_rate: float):
        """Log training metrics for current epoch"""
        metrics = {
            'epoch': epoch,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'total_loss': total_loss,
            'learning_rate': learning_rate,
            'timestamp': time.time() - self.start_time
        }
        
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        wandb.log(metrics)
    
    def log_self_play_metrics(self, games_played: int, avg_game_length: float, avg_pieces_captured: float, win_rate: float):
        """Log self-play generation metrics"""
        pass
    
    def log_evaluation_metrics(self, elo_rating: float, win_rate_vs_random: float, win_rate_vs_greedy: float, position_accuracy: float):
        """Log model evaluation metrics"""
        pass
    
    def log_system_metrics(self, gpu_utilization: float, memory_usage: float, games_per_second: float):
        """Log system performance metrics"""
        pass
    
    def compute_game_statistics(self, games: List[AntichessGame]) -> Dict[str, Any]:
        """Compute comprehensive statistics from game collection"""
        pass
    
    def compute_move_diversity(self, games: List[AntichessGame]) -> float:
        """Compute move diversity metric"""
        pass
    
    def compute_tactical_accuracy(self, positions: List[Tuple[AntichessGame, chess.Move]], agent_moves: List[chess.Move]) -> float:
        """Compute tactical accuracy on test positions"""
        pass

class WandbLogger:
    """Weights & Biases integration for experiment tracking"""
    
    def __init__(self, project_name: str = "antichess-rl", run_name: str = None):
        self.project_name = project_name
        self.run_name = run_name
        self.run = None
    
    def initialize_run(self, config: Dict[str, Any]):
        """Initialize W&B run with configuration"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=config
        )
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to W&B"""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_game_video(self, game: AntichessGame, filename: str):
        """Log game visualization as video"""
        pass
    
    def log_model_architecture(self, model):
        """Log model architecture to W&B"""
        if self.run:
            wandb.watch(model)
    
    def save_model_artifact(self, model_path: str, alias: str = "latest"):
        """Save model as W&B artifact"""
        pass
    
    def finish_run(self):
        """Finish W&B run"""
        if self.run:
            wandb.finish()
```

#### Step 17: Configuration and CLI
```python
# File: scripts/train.py
# Copilot Task: Create training script with CLI interface
import argparse
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import ModelConfig, TrainingConfig, MCTSConfig, SystemConfig
from src.training.trainer import AntichessTrainer
from src.utils.metrics import WandbLogger
import torch

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_configs_from_dict(config_dict: Dict[str, Any]) -> Tuple[ModelConfig, TrainingConfig, MCTSConfig, SystemConfig]:
    """Create config objects from dictionary"""
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    mcts_config = MCTSConfig(**config_dict.get('mcts', {}))
    system_config = SystemConfig(**config_dict.get('system', {}))
    
    return model_config, training_config, mcts_config, system_config

def main():
    parser = argparse.ArgumentParser(description='Train Antichess RL Agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--run-name', type=str, help='W&B run name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config_dict = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config_dict = {}
    
    # Create config objects
    model_config, training_config, mcts_config, system_config = create_configs_from_dict(config_dict)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = AntichessTrainer(model_config, training_config, mcts_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.agent.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    print(f"Starting training for {args.iterations} iterations...")
    trainer.train(args.iterations)

if __name__ == "__main__":
    main()

# File: configs/default.yaml
# Copilot Task: Create default configuration file
model:
  input_channels: 19
  residual_blocks: 12
  filters: 256
  policy_head_size: 4096
  value_head_size: 256
  dropout_rate: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs_per_iteration: 10
  games_per_iteration: 1000
  evaluation_games: 100

mcts:
  simulations: 800
  c_puct: 1.0
  temperature: 1.0
  temperature_threshold: 10
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

system:
  num_workers: 4
  redis_host: "localhost"
  redis_port: 6379
  mongodb_uri: "mongodb://localhost:27017/"
  wandb_project: "antichess-rl"
  checkpoint_interval: 100
```

#### Step 18: Distributed Training Setup
```python
# File: src/training/distributed.py
# Copilot Task: Implement distributed training with Ray
import ray
from ray import tune
from ray.rllib.agents import ppo
import numpy as np
from typing import Dict, Any, List
from ..models.alphazero_agent import AlphaZeroAgent
from ..environment.antichess_env import AntichessEnv
from ..config.settings import TrainingConfig

@ray.remote
class DistributedSelfPlayWorker:
    """Ray actor for distributed self-play"""
    
    def __init__(self, model_config: Dict, mcts_config: Dict):
        self.agent = AlphaZeroAgent(model_config, mcts_config)
    
    def update_weights(self, weights: Dict):
        """Update neural network weights"""
        self.agent.neural_net.load_state_dict(weights)
    
    def generate_games(self, num_games: int) -> List:
        """Generate self-play games"""
        games = []
        for _ in range(num_games):
            game_data = self.agent.self_play_game()
            games.append(game_data)
        return games

@ray.remote
class ParameterServer:
    """Parameter server for distributed training"""
    
    def __init__(self, model_config: Dict):
        self.agent = AlphaZeroAgent(model_config, {})
        self.version = 0
    
    def get_weights(self) -> Tuple[Dict, int]:
        """Get current model weights and version"""
        return self.agent.neural_net.state_dict(), self.version
    
    def update_weights(self, weights: Dict):
        """Update model weights"""
        self.agent.neural_net.load_state_dict(weights)
        self.version += 1
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        self.agent.save_checkpoint(path)

class DistributedTrainer:
    """Distributed training coordinator"""
    
    def __init__(self, model_config: Dict, training_config: TrainingConfig, num_workers: int = 4):
        ray.init(ignore_reinit_error=True)
        
        self.model_config = model_config
        self.training_config = training_config
        self.num_workers = num_workers
        
        # Initialize parameter server
        self.parameter_server = ParameterServer.remote(model_config)
        
        # Initialize workers
        self.workers = [
            DistributedSelfPlayWorker.remote(model_config, {})
            for _ in range(num_workers)
        ]
    
    def train_distributed(self, num_iterations: int):
        """Run distributed training loop"""
        pass
    
    def collect_experience(self, games_per_worker: int) -> List:
        """Collect experience from all workers"""
        pass
    
    def synchronize_workers(self):
        """Synchronize worker weights with parameter server"""
        pass

# File: scripts/distributed_train.py
# Copilot Task: Create distributed training script
import ray
from ray import tune
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training.distributed import DistributedTrainer
from src.config.settings import ModelConfig, TrainingConfig

def main():
    parser = argparse.ArgumentParser(description='Distributed Antichess RL Training')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--iterations', type=int, default=1000, help='Training iterations')
    parser.add_argument('--ray-address', type=str, help='Ray cluster address')
    
    args = parser.parse_args()
    
    # Initialize Ray cluster
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()
    
    # Create configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(
        model_config.__dict__,
        training_config,
        num_workers=args.workers
    )
    
    # Start training
    trainer.train_distributed(args.iterations)

if __name__ == "__main__":
    main()
```

#### Step 19: Docker Configuration
```dockerfile
# File: docker/Dockerfile
# Copilot Task: Create production Docker container
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8080 6379 27017

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]

# File: docker/docker-compose.yml
# Copilot Task: Create Docker Compose for full stack
version: '3.8'

services:
  antichess-trainer:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - redis
      - mongodb
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  mongodb:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password
  
  ray-head:
    image: rayproject/ray:2.5.1-py39-gpu
    command: ray start --head --port=6379 --redis-password="" --block
    ports:
      - "8265:8265"  # Ray dashboard
      - "10001:10001"
    volumes:
      - ../:/app
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1

volumes:
  redis_data:
  mongodb_data:
```

#### Step 20: Production Deployment
```python
# File: scripts/deploy.py
# Copilot Task: Create deployment automation script
import subprocess
import argparse
import yaml
from pathlib import Path
import os

class DeploymentManager:
    """Manage deployment to various environments"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
    
    def deploy_local(self, config_path: str):
        """Deploy locally with Docker Compose"""
        print("Deploying locally with Docker Compose...")
        
        # Build and start services
        subprocess.run([
            "docker-compose", "-f", "docker/docker-compose.yml", 
            "up", "--build", "-d"
        ], cwd=self.project_root)
        
        print("Local deployment complete!")
        print("Access Ray dashboard at: http://localhost:8265")
        print("MongoDB at: localhost:27017")
        print("Redis at: localhost:6379")
    
    def deploy_cloud(self, platform: str, config_path: str):
        """Deploy to cloud platform (AWS, GCP, Azure)"""
        if platform == "aws":
            self._deploy_aws(config_path)
        elif platform == "gcp":
            self._deploy_gcp(config_path)
        elif platform == "azure":
            self._deploy_azure(config_path)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _deploy_aws(self, config_path: str):
        """Deploy to AWS using EC2 and ECS"""
        pass
    
    def _deploy_gcp(self, config_path: str):
        """Deploy to Google Cloud Platform"""
        pass
    
    def _deploy_azure(self, config_path: str):
        """Deploy to Microsoft Azure"""
        pass
    
    def setup_monitoring(self):
        """Set up monitoring and alerting"""
        pass
    
    def create_backup_strategy(self):
        """Create backup strategy for models and data"""
        pass

# File: scripts/monitor.py
# Copilot Task: Create monitoring script
import psutil
import GPUtil
import time
import wandb
from typing import Dict

class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, wandb_project: str = "antichess-rl"):
        self.wandb_project = wandb_project
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous system monitoring"""
        wandb.init(project=self.wandb_project, job_type="monitoring")
        
        while True:
            metrics = self.collect_metrics()
            wandb.log(metrics)
            time.sleep(interval)
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = disk.percent
        metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_temperature'] = gpu.temperature
        except:
            pass
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Deploy Antichess RL System')
    parser.add_argument('command', choices=['local', 'cloud', 'monitor'], help='Deployment command')
    parser.add_argument('--platform', choices=['aws', 'gcp', 'azure'], help='Cloud platform')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'local':
        manager = DeploymentManager()
        manager.deploy_local(args.config)
    elif args.command == 'cloud':
        if not args.platform:
            print("Platform required for cloud deployment")
            return
        manager = DeploymentManager()
        manager.deploy_cloud(args.platform, args.config)
    elif args.command == 'monitor':
        monitor = SystemMonitor()
        monitor.start_monitoring()

if __name__ == "__main__":
    main()
```

### Phase 3: Advanced Features (Steps 21-30)

#### Step 21: Advanced MCTS Improvements
```python
# File: src/models/advanced_mcts.py
# Copilot Task: Implement advanced MCTS features
import numpy as np
from typing import Dict, List, Optional, Tuple
import chess
from .mcts import MCTS, MCTSNode
from ..game.antichess_rules import AntichessGame

class AdvancedMCTS(MCTS):
    """Enhanced MCTS with advanced features for antichess"""
    
    def __init__(self, neural_net, config):
        super().__init__(neural_net, config)
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_heuristic = {}
    
    def search_with_transposition(self, root_state: AntichessGame) -> Dict[chess.Move, float]:
        """MCTS with transposition table for position caching"""
        pass
    
    def add_domain_knowledge(self, node: MCTSNode):
        """Add antichess-specific domain knowledge to MCTS"""
        # Prioritize forced captures
        # Prefer piece sacrifices
        # Avoid king safety (opposite of regular chess)
        pass
    
    def progressive_widening(self, node: MCTSNode) -> bool:
        """Implement progressive widening for large branching factors"""
        pass
    
    def rave_update(self, node: MCTSNode, move: chess.Move, value: float):
        """Rapid Action Value Estimation (RAVE) updates"""
        pass
    
    def virtual_loss(self, path: List[MCTSNode]):
        """Apply virtual loss for parallel MCTS"""
        pass

class ParallelMCTS:
    """Parallel MCTS implementation for faster search"""
    
    def __init__(self, neural_net, config, num_threads: int = 4):
        self.neural_net = neural_net
        self.config = config
        self.num_threads = num_threads
    
    def parallel_search(self, root_state: AntichessGame) -> Dict[chess.Move, float]:
        """Run parallel MCTS simulations"""
        pass
    
    def worker_search(self, root_node: MCTSNode, simulations_per_worker: int):
        """Worker thread for parallel search"""
        pass
```

#### Step 22: Advanced Neural Network Architectures
```python
# File: src/models/advanced_networks.py
# Copilot Task: Implement advanced neural network architectures
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class AttentionBlock(nn.Module):
    """Multi-head attention for chess positions"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TransformerChessNet(nn.Module):
    """Transformer-based architecture for chess"""
    
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.filters
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(64, self.embed_dim))
        
        # Input projection
        self.input_proj = nn.Linear(config.input_channels, self.embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(self.embed_dim) for _ in range(config.residual_blocks)
        ])
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.embed_dim, config.policy_head_size),
            nn.ReLU(),
            nn.Linear(config.policy_head_size, 4096)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, config.value_head_size),
            nn.ReLU(),
            nn.Linear(config.value_head_size, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, channels, 8, 8)
        batch_size = x.size(0)
        
        # Reshape to sequence format
        x = x.view(batch_size, x.size(1), -1).transpose(1, 2)  # (batch, 64, channels)
        
        # Project and add position embedding
        x = self.input_proj(x) + self.pos_embedding.unsqueeze(0)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x_pooled = x.mean(dim=1)  # (batch, embed_dim)
        
        # Output heads
        policy = self.policy_head(x_pooled)
        value = self.value_head(x_pooled)
        
        return policy, value

class EfficientNet(nn.Module):
    """EfficientNet-style architecture optimized for chess"""
    
    def __init__(self, config):
        super().__init