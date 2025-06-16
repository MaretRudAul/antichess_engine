"""
Gymnasium-compatible environment for Antichess (Losing Chess)

This module implements a professional-grade RL environment following the Gymnasium
interface for training reinforcement learning agents on antichess. The environment
handles game state management, action spaces, observations, and reward calculation
optimized for the inverted objectives of antichess.

Features:
- Standard Gymnasium interface compatibility
- AlphaZero-style strict action masking (no invalid move penalties)
- Configurable opponents (human, random, AI)
- Professional reward shaping for antichess objectives
- Comprehensive game state tracking and logging

AlphaZero Design Philosophy:
This environment follows the AlphaZero approach where invalid moves should never
be attempted by the agent. Instead of penalizing invalid moves, the environment
expects the agent/training algorithm to use action masking to filter out illegal
moves before selection. This leads to more efficient learning and cleaner code.

Key differences from "forgiving" environments:
- No penalties for invalid moves (should never happen)
- Action masking is mandatory, not optional
- Runtime errors if invalid moves are attempted (indicates bugs)
- Cleaner reward signal focused on game objectives only
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union
import logging
from enum import Enum

from ..game.antichess_rules import AntichessGame, GameResult
from ..game.board_representation import BoardEncoder
from ..config.settings import SystemConfig

logger = logging.getLogger(__name__)


class OpponentType(Enum):
    """Types of opponents for training/evaluation"""
    RANDOM = "random"
    HUMAN = "human"
    AI = "ai"
    SELF_PLAY = "self_play"


class AntichessEnv(gym.Env):
    """
    Antichess environment following Gymnasium interface.
    
    This environment implements the inverted chess game where the objective
    is to lose all pieces or reach stalemate. It provides a standard
    reinforcement learning interface for training agents.
    
    Observation Space:
        Box(0, 1, (8, 8, 19), float32) - Board state encoded as tensor
        
    Action Space:
        Discrete(4096) - All possible moves in from-to encoding (64x64)
        
    Rewards:
        +1.0: Win (lost all pieces or reached stalemate)
        -1.0: Loss (opponent won)
         0.0: Game continues
        +0.1: Successful forced capture (progress toward goal)
        
    Note: This environment follows the AlphaZero approach with strict action masking.
    Invalid moves should never be attempted - the agent must use action masks to
    filter legal moves before selection. No penalties are given for invalid moves
    since they should be impossible with proper masking.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'ascii'],
        'render_fps': 1
    }
    
    def __init__(
        self,
        opponent: Optional[Union[str, OpponentType]] = None,
        max_moves: int = 500,
        reward_shaping: bool = True,
        render_mode: Optional[str] = None
    ):
        """
        Initialize Antichess environment.
        
        Args:
            opponent: Type of opponent ('random', 'human', 'ai', 'self_play', or None)
            max_moves: Maximum moves before draw (prevents infinite games)
            reward_shaping: Whether to use intermediate rewards for learning
            render_mode: How to render the environment
        """
        super().__init__()
        
        # Action space: all possible moves (4096 for 64x64 from-to encoding)
        self.action_space = spaces.Discrete(4096)
        
        # Observation space: 8x8x19 board representation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 19), dtype=np.float32
        )
        
        # Initialize game components
        self.game = AntichessGame()
        self.encoder = BoardEncoder()
        
        # Environment configuration
        self.opponent = OpponentType(opponent) if opponent else None
        self.max_moves = max_moves
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        
        # Game state tracking
        self.move_count = 0
        self.episode_rewards = []
        self.last_action = None
        self.game_history = []
        
        # Performance metrics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        logger.info(f"Initialized AntichessEnv with opponent={opponent}, max_moves={max_moves}")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game state
        self.game = AntichessGame()
        self.move_count = 0
        self.episode_rewards = []
        self.last_action = None
        self.game_history = []
        
        # Get initial observation
        observation = self.encoder.encode_position(self.game.board)
        
        # Prepare info dictionary
        info = {
            'legal_moves': len(self.game.get_legal_moves()),
            'game_over': False,
            'result': None,
            'move_count': self.move_count,
            'fen': self.game.board.fen()
        }
        
        logger.debug(f"Environment reset - Legal moves: {info['legal_moves']}")
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return environment response.
        
        Args:
            action: Integer action from action space (0-4095)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action is in action space
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} not in action space")
        
        # Store action for analysis
        self.last_action = action
        reward = 0.0
        terminated = False
        truncated = False
         # AlphaZero approach: Only legal moves should be possible through action masking
        # If we reach here with an invalid action, it's a programming error, not agent error
        legal_moves = self.game.get_legal_moves()
        
        # Convert action to move
        try:
            move = self.encoder.decode_action(action)
        except ValueError as e:
            # This should never happen with proper action masking
            raise RuntimeError(f"Invalid action {action} passed to step() - action masking failed: {e}")

        # Verify move is legal (should always be true with proper masking)
        if move not in legal_moves:
            # This should never happen with proper action masking
            raise RuntimeError(f"Illegal move {move} passed to step() - action masking failed. "
                             f"Legal moves: {[str(m) for m in legal_moves[:5]]}...")
        
        # Execute move
        try:
            self.game.make_move(move)
            self.move_count += 1
            self.game_history.append(move)
            
            # Check for game termination
            if self.game.is_game_over():
                result = self.game.get_result()
                terminated = True
                reward = self._calculate_reward(result)
                self._update_statistics(result)
            
            # Check for truncation (max moves reached)
            elif self.move_count >= self.max_moves:
                truncated = True
                reward = 0.0  # Draw
                self.draws += 1
                logger.info(f"Game truncated after {self.max_moves} moves")
            
            # Apply reward shaping if enabled
            elif self.reward_shaping:
                reward += self._calculate_intermediate_reward(move, legal_moves)
            
            # Handle opponent move (if not self-play or game over)
            if not terminated and not truncated and self.opponent and self.opponent != OpponentType.SELF_PLAY:
                opponent_reward = self._handle_opponent_move()
                if opponent_reward != 0:  # Opponent caused game to end
                    terminated = True
                    reward = -opponent_reward  # Flip reward for opponent outcome
        
        except Exception as e:
            logger.error(f"Error executing move {move}: {e}")
            reward = -1.0
            terminated = True
        
        # Get new observation
        observation = self.encoder.encode_position(self.game.board)
        
        # Update episode rewards
        self.episode_rewards.append(reward)
        
        # Create info dictionary
        info = self._create_info_dict(terminated, truncated, reward)
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: Optional[str] = None):
        """
        Render current game state.
        
        Args:
            mode: Render mode ('human', 'rgb_array', 'ascii')
        """
        render_mode = mode or self.render_mode or 'human'
        
        if render_mode == 'human':
            print("\nCurrent Board:")
            print(self.game.board)
            print(f"Move count: {self.move_count}")
            print(f"Legal moves: {len(self.game.get_legal_moves())}")
            if self.game.is_game_over():
                result = self.game.get_result()
                print(f"Game Over - Result: {result}")
        
        elif render_mode == 'ascii':
            return str(self.game.board)
        
        elif render_mode == 'rgb_array':
            # TODO: Implement visual board rendering
            logger.warning("RGB array rendering not yet implemented")
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of legal actions.
        
        Returns:
            Boolean array of shape (4096,) indicating legal actions
        """
        return self.encoder.get_action_mask(self.game.get_legal_moves())
    
    def validate_action_masking(self, action: int) -> bool:
        """
        Validate that an action is properly masked (for debugging).
        
        This method can be used during development to verify that the action
        masking is working correctly before passing actions to step().
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        try:
            # Check if action is in action space
            if not self.action_space.contains(action):
                return False
            
            # Check if action mask allows this action
            action_mask = self.get_action_mask()
            if not action_mask[action]:
                return False
            
            # Check if action decodes to a legal move
            move = self.encoder.decode_action(action)
            legal_moves = self.game.get_legal_moves()
            
            return move in legal_moves
            
        except Exception:
            return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get comprehensive debug information about current state.
        
        Returns:
            Dictionary with debug information
        """
        legal_moves = self.game.get_legal_moves()
        action_mask = self.get_action_mask()
        
        return {
            'legal_moves_count': len(legal_moves),
            'legal_moves': [str(move) for move in legal_moves],
            'action_mask_true_count': np.sum(action_mask),
            'action_mask_shape': action_mask.shape,
            'current_player': 'white' if self.game.board.turn else 'black',
            'piece_count': len(self.game.board.piece_map()),
            'move_count': self.move_count,
            'game_over': self.game.is_game_over(),
            'fen': self.game.board.fen()
        }

    def _calculate_reward(self, result: GameResult) -> float:
        """
        Calculate reward based on antichess objectives.
        
        In antichess, the goal is to lose all pieces or reach stalemate.
        
        Args:
            result: Game result from antichess rules
            
        Returns:
            Reward value for the current player
        """
        if result == GameResult.WHITE_WINS:
            # White won (lost all pieces or stalemate)
            return 1.0 if self.game.board.turn else -1.0
        elif result == GameResult.BLACK_WINS:
            # Black won (lost all pieces or stalemate)
            return -1.0 if self.game.board.turn else 1.0
        else:
            # Draw or ongoing game
            return 0.0
    
    def _calculate_intermediate_reward(self, move, legal_moves) -> float:
        """
        Calculate intermediate rewards for learning acceleration.
        
        Args:
            move: The move that was made
            legal_moves: List of legal moves before the move
            
        Returns:
            Small intermediate reward
        """
        reward = 0.0
        
        # Reward for making forced captures (key antichess mechanic)
        if self.game.board.is_capture(move):
            reward += 0.1
        
        # Small penalty for having many pieces (goal is to lose them)
        piece_count = len(self.game.board.piece_map())
        if piece_count > 16:  # More than half pieces remaining
            reward -= 0.01
        
        return reward
    
    def _handle_opponent_move(self) -> float:
        """
        Handle opponent's move based on opponent type.
        
        Returns:
            Reward if game ended, 0.0 otherwise
        """
        if self.opponent == OpponentType.RANDOM:
            return self._make_random_opponent_move()
        elif self.opponent == OpponentType.HUMAN:
            # Human input would be handled externally
            return 0.0
        elif self.opponent == OpponentType.AI:
            # AI opponent would be implemented separately
            logger.warning("AI opponent not yet implemented")
            return 0.0
        
        return 0.0
    
    def _make_random_opponent_move(self) -> float:
        """
        Make a random move for the opponent.
        
        Returns:
            Reward if game ended after opponent move, 0.0 otherwise
        """
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return 0.0
        
        # Choose random legal move
        opponent_move = np.random.choice(legal_moves)
        self.game.make_move(opponent_move)
        self.move_count += 1
        self.game_history.append(opponent_move)
        
        # Check if opponent move ended the game
        if self.game.is_game_over():
            result = self.game.get_result()
            self._update_statistics(result)
            return self._calculate_reward(result)
        
        return 0.0
    
    def _create_info_dict(
        self, 
        terminated: bool, 
        truncated: bool, 
        reward: float
    ) -> Dict[str, Any]:
        """Create comprehensive info dictionary for step return."""
        legal_moves = self.game.get_legal_moves()
        
        info = {
            'legal_moves': len(legal_moves),
            'game_over': terminated or truncated,
            'result': self.game.get_result() if self.game.is_game_over() else None,
            'move_count': self.move_count,
            'fen': self.game.board.fen(),
            'last_action': self.last_action,
            'reward': reward,
            'episode_rewards': self.episode_rewards.copy(),
            'piece_count': len(self.game.board.piece_map()),
            'action_mask': self.get_action_mask() if legal_moves else None
        }
        
        # Add game statistics if terminated
        if terminated:
            info.update({
                'games_played': self.games_played,
                'win_rate': self.wins / max(1, self.games_played),
                'total_reward': sum(self.episode_rewards)
            })
        
        return info
    
    def _update_statistics(self, result: GameResult) -> None:
        """Update win/loss statistics."""
        self.games_played += 1
        
        if result == GameResult.WHITE_WINS:
            if self.game.board.turn:  # It's black's turn, so white just won
                self.losses += 1
            else:
                self.wins += 1
        elif result == GameResult.BLACK_WINS:
            if self.game.board.turn:  # It's black's turn, so they just won
                self.wins += 1
            else:
                self.losses += 1
        else:
            self.draws += 1
        
        logger.info(f"Game ended - W:{self.wins} L:{self.losses} D:{self.draws} "
                   f"Rate:{self.wins/self.games_played:.3f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics for monitoring."""
        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / max(1, self.games_played),
            'avg_game_length': self.move_count / max(1, self.games_played)
        }
    
    def close(self):
        """Clean up environment resources."""
        logger.info(f"Closing environment - Final stats: {self.get_statistics()}")
        super().close()


class AntichessVecEnv:
    """
    Vectorized environment wrapper for parallel training.
    
    This enables running multiple antichess environments in parallel
    for more efficient data collection during training.
    """
    
    def __init__(self, num_envs: int = 4, **env_kwargs):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            **env_kwargs: Arguments passed to each environment
        """
        self.num_envs = num_envs
        self.envs = [AntichessEnv(**env_kwargs) for _ in range(num_envs)]
        
        # Copy space definitions from single environment
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        logger.info(f"Initialized vectorized environment with {num_envs} parallel envs")
    
    def reset(self, seed=None):
        """Reset all environments."""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos
    
    def step(self, actions):
        """Step all environments with given actions."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def get_action_masks(self):
        """Get action masks for all environments."""
        return np.array([env.get_action_mask() for env in self.envs])
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
