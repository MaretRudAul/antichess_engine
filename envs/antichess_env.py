import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from antichess.board import AntichessBoard, Move, Player
from antichess.utils import encode_board, action_to_move, legal_moves_mask

class AntichessEnv(gym.Env):
    """
    A Gym environment for the game of Antichess.
    
    Observation:
        A 13x8x8 tensor representing the board state.
    
    Actions:
        Integer from 0 to 4095 representing a move (from, to) as a board index.
        
    Reward:
        +1 for winning the game
        -1 for losing the game
        [intermediate rewards for losing pieces]
        0 for all other transitions
        
    Episode Termination:
        When the game is over (a player has no pieces or no legal moves)
    """
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, opponent="random"):
        """
        Initialize the Antichess environment.
        
        Args:
            opponent: Strategy for opponent moves. Options: 
                     "random", "heuristic", "self_play"
        """
        super().__init__()
        
        # There are 13 planes: 6 for white pieces, 6 for black pieces, 1 for turn
        self.observation_space = gym.spaces.Dict({
            "observation": spaces.Box(
                low=0, high=1, shape=(13, 8, 8), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(4096,), dtype=np.float32
            )
        })
        
        # The action space is 64*64=4096 possible moves
        # Each index represents a source and target square
        self.action_space = spaces.Discrete(4096)
        
        self.board = None
        self.done = False
        self.opponent = opponent
        self.player_color = Player.WHITE  # The agent always plays as WHITE
        
        # Self-play specific attributes
        self.opponent_model = None
        self.self_play_probability = 0.8  # Probability of using model vs random

    def set_opponent_model(self, model):
        """
        Set the model to use for self-play opponent.
        
        Args:
            model: The trained model to use as opponent
        """
        self.opponent_model = model

    def set_self_play_probability(self, probability: float):
        """
        Set the probability of using the model vs random play in self-play mode.
        
        Args:
            probability: Float between 0 and 1
        """
        self.self_play_probability = probability

    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generator.
        
        Args:
            seed: The seed value
        """
        np.random.seed(seed)
        return [seed]
        
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, dict]:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: An optional seed for random number generation
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        self.board = AntichessBoard()
        self.done = False
        
        # If the agent is BLACK, make an opponent move first
        if self.player_color == Player.BLACK and self.board.current_player == Player.WHITE:
            self._make_opponent_move()
        
        observation = self._get_observation()
        return observation, {}

    def calculate_reward(self, winner):
        if self.done:
            if winner == self.player_color:
                return 1.0  # Win
            else:
                return -1.0  # Loss
        
        return 0  # No reward during gameplay
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by making a move.
        
        Args:
            action: An integer from 0 to 4095 representing the move
            
        Returns:
            observation: The next board state
            reward: The reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}
        
        # Get the action mask for legal moves
        action_mask = self.get_action_mask()
        
        # Convert action index to chess move
        move = action_to_move(self.board, action)
        
        # Check if move is legal using the mask
        if action_mask[action] == 1.0:
            self.board.make_move(move)
            
            # Check if game is over after the agent's move
            if self.board.is_game_over():
                self.done = True
                winner = self.board.get_winner()
                reward = 1.0 if winner == self.player_color else -1.0
                return self._get_observation(), reward, True, False, {"winner": winner}
            
            # Make opponent's move if the game is not over
            self._make_opponent_move()
            
            # Check if game is over after the opponent's move
            if self.board.is_game_over():
                self.done = True
                winner = self.board.get_winner()
                reward = self.calculate_reward(winner)
                return self._get_observation(), reward, True, False, {"winner": winner}
                
            # Game continues
            return self._get_observation(), 0.0, False, False, {}
        else:
            # Illegal move - in a real game this would forfeit, but for RL we'll just give a negative reward
            self.done = True
            return self._get_observation(), -1.0, True, False, {"illegal_move": True}
    
    def render(self, mode="human"):
        """
        Render the current board state.
        
        Args:
            mode: The rendering mode
            
        Returns:
            The rendering result
        """
        if mode == "human":
            print(self.board)
            return None
        elif mode == "rgb_array":
            # This would require implementing a visual rendering
            # For simplicity, we'll return a dummy array
            return np.zeros((400, 400, 3))
    
    def close(self):
        """Close the environment."""
        pass
    
    def _get_observation(self):
        """
        Get the observation with action mask.
        
        Returns:
            Dictionary with observation and action mask
        """
        return {
            "observation": encode_board(self.board),
            "action_mask": self.get_action_mask()
        }
    
    def _make_opponent_move(self) -> None:
        """Make a move for the opponent based on the selected strategy."""
        if self.opponent == "random":
            self._make_random_opponent_move()
        elif self.opponent == "heuristic":
            self._make_heuristic_opponent_move()
        elif self.opponent == "self_play":
            self._make_self_play_opponent_move()
        else:
            # Default to random if unknown opponent type
            self._make_random_opponent_move()
    
    def _make_random_opponent_move(self) -> None:
        """Make a random legal move for the opponent."""
        legal_moves = list(self.board.get_legal_moves())
        if legal_moves:
            move = np.random.choice(legal_moves)
            self.board.make_move(move)
    
    def _make_heuristic_opponent_move(self) -> None:
        """
        Make a heuristic-based move for the opponent.
        Simple strategy for Antichess.
        """
        legal_moves = list(self.board.get_legal_moves())
        if not legal_moves:
            return
        
        # In Antichess, captures are already mandatory, so legal_moves already
        # contains only captures if any exist, otherwise all legal moves
        
        # Simple heuristic: if multiple moves available, prefer moves that
        # capture higher-value pieces or moves to central squares
        if len(legal_moves) == 1:
            self.board.make_move(legal_moves[0])
            return
        
        # Score moves based on simple criteria
        move_scores = []
        for move in legal_moves:
            score = 0
            target_piece = self.board.board[move.to_square]
            
            # Prefer capturing higher-value pieces
            if target_piece.player != Player.NONE:
                piece_values = {
                    PieceType.PAWN: 1,
                    PieceType.KNIGHT: 3,
                    PieceType.BISHOP: 3,
                    PieceType.ROOK: 5,
                    PieceType.QUEEN: 9,
                    PieceType.KING: 10
                }
                score += piece_values.get(target_piece.piece_type, 0)
            
            # Slightly prefer central squares
            to_row, to_col = move.to_square
            center_distance = abs(3.5 - to_row) + abs(3.5 - to_col)
            score += (7 - center_distance) * 0.1
            
            move_scores.append(score)
        
        # Choose the move with highest score (with some randomness)
        if max(move_scores) > min(move_scores):
            # Weight moves by their scores
            weights = np.array(move_scores)
            weights = weights - min(weights) + 1  # Ensure all positive
            probabilities = weights / sum(weights)
            chosen_move = np.random.choice(legal_moves, p=probabilities)
        else:
            # If all moves have equal score, choose randomly
            chosen_move = np.random.choice(legal_moves)
        
        self.board.make_move(chosen_move)
    
    def _make_self_play_opponent_move(self) -> None:
        """
        Make a move using the self-play model or fall back to random.
        """
        # Use model with specified probability
        use_model = (self.opponent_model is not None and 
                    np.random.random() < self.self_play_probability)
        
        if use_model:
            try:
                self._make_model_opponent_move()
            except Exception as e:
                # If model fails, fall back to random
                print(f"Self-play model failed: {e}. Falling back to random.")
                self._make_random_opponent_move()
        else:
            self._make_random_opponent_move()
    
    def _make_model_opponent_move(self) -> None:
        """Make a move using the opponent model."""
        # Get current observation from opponent's perspective
        obs = self._get_observation()
        
        # For simplicity, we'll use the same observation
        # In a more sophisticated implementation, you might flip the board perspective
        
        # Predict action using the model
        action, _ = self.opponent_model.predict(obs, deterministic=False)
        
        # Convert action to move
        move = action_to_move(self.board, action)
        
        # Verify the move is legal (safety check)
        legal_moves = list(self.board.get_legal_moves())
        if move in legal_moves:
            self.board.make_move(move)
        else:
            # If model suggests illegal move, fall back to random
            self._make_random_opponent_move()

    def get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask of legal actions.
        
        Returns:
            A binary array of shape (4096,) where 1s indicate legal actions
        """
        return legal_moves_mask(self.board)