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
        0 for all other transitions
        
    Episode Termination:
        When the game is over (a player has no pieces or no legal moves)
    """
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, opponent="random"):
        """
        Initialize the Antichess environment.
        
        Args:
            opponent: Strategy for opponent moves. Options: "random", "heuristic"
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
        # print(f"Reset observation shape: {observation['observation'].shape}")
        # print(f"Non-zero elements: {np.count_nonzero(observation['observation'])}/{observation['observation'].size}")
        return observation, {}  # Return observation and empty info dict

    def calculate_reward(self, winner):
        if self.done:
            if winner == self.player_color:
                return 1.0  # Win
            else:
                return -1.0  # Loss
        
        # Smaller rewards during gameplay
        pieces_captured = self.previous_piece_count - len(self.board.piece_map())
        if pieces_captured > 0:
            return 0.01 * pieces_captured  # Small reward for forcing captures
        
        return 0  # No special event
    
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
        
        # Debug information using the action mask
        is_legal = action_mask[action] == 1.0
        # print(f"Agent action: {action}, Move: {move}, Legal: {is_legal}")
        legal_moves_count = int(np.sum(action_mask))
        # print(f"Available legal moves: {legal_moves_count}")
        
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
    
    def _make_random_opponent_move(self) -> None:
        """Make a random legal move for the opponent."""
        legal_moves = self.board.get_legal_moves()
        if legal_moves:
            move = np.random.choice(legal_moves)
            self.board.make_move(move)
    
    def _make_heuristic_opponent_move(self) -> None:
        """
        Make a heuristic-based move for the opponent.
        Prefers captures and moves that minimize material.
        """
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return
            
        # Simple heuristic: prefer capture moves
        capturing_moves = [move for move in legal_moves if 
                          self.board.board[move.to_square].player != Player.NONE]
        
        if capturing_moves:
            # Among capturing moves, prioritize capturing the highest value piece
            piece_values = {
                PieceType.PAWN: 1,
                PieceType.KNIGHT: 3,
                PieceType.BISHOP: 3,
                PieceType.ROOK: 5,
                PieceType.QUEEN: 9,
                PieceType.KING: 0
            }
            
            best_move = max(capturing_moves, key=lambda move: 
                        piece_values.get(self.board.board[move.to_square].piece_type, 0))
            self.board.make_move(best_move)
        else:
            # If no capturing moves, just pick randomly
            self._make_random_opponent_move()

    def get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask of legal actions.
        
        Returns:
            A binary array of shape (4096,) where 1s indicate legal actions
        """
        return legal_moves_mask(self.board)