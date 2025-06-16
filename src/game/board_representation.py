"""
Neural network input representation for antichess
Converts chess board positions to 8x8x19 tensors for deep learning
"""

import numpy as np
import chess
from typing import List, Union


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
        Encode board position as 19x8x8 tensor (channels-first for PyTorch)
        Planes 0-5: White pieces (P,R,N,B,Q,K)
        Planes 6-11: Black pieces (P,R,N,B,Q,K)
        Plane 12: Turn (1 for white, 0 for black)
        Plane 13: Castling rights (white kingside)
        Plane 14: Castling rights (white queenside)
        Plane 15: Castling rights (black kingside)
        Plane 16: Castling rights (black queenside)
        Plane 17: En passant target square
        Plane 18: Move count (normalized)
        
        Returns: np.ndarray of shape (19, 8, 8)
        """
        # Initialize tensor
        tensor = np.zeros((19, 8, 8), dtype=np.float32)
        
        # Encode piece positions (planes 0-11)
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            
            # Get piece type plane offset
            piece_plane = self.piece_to_plane[piece.piece_type]
            
            # Add color offset (white: 0-5, black: 6-11)
            if piece.color == chess.WHITE:
                plane_idx = piece_plane
            else:
                plane_idx = piece_plane + 6
            
            tensor[plane_idx, row, col] = 1.0
        
        # Plane 12: Current turn (1 for white, 0 for black)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
        
        # Planes 13-16: Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0
        
        # Plane 17: En passant target square
        if board.ep_square is not None:
            ep_row, ep_col = divmod(board.ep_square, 8)
            tensor[17, ep_row, ep_col] = 1.0
        
        # Plane 18: Move count (normalized)
        # Normalize fullmove number to [0, 1] range (assuming max 200 moves)
        normalized_moves = min(board.fullmove_number / 200.0, 1.0)
        tensor[18, :, :] = normalized_moves
        
        return tensor
    
    def encode_moves(self, moves: List[chess.Move]) -> np.ndarray:
        """
        Encode legal moves as action mask
        Returns: np.ndarray of shape (4096,) with 1s for legal moves
        """
        action_mask = np.zeros(4096, dtype=np.float32)
        
        for move in moves:
            action_idx = self._move_to_action(move)
            if action_idx < 4096:  # Valid action index
                action_mask[action_idx] = 1.0
        
        return action_mask
    
    def decode_move(self, move_index: int, board: chess.Board) -> chess.Move:
        """
        Convert neural network output to chess move
        Args:
            move_index: Action index from neural network (0-4095)
            board: Current board state
        Returns: chess.Move object
        """
        return self._action_to_move(move_index)
    
    def _move_to_action(self, move: chess.Move) -> int:
        """
        Convert chess move to action index
        Uses from-to encoding: from_square * 64 + to_square
        """
        return move.from_square * 64 + move.to_square
    
    def _action_to_move(self, action: int) -> chess.Move:
        """
        Convert action index to chess move
        Args:
            action: Action index (0-4095)
        Returns: chess.Move object
        """
        from_square = action // 64
        to_square = action % 64
        
        # Basic move (promotion will be handled separately if needed)
        return chess.Move(from_square, to_square)
    
    def get_action_size(self) -> int:
        """Get size of action space"""
        return 4096
    
    def get_observation_shape(self) -> tuple:
        """Get shape of observation space"""
        return (8, 8, 19)
