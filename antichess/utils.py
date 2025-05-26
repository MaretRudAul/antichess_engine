import numpy as np
from typing import Tuple, List
from .board import AntichessBoard, Move, Player, PieceType, Piece

def encode_board(board: AntichessBoard) -> np.ndarray:
    """
    Convert the board state into a tensor representation suitable for neural networks.
    
    The representation uses 12 channels:
    - 6 channels for White's pieces (pawn, knight, bishop, rook, queen, king)
    - 6 channels for Black's pieces (pawn, knight, bishop, rook, queen, king)
    
    Each channel is an 8x8 binary matrix where 1 indicates the presence of that piece.
    
    Args:
        board: The current board state
        
    Returns:
        A tensor of shape (12, 8, 8) representing the board state
    """
    # Initialize observation tensor: 12 binary feature planes
    # 6 for white pieces, 6 for black pieces
    observation = np.zeros((12, 8, 8), dtype=np.float32)
    
    for row in range(8):
        for col in range(8):
            piece = board.board[row, col]
            
            if piece.player == Player.NONE:
                continue
                
            piece_type_idx = piece.piece_type.value - 1  # -1 because EMPTY is 0
            
            # Calculate channel index: whites in channels 0-5, blacks in 6-11
            channel_idx = piece_type_idx if piece.player == Player.WHITE else piece_type_idx + 6
            
            # Mark the piece's position
            observation[channel_idx, row, col] = 1.0
    
    # Add a channel for whose turn it is (all 1s if white, all 0s if black)
    turn_plane = np.full((1, 8, 8), board.current_player == Player.WHITE, dtype=np.float32)
    
    # Combine planes
    return np.vstack([observation, turn_plane])

def decode_action_index(action_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert a flat action index to a (from_square, to_square) tuple.
    
    Args:
        action_idx: An integer between 0 and 4095 (64*64-1)
        
    Returns:
        A tuple ((from_row, from_col), (to_row, to_col)) representing the move
    """
    # In the 64x64 action space, we interpret action_idx as:
    # from_square * 64 + to_square
    from_square_idx = action_idx // 64
    to_square_idx = action_idx % 64
    
    from_row, from_col = from_square_idx // 8, from_square_idx % 8
    to_row, to_col = to_square_idx // 8, to_square_idx % 8
    
    return ((from_row, from_col), (to_row, to_col))

def encode_action(move: Move) -> int:
    """
    Convert a Move object to a flat action index.
    
    Args:
        move: The Move object
        
    Returns:
        An integer between 0 and 4095 representing the action
    """
    from_row, from_col = move.from_square
    to_row, to_col = move.to_square
    
    from_idx = from_row * 8 + from_col
    to_idx = to_row * 8 + to_col
    
    return from_idx * 64 + to_idx

def legal_moves_mask(board: AntichessBoard) -> np.ndarray:
    """
    Create a mask of legal moves for the current state.
    
    Args:
        board: The current board state
        
    Returns:
        A binary array of shape (4096,) where 1s indicate legal moves
    """
    legal_moves = board.get_legal_moves()
    mask = np.zeros(4096, dtype=np.float32)
    
    for move in legal_moves:
        # We currently ignore promotion in the mask
        action_idx = encode_action(move)
        mask[action_idx] = 1.0
        
    return mask

def action_to_move(board: AntichessBoard, action_idx: int) -> Move:
    """
    Convert an action index to a valid Move object.
    
    Args:
        board: The current board state
        action_idx: An integer between 0 and 4095
        
    Returns:
        A Move object, including promotion information if relevant
    """
    (from_row, from_col), (to_row, to_col) = decode_action_index(action_idx)
    
    # Check if this is a pawn promotion move
    piece = board.board[from_row, from_col]
    promotion = None
    
    if piece.piece_type == PieceType.PAWN:
        # Check if moving to the last rank
        if (piece.player == Player.WHITE and to_row == 7) or \
           (piece.player == Player.BLACK and to_row == 0):
            # In Antichess, we default to Queen promotion 
            # (in a real game, the player would choose)
            promotion = PieceType.QUEEN
    
    return Move((from_row, from_col), (to_row, to_col), promotion)