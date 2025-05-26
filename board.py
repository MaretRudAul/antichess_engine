# File: antichess/board.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum, auto

class PieceType(Enum):
    """Enum representing piece types."""
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

class Player(Enum):
    """Enum representing players."""
    NONE = 0
    WHITE = 1
    BLACK = 2

class Piece:
    """Class representing a chess piece."""
    
    def __init__(self, piece_type: PieceType, player: Player):
        """
        Initialize a piece.
        
        Args:
            piece_type: The type of the piece
            player: The player owning the piece
        """
        self.piece_type = piece_type
        self.player = player
        
    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.piece_type == other.piece_type and self.player == other.player
    
    def __repr__(self):
        if self.piece_type == PieceType.EMPTY:
            return "."
        
        symbols = {
            PieceType.PAWN: "p",
            PieceType.KNIGHT: "n",
            PieceType.BISHOP: "b",
            PieceType.ROOK: "r",
            PieceType.QUEEN: "q",
            PieceType.KING: "k",
        }
        
        symbol = symbols[self.piece_type]
        return symbol.upper() if self.player == Player.WHITE else symbol


class Move:
    """Class representing a chess move."""
    
    def __init__(self, from_square: Tuple[int, int], to_square: Tuple[int, int], 
                 promotion: Optional[PieceType] = None):
        """
        Initialize a move.
        
        Args:
            from_square: Starting position (row, col)
            to_square: Target position (row, col)
            promotion: The piece type to promote to, if applicable
        """
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
        
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and 
                self.to_square == other.to_square and 
                self.promotion == other.promotion)
    
    def __repr__(self):
        files = "abcdefgh"
        ranks = "87654321"
        from_file, from_rank = self.from_square[1], self.from_square[0]
        to_file, to_rank = self.to_square[1], self.to_square[0]
        
        move_str = f"{files[from_file]}{ranks[from_rank]}{files[to_file]}{ranks[to_rank]}"
        
        if self.promotion:
            promotion_symbols = {
                PieceType.KNIGHT: "n",
                PieceType.BISHOP: "b",
                PieceType.ROOK: "r",
                PieceType.QUEEN: "q"
            }
            move_str += promotion_symbols[self.promotion]
            
        return move_str


class AntichessBoard:
    """
    Class representing the Antichess board state and game logic.
    
    In Antichess, captures are mandatory and the goal is to lose all pieces
    or have no legal moves.
    """
    
    def __init__(self):
        """Initialize the board with the standard chess setup."""
        self.board = np.empty((8, 8), dtype=object)
        self._setup_board()
        self.current_player = Player.WHITE
        self.move_history = []
        
    def _setup_board(self):
        """Set up the initial chess position."""
        # Initialize with empty squares
        for row in range(8):
            for col in range(8):
                self.board[row, col] = Piece(PieceType.EMPTY, Player.NONE)
        
        # Set up pawns
        for col in range(8):
            self.board[1, col] = Piece(PieceType.PAWN, Player.WHITE)
            self.board[6, col] = Piece(PieceType.PAWN, Player.BLACK)
        
        # Set up other pieces
        back_rank_pieces = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
            PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
        ]
        
        for col, piece_type in enumerate(back_rank_pieces):
            self.board[0, col] = Piece(piece_type, Player.WHITE)
            self.board[7, col] = Piece(piece_type, Player.BLACK)
    
    def get_piece(self, square: Tuple[int, int]) -> Piece:
        """
        Get the piece at a given square.
        
        Args:
            square: Position (row, col)
            
        Returns:
            The piece at the square
        """
        row, col = square
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row, col]
        return None
    
    def get_piece_positions(self, player: Player) -> List[Tuple[int, int]]:
        """
        Get positions of all pieces for a player.
        
        Args:
            player: The player whose pieces to get
            
        Returns:
            List of positions (row, col) of all pieces belonging to the player
        """
        positions = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece.player == player:
                    positions.append((row, col))
        return positions
    
    def _get_pawn_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible pawn moves from a square."""
        row, col = square
        piece = self.board[row, col]
        moves = []
        
        # Determine direction based on player
        direction = 1 if piece.player == Player.WHITE else -1
        
        # Forward move
        new_row = row + direction
        if 0 <= new_row < 8 and self.board[new_row, col].piece_type == PieceType.EMPTY:
            # Check for promotion
            if (new_row == 7 and piece.player == Player.WHITE) or (new_row == 0 and piece.player == Player.BLACK):
                for promotion in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                    moves.append(Move(square, (new_row, col), promotion))
            else:
                moves.append(Move(square, (new_row, col)))
                
                # Two-square forward move from starting position
                if ((row == 1 and piece.player == Player.WHITE) or 
                    (row == 6 and piece.player == Player.BLACK)):
                    new_row = row + 2 * direction
                    if self.board[new_row, col].piece_type == PieceType.EMPTY:
                        moves.append(Move(square, (new_row, col)))
        
        # Captures
        for col_offset in [-1, 1]:
            new_col = col + col_offset
            new_row = row + direction
            
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row, new_col]
                if target.player != Player.NONE and target.player != piece.player:
                    # Check for promotion
                    if (new_row == 7 and piece.player == Player.WHITE) or (new_row == 0 and piece.player == Player.BLACK):
                        for promotion in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                            moves.append(Move(square, (new_row, new_col), promotion))
                    else:
                        moves.append(Move(square, (new_row, new_col)))
        
        return moves
    
    def _get_knight_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible knight moves from a square."""
        row, col = square
        piece = self.board[row, col]
        moves = []
        
        # All possible knight offsets
        offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for row_offset, col_offset in offsets:
            new_row = row + row_offset
            new_col = col + col_offset
            
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row, new_col]
                # Either empty square or opponent piece
                if target.player != piece.player:
                    moves.append(Move(square, (new_row, new_col)))
                    
        return moves
    
    def _get_sliding_moves(self, square: Tuple[int, int], directions: List[Tuple[int, int]]) -> List[Move]:
        """Get all possible moves in specific directions from a square."""
        row, col = square
        piece = self.board[row, col]
        moves = []
        
        for row_dir, col_dir in directions:
            new_row, new_col = row + row_dir, col + col_dir
            
            # Continue in this direction while on the board
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row, new_col]
                
                if target.player == Player.NONE:
                    # Empty square, can move here
                    moves.append(Move(square, (new_row, new_col)))
                elif target.player != piece.player:
                    # Opponent's piece, can capture
                    moves.append(Move(square, (new_row, new_col)))
                    break
                else:
                    # Own piece, stop in this direction
                    break
                
                new_row += row_dir
                new_col += col_dir
                
        return moves
    
    def _get_bishop_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible bishop moves from a square."""
        # Diagonal directions
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return self._get_sliding_moves(square, directions)
    
    def _get_rook_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible rook moves from a square."""
        # Horizontal and vertical directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_sliding_moves(square, directions)
    
    def _get_queen_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible queen moves from a square."""
        # Combine bishop and rook directions
        directions = [
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal
            (-1, 0), (1, 0), (0, -1), (0, 1)     # Horizontal/Vertical
        ]
        return self._get_sliding_moves(square, directions)
    
    def _get_king_moves(self, square: Tuple[int, int]) -> List[Move]:
        """Get all possible king moves from a square."""
        row, col = square
        piece = self.board[row, col]
        moves = []
        
        # All adjacent squares
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for row_dir, col_dir in directions:
            new_row, new_col = row + row_dir, col + col_dir
            
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row, new_col]
                if target.player != piece.player:
                    moves.append(Move(square, (new_row, new_col)))
        
        return moves

    def get_moves(self, square: Tuple[int, int]) -> List[Move]:
        """
        Get all possible moves from a square according to piece movement rules.
        
        Args:
            square: Position (row, col)
            
        Returns:
            List of all legal moves for the piece at the square
        """
        piece = self.get_piece(square)
        if piece.player == Player.NONE or piece.player != self.current_player:
            return []
        
        if piece.piece_type == PieceType.PAWN:
            return self._get_pawn_moves(square)
        elif piece.piece_type == PieceType.KNIGHT:
            return self._get_knight_moves(square)
        elif piece.piece_type == PieceType.BISHOP:
            return self._get_bishop_moves(square)
        elif piece.piece_type == PieceType.ROOK:
            return self._get_rook_moves(square)
        elif piece.piece_type == PieceType.QUEEN:
            return self._get_queen_moves(square)
        elif piece.piece_type == PieceType.KING:
            return self._get_king_moves(square)
        
        return []
    
    def get_capturing_moves(self) -> List[Move]:
        """
        Get all moves that capture a piece for the current player.
        In Antichess, captures are mandatory.
        
        Returns:
            List of all capturing moves
        """
        capturing_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece.player == self.current_player:
                    for move in self.get_moves((row, col)):
                        target = self.board[move.to_square]
                        if target.player != Player.NONE and target.player != self.current_player:
                            capturing_moves.append(move)
        
        return capturing_moves
    
    def get_legal_moves(self) -> List[Move]:
        """
        Get all legal moves for the current player according to Antichess rules.
        If captures are available, they are mandatory.
        
        Returns:
            List of all legal moves
        """
        # Check for captures first (mandatory in Antichess)
        capturing_moves = self.get_capturing_moves()
        if capturing_moves:
            return capturing_moves
            
        # If no captures, get all possible moves
        all_moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece.player == self.current_player:
                    all_moves.extend(self.get_moves((row, col)))
        
        return all_moves
    
    def make_move(self, move: Move) -> None:
        """
        Execute a move on the board.
        
        Args:
            move: The move to execute
        """
        from_row, from_col = move.from_square
        to_row, to_col = move.to_square
        
        piece = self.board[from_row, from_col]
        
        # Handle promotion
        if move.promotion and piece.piece_type == PieceType.PAWN:
            piece = Piece(move.promotion, piece.player)
        
        # Update board
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = Piece(PieceType.EMPTY, Player.NONE)
        
        # Record the move
        self.move_history.append(move)
        
        # Switch player
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        In Antichess, a player wins when they have no pieces or no legal moves.
        
        Returns:
            True if the game is over, otherwise False
        """
        # Check if current player has no pieces
        has_pieces = False
        for row in range(8):
            for col in range(8):
                if self.board[row, col].player == self.current_player:
                    has_pieces = True
                    break
            if has_pieces:
                break
                
        if not has_pieces:
            return True
        
        # Check if current player has no legal moves
        legal_moves = self.get_legal_moves()
        return len(legal_moves) == 0
    
    def get_winner(self) -> Optional[Player]:
        """
        Determine the winner of the game, if any.
        In Antichess, the player who loses all pieces or has no legal moves wins.
        
        Returns:
            The winning player, or None if the game is still ongoing
        """
        if not self.is_game_over():
            return None
        
        # The current player has no moves or pieces, so the opponent wins
        return Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
    
    def __repr__(self) -> str:
        """String representation of the board."""
        result = ""
        for row in range(8):
            for col in range(8):
                result += str(self.board[row, col]) + " "
            result += "\n"
        return result