import chess
from typing import List, Optional, Tuple
from enum import Enum
import copy

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
        # Get all legal moves from standard chess
        all_moves = list(self.board.legal_moves)
        
        # Filter for capture moves (including en passant)
        capture_moves = []
        for move in all_moves:
            # Check if it's a capture or en passant
            if self.board.is_capture(move) or self.board.is_en_passant(move):
                capture_moves.append(move)
        
        # In antichess, captures are forced - return only captures if any exist
        if capture_moves:
            return capture_moves
        else:
            return all_moves
    
    def make_move(self, move: chess.Move) -> bool:
        """
        Execute a move with antichess rule validation
        Returns: True if move was legal and executed
        """
        # Check if move is legal according to antichess rules
        legal_moves = self.get_legal_moves()
        if move not in legal_moves:
            return False
        
        # Store position for history
        self.position_history.append(self.board.fen())
        
        # Execute the move
        self.board.push(move)
        self.move_history.append(move)
        
        return True
    
    def is_game_over(self) -> Tuple[bool, GameResult]:
        """
        Check if game is finished according to antichess rules
        Returns: (is_over, result)
        """
        # Count pieces for each side
        white_piece_count = 0
        black_piece_count = 0
        
        for square, piece in self.board.piece_map().items():
            if piece.color == chess.WHITE:
                white_piece_count += 1
            else:
                black_piece_count += 1
        
        # Check if either side has no pieces left (wins in antichess)
        if white_piece_count == 0:
            return True, GameResult.WHITE_WINS
        if black_piece_count == 0:
            return True, GameResult.BLACK_WINS
        
        # Check for stalemate (also wins in antichess)
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            # Current player is stalemated and wins
            if self.board.turn == chess.WHITE:
                return True, GameResult.WHITE_WINS
            else:
                return True, GameResult.BLACK_WINS
        
        # Check for draw by repetition (rare in antichess)
        if len(self.position_history) >= 6:
            current_fen = self.board.fen()
            # Simple repetition check - could be enhanced
            repetition_count = self.position_history.count(current_fen)
            if repetition_count >= 2:
                return True, GameResult.DRAW
        
        # Check for 50-move rule (also rare in antichess)
        if self.board.halfmove_clock >= 100:
            return True, GameResult.DRAW
        
        return False, GameResult.ONGOING
    
    def get_result(self) -> GameResult:
        """Get current game result"""
        is_over, result = self.is_game_over()
        return result
    
    def clone(self) -> 'AntichessGame':
        """Create deep copy of game state"""
        cloned_game = AntichessGame()
        cloned_game.board = self.board.copy()
        cloned_game.move_history = self.move_history.copy()
        cloned_game.position_history = self.position_history.copy()
        return cloned_game
    
    def get_fen(self) -> str:
        """Get current position as FEN string (useful for debugging and persistence)"""
        return self.board.fen()
    
    def undo_move(self) -> bool:
        """
        Undo the last move
        Returns: True if move was undone, False if no moves to undo
        """
        if not self.move_history:
            return False
        
        self.board.pop()
        self.move_history.pop()
        if self.position_history:
            self.position_history.pop()
        
        return True
    
    def make_move_fast(self, move: chess.Move) -> bool:
        """
        Fast move execution for MCTS simulations
        WARNING: This assumes the move is already validated as legal!
        Only use after calling get_legal_moves() to ensure move validity.
        
        Usage pattern:
            legal_moves = game.get_legal_moves()
            for move in legal_moves:  # We know these are legal
                game.make_move_fast(move)  # Safe to use
                # ... simulation ...
                game.undo_move()
        
        Returns: True if move was executed
        """
        # Store position for history (needed for repetition detection)
        self.position_history.append(self.board.fen())
        
        # Execute the move (will raise exception if move is illegal)
        self.board.push(move)
        self.move_history.append(move)
        
        return True
    
    def copy_for_thread(self) -> 'AntichessGame':
        """
        Create thread-safe deep copy optimized for MCTS workers
        This is the preferred method for creating per-thread game instances
        """
        cloned_game = AntichessGame()
        cloned_game.board = self.board.copy(stack=True)  # Ensure full stack copy
        cloned_game.move_history = self.move_history.copy()
        cloned_game.position_history = self.position_history.copy()
        return cloned_game
    
    def get_move_count(self) -> int:
        """Get number of moves played so far"""
        return len(self.move_history)
    
    def verify_state_consistency(self) -> bool:
        """
        Verify internal state consistency (useful for debugging threading issues)
        Returns: True if state is consistent
        """
        try:
            # Check that move history length matches position history
            if len(self.move_history) != len(self.position_history):
                return False
            
            # Check that board state is valid
            if self.board.is_valid():
                return True
            
            return False
        except:
            return False
    
    def __str__(self) -> str:
        """String representation of the board"""
        return str(self.board)
