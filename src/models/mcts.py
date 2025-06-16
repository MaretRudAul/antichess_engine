"""
Monte Carlo Tree Search for Antichess

This module implements MCTS following the AlphaZero approach, specifically adapted
for antichess gameplay. The implementation includes UCB selection, neural network
guided expansion, and efficient tree traversal optimized for antichess characteristics.

Key features:
- UCB1 selection with neural network priors
- Dirichlet noise for exploration at root
- Virtual loss for parallel search capability
- Antichess-specific adaptations for forced captures
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import chess
import torch
import logging

from ..game.antichess_rules import AntichessGame
from ..game.board_representation import BoardEncoder
from ..config.settings import MCTSConfig

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Each node represents a game state and maintains statistics about
    visits, values, and children for the MCTS algorithm.
    """
    
    def __init__(self, state: AntichessGame, parent=None, move=None, prior=0.0):
        """
        Initialize MCTS node.
        
        Args:
            state: Game state at this node
            parent: Parent node (None for root)
            move: Move that led to this state
            prior: Neural network prior probability for this move
        """
        self.state = state.clone()  # Store independent copy of game state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False
        
        # For parallel MCTS (future enhancement)
        self.virtual_loss = 0
    
    def value(self) -> float:
        """
        Get average value of this node.
        
        Returns:
            Average value based on all visits to this node
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate Upper Confidence Bound score for this node.
        
        UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant
            parent_visits: Number of visits to parent node
            
        Returns:
            UCB score for node selection
        """
        if self.visit_count == 0:
            # Unvisited nodes have infinite UCB score
            return float('inf')
        
        # Q-value (average value)
        q_value = self.value()
        
        # UCB exploration term
        exploration = (c_puct * self.prior * math.sqrt(parent_visits) / 
                      (1 + self.visit_count))
        
        return q_value + exploration
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """
        Select child with highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Child node with highest UCB score
        """
        if not self.children:
            raise ValueError("Cannot select child from unexpanded node")
        
        # Find child with maximum UCB score
        best_child = max(
            self.children.values(),
            key=lambda child: child.ucb_score(c_puct, self.visit_count)
        )
        
        return best_child
    
    def expand(self, policy_probs: Dict[chess.Move, float], legal_moves: List[chess.Move]):
        """
        Expand node by creating children for all legal moves.
        
        Args:
            policy_probs: Neural network policy probabilities for each move
            legal_moves: List of legal moves from this position
        """
        if self.is_expanded:
            return
        
        self.is_expanded = True
        
        for move in legal_moves:
            # Get prior probability for this move
            prior = policy_probs.get(move, 0.0)
            
            # Create child state by making the move
            child_state = self.state.clone()
            child_state.make_move(move)
            
            # Create child node
            child_node = MCTSNode(
                state=child_state,
                parent=self,
                move=move,
                prior=prior
            )
            
            self.children[move] = child_node
        
        logger.debug(f"Expanded node with {len(self.children)} children")
    
    def backup(self, value: float):
        """
        Backup value through the tree to the root.
        
        Args:
            value: Value to backup (from perspective of player to move at this node)
        """
        self.visit_count += 1
        self.value_sum += value
        
        # Recursively backup to parent (flip value for opponent)
        if self.parent is not None:
            self.parent.backup(-value)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return not self.is_expanded
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Get child with most visits."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.visit_count)


class MCTS:
    """
    Monte Carlo Tree Search implementation for antichess.
    
    Follows the AlphaZero approach with neural network guided expansion
    and UCB-based selection. Adapted for antichess with proper handling
    of forced captures and win/loss conditions.
    """
    
    def __init__(self, neural_net, config: MCTSConfig):
        """
        Initialize MCTS.
        
        Args:
            neural_net: Neural network for position evaluation and move prediction
            config: MCTS configuration parameters
        """
        self.neural_net = neural_net
        self.config = config
        self.encoder = BoardEncoder()
        
        # Set neural network to evaluation mode
        self.neural_net.eval()
        
        logger.info(f"Initialized MCTS with {config.simulations} simulations")
    
    def search(self, root_state: AntichessGame) -> Dict[chess.Move, float]:
        """
        Run MCTS simulations and return move probabilities.
        
        Args:
            root_state: Starting game state
            
        Returns:
            Dictionary mapping moves to visit probabilities
        """
        # Create root node
        root = MCTSNode(root_state)
        
        # Add Dirichlet noise to root for exploration
        self._add_dirichlet_noise(root, root_state.get_legal_moves())
        
        # Run simulations
        for simulation in range(self.config.simulations):
            # Single MCTS simulation
            self._simulate(root)
            
            if simulation > 0 and simulation % 100 == 0:
                logger.debug(f"Completed {simulation}/{self.config.simulations} simulations")
        
        # Convert visit counts to move probabilities
        return self._get_move_probabilities(root)
    
    def _simulate(self, root: MCTSNode) -> float:
        """
        Perform single MCTS simulation.
        
        Args:
            root: Root node to start simulation from
            
        Returns:
            Value to backup from leaf evaluation
        """
        path = []
        current = root
        
        # Selection phase: traverse tree using UCB until leaf
        while not current.is_leaf() and not current.state.is_game_over():
            path.append(current)
            current = current.select_child(self.config.c_puct)
        
        path.append(current)
        
        # Terminal node check
        if current.state.is_game_over():
            # Game is over, get actual result
            result = current.state.get_result()
            if result.value == 1:  # WHITE_WINS
                value = 1.0 if current.state.board.turn else -1.0
            elif result.value == 2:  # BLACK_WINS
                value = -1.0 if current.state.board.turn else 1.0
            else:  # DRAW
                value = 0.0
        else:
            # Expansion and evaluation phase
            value = self._expand_and_evaluate(current)
        
        # Backup phase: propagate value up the tree
        for node in reversed(path):
            node.backup(value)
            value = -value  # Flip value for opponent
        
        return value
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node and evaluate with neural network.
        
        Args:
            node: Node to expand and evaluate
            
        Returns:
            Value estimate from neural network
        """
        # Get legal moves
        legal_moves = node.state.get_legal_moves()
        
        if not legal_moves:
            # No legal moves - this shouldn't happen in antichess normally
            logger.warning("No legal moves found in non-terminal position")
            return 0.0
        
        # Encode position for neural network
        position_tensor = self.encoder.encode_board(node.state.board)
        position_tensor = torch.FloatTensor(position_tensor).unsqueeze(0)
        
        # Get neural network predictions
        with torch.no_grad():
            policy_logits, value = self.neural_net(position_tensor)
            
            # Convert policy logits to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1)
            
            # Decode policy probabilities for legal moves
            move_probs = self._decode_policy(policy_probs[0], legal_moves)
            
            # Get value estimate
            value_estimate = torch.tanh(value[0]).item()
        
        # Expand node with neural network policy
        node.expand(move_probs, legal_moves)
        
        return value_estimate
    
    def _decode_policy(self, policy_probs: torch.Tensor, legal_moves: List[chess.Move]) -> Dict[chess.Move, float]:
        """
        Decode neural network policy output for legal moves.
        
        Args:
            policy_probs: Neural network policy probabilities (4096,)
            legal_moves: List of legal moves
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        move_probs = {}
        total_prob = 0.0
        
        # Get probabilities for each legal move
        for move in legal_moves:
            try:
                action_index = self.encoder._move_to_action(move)
                prob = policy_probs[action_index].item()
                move_probs[move] = prob
                total_prob += prob
            except Exception as e:
                logger.warning(f"Failed to encode move {move}: {e}")
                move_probs[move] = 0.0
        
        # Normalize probabilities
        if total_prob > 0:
            for move in move_probs:
                move_probs[move] /= total_prob
        else:
            # Uniform distribution if no valid probabilities
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probs[move] = uniform_prob
        
        return move_probs
    
    def _add_dirichlet_noise(self, root: MCTSNode, legal_moves: List[chess.Move]):
        """
        Add Dirichlet noise to root node for exploration.
        
        Args:
            root: Root node to add noise to
            legal_moves: Legal moves from root position
        """
        if not legal_moves:
            return
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
        
        # First expand root to get policy probabilities
        position_tensor = self.encoder.encode_board(root.state.board)
        position_tensor = torch.FloatTensor(position_tensor).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, _ = self.neural_net(position_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)
            move_probs = self._decode_policy(policy_probs[0], legal_moves)
        
        # Apply Dirichlet noise
        for i, move in enumerate(legal_moves):
            original_prob = move_probs.get(move, 0.0)
            noisy_prob = ((1 - self.config.dirichlet_epsilon) * original_prob + 
                         self.config.dirichlet_epsilon * noise[i])
            move_probs[move] = noisy_prob
        
        # Expand root with noisy probabilities
        root.expand(move_probs, legal_moves)
    
    def _get_move_probabilities(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """
        Convert visit counts to move probabilities.
        
        Args:
            root: Root node with visit statistics
            
        Returns:
            Dictionary mapping moves to selection probabilities
        """
        if not root.children:
            return {}
        
        # Get visit counts for each move
        move_visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_visits.values())
        
        if total_visits == 0:
            # No visits, return uniform distribution
            num_moves = len(root.children)
            return {move: 1.0 / num_moves for move in root.children.keys()}
        
        # Convert visits to probabilities
        move_probs = {move: visits / total_visits for move, visits in move_visits.items()}
        
        return move_probs
    
    def get_best_move(self, root_state: AntichessGame, temperature: float = 0.0) -> chess.Move:
        """
        Get best move using MCTS.
        
        Args:
            root_state: Current game state
            temperature: Temperature for move selection (0 = greedy, 1 = proportional)
            
        Returns:
            Best move according to MCTS
        """
        move_probs = self.search(root_state)
        
        if not move_probs:
            # No moves available
            legal_moves = root_state.get_legal_moves()
            if legal_moves:
                return legal_moves[0]
            else:
                raise ValueError("No legal moves available")
        
        if temperature == 0.0:
            # Greedy selection
            return max(move_probs.keys(), key=lambda move: move_probs[move])
        else:
            # Temperature-based sampling
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.array(probs) ** (1.0 / temperature)
                probs = probs / np.sum(probs)
            
            # Sample move
            return np.random.choice(moves, p=probs)
    
    def get_search_statistics(self, root_state: AntichessGame) -> Dict[str, float]:
        """
        Get detailed search statistics for analysis.
        
        Args:
            root_state: Game state to analyze
            
        Returns:
            Dictionary of search statistics
        """
        move_probs = self.search(root_state)
        
        if not move_probs:
            return {}
        
        return {
            'total_moves': len(move_probs),
            'max_probability': max(move_probs.values()),
            'min_probability': min(move_probs.values()),
            'entropy': -sum(p * math.log(p + 1e-8) for p in move_probs.values()),
            'best_move_prob': max(move_probs.values()),
        }


# Export main classes
__all__ = ['MCTS', 'MCTSNode']
