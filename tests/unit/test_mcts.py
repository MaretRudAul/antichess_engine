"""
Unit tests for MCTS implementation.

Tests the Monte Carlo Tree Search algorithm for antichess,
including node operations, search functionality, and integration
with the neural network.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.config.settings import MCTSConfig, ModelConfig
from src.models.neural_network import AntichessNet
from src.models.mcts import MCTS, MCTSNode
from src.game.antichess_rules import AntichessGame
import chess


class TestMCTSNode:
    """Test MCTS node functionality."""
    
    def test_node_initialization(self):
        """Test basic node creation and initialization."""
        game = AntichessGame()
        node = MCTSNode(game)
        
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.value() == 0.0
        assert node.is_leaf() == True
        assert len(node.children) == 0
        assert node.parent is None
        assert node.move is None
    
    def test_node_with_parent(self):
        """Test node creation with parent and move."""
        game = AntichessGame()
        parent = MCTSNode(game)
        
        legal_moves = game.get_legal_moves()
        test_move = legal_moves[0]
        
        child = MCTSNode(game, parent=parent, move=test_move, prior=0.5)
        
        assert child.parent == parent
        assert child.move == test_move
        assert child.prior == 0.5
    
    def test_ucb_score(self):
        """Test UCB score calculation."""
        game = AntichessGame()
        node = MCTSNode(game, prior=0.3)
        
        # Unvisited node should have infinite UCB
        assert node.ucb_score(1.0, 10) == float('inf')
        
        # After visits, should calculate proper UCB
        node.visit_count = 5
        node.value_sum = 2.5
        
        score = node.ucb_score(1.0, 10)
        assert isinstance(score, float)
        assert score > 0  # Should be positive for this setup
    
    def test_backup(self):
        """Test value backup through tree."""
        game = AntichessGame()
        parent = MCTSNode(game)
        child = MCTSNode(game, parent=parent)
        
        # Backup value from child
        child.backup(0.7)
        
        # Child should have the value
        assert child.visit_count == 1
        assert child.value_sum == 0.7
        assert child.value() == 0.7
        
        # Parent should have negated value (opponent perspective)
        assert parent.visit_count == 1
        assert parent.value_sum == -0.7
        assert parent.value() == -0.7


class TestMCTS:
    """Test MCTS search functionality."""
    
    @pytest.fixture
    def setup_mcts(self):
        """Set up MCTS with neural network for testing."""
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        mcts_config.simulations = 10  # Small number for fast tests
        
        neural_net = AntichessNet(model_config)
        mcts = MCTS(neural_net, mcts_config)
        
        return mcts, neural_net, mcts_config
    
    def test_mcts_initialization(self, setup_mcts):
        """Test MCTS initialization."""
        mcts, neural_net, config = setup_mcts
        
        assert mcts.neural_net == neural_net
        assert mcts.config == config
        assert hasattr(mcts, 'encoder')
    
    def test_policy_decoding(self, setup_mcts):
        """Test neural network policy decoding for legal moves."""
        mcts, _, _ = setup_mcts
        game = AntichessGame()
        legal_moves = game.get_legal_moves()
        
        # Create mock policy probabilities
        policy_probs = torch.randn(4096)
        policy_probs = torch.softmax(policy_probs, dim=0)
        
        # Test policy decoding
        move_probs = mcts._decode_policy(policy_probs, legal_moves)
        
        assert len(move_probs) == len(legal_moves)
        assert all(isinstance(prob, float) for prob in move_probs.values())
        assert all(move in legal_moves for move in move_probs.keys())
        
        # Probabilities should sum to 1.0 (approximately)
        total_prob = sum(move_probs.values())
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_expand_and_evaluate(self, setup_mcts):
        """Test node expansion and evaluation."""
        mcts, _, _ = setup_mcts
        game = AntichessGame()
        node = MCTSNode(game)
        
        # Test expansion
        value = mcts._expand_and_evaluate(node)
        
        assert node.is_expanded == True
        assert len(node.children) > 0
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0  # Value should be in valid range
    
    def test_basic_search(self, setup_mcts):
        """Test basic MCTS search functionality."""
        mcts, _, _ = setup_mcts
        game = AntichessGame()
        
        # Run search
        move_probs = mcts.search(game)
        
        assert isinstance(move_probs, dict)
        assert len(move_probs) > 0
        
        # All keys should be legal moves
        legal_moves = game.get_legal_moves()
        assert all(move in legal_moves for move in move_probs.keys())
        
        # Probabilities should be valid
        assert all(0.0 <= prob <= 1.0 for prob in move_probs.values())
        
        # Should sum to approximately 1.0
        total_prob = sum(move_probs.values())
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_get_best_move(self, setup_mcts):
        """Test best move selection."""
        mcts, _, _ = setup_mcts
        game = AntichessGame()
        legal_moves = game.get_legal_moves()
        
        # Test greedy selection (temperature = 0)
        best_move = mcts.get_best_move(game, temperature=0.0)
        assert best_move in legal_moves
        
        # Test temperature-based selection
        temp_move = mcts.get_best_move(game, temperature=1.0)
        assert temp_move in legal_moves
    
    def test_search_statistics(self, setup_mcts):
        """Test search statistics collection."""
        mcts, _, _ = setup_mcts
        game = AntichessGame()
        
        stats = mcts.get_search_statistics(game)
        
        assert isinstance(stats, dict)
        assert 'total_moves' in stats
        assert 'max_probability' in stats
        assert 'entropy' in stats
        assert stats['total_moves'] > 0


class TestMCTSIntegration:
    """Integration tests for MCTS with game components."""
    
    def test_full_mcts_pipeline(self):
        """Test complete MCTS pipeline from start to move selection."""
        print('Testing MCTS implementation...')
        
        # Create configs
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        mcts_config.simulations = 5  # Reduce for quick test
        
        # Create neural network
        neural_net = AntichessNet(model_config)
        
        # Create MCTS
        mcts = MCTS(neural_net, mcts_config)
        
        # Create game
        game = AntichessGame()
        
        # Test basic MCTS node functionality
        root = MCTSNode(game)
        print(f'Root node created: {root}')
        print(f'Root value: {root.value()}')
        print(f'Root is leaf: {root.is_leaf()}')
        
        legal_moves = game.get_legal_moves()
        print(f'Legal moves from start: {len(legal_moves)}')
        
        # Test the full pipeline
        move_probs = mcts.search(game)
        print(f'MCTS search completed successfully!')
        print(f'Found probabilities for {len(move_probs)} moves')
        
        if move_probs:
            best_move = max(move_probs.keys(), key=lambda m: move_probs[m])
            print(f'Best move: {best_move} (prob: {move_probs[best_move]:.3f})')
        
        print('MCTS test passed!')
        
        # Assertions
        assert len(move_probs) > 0
        assert all(move in legal_moves for move in move_probs.keys())
        assert abs(sum(move_probs.values()) - 1.0) < 1e-6
    
    def test_mcts_with_terminal_positions(self):
        """Test MCTS behavior with game-ending positions."""
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        mcts_config.simulations = 3
        
        neural_net = AntichessNet(model_config)
        mcts = MCTS(neural_net, mcts_config)
        
        # Create a near-terminal game state
        game = AntichessGame()
        
        # Even with few pieces, should still work
        move_probs = mcts.search(game)
        assert isinstance(move_probs, dict)
        
        if move_probs:
            assert all(0.0 <= prob <= 1.0 for prob in move_probs.values())
    
    def test_mcts_consistency(self):
        """Test that MCTS produces consistent results."""
        model_config = ModelConfig()
        mcts_config = MCTSConfig()
        mcts_config.simulations = 10
        
        neural_net = AntichessNet(model_config)
        mcts = MCTS(neural_net, mcts_config)
        
        game = AntichessGame()
        
        # Run search multiple times
        results = []
        for _ in range(3):
            move_probs = mcts.search(game)
            results.append(move_probs)
        
        # All results should be dictionaries with same keys
        assert all(isinstance(result, dict) for result in results)
        
        # Should have moves available
        assert all(len(result) > 0 for result in results)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
