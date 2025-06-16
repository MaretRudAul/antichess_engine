"""
Environment module for Antichess reinforcement learning.

This module provides gymnasium-compatible environments for training RL agents
on antichess (losing chess). It includes both single and vectorized environments
for efficient training and evaluation.
"""

from .antichess_env import AntichessEnv, AntichessVecEnv, OpponentType

__all__ = ['AntichessEnv', 'AntichessVecEnv', 'OpponentType']