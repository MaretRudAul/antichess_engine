# Antichess Reinforcement Learning Project

## Overview

This project implements a professional-grade reinforcement learning system that masters antichess (losing chess) using self-play and modern deep learning techniques, based on the AlphaZero architecture.

## Objective

Achieve superhuman performance in antichess through AlphaZero-style training with custom adaptations for antichess rule variants.

## Tech Stack

- **Python 3.9+** with PyTorch 2.0+
- **PyTorch Lightning** for structured training
- **Weights & Biases** for experiment tracking
- **Ray RLlib** for distributed training
- **python-chess** for game engine foundation
- **Docker** for containerization
- **Redis** & **MongoDB** for data management

## Project Structure

```
antichess_engine/
├── src/
│   ├── game/              # Game logic and rules
│   ├── environment/       # RL environment interface
│   ├── models/            # Neural networks and MCTS
│   ├── training/          # Training infrastructure
│   ├── utils/             # Utilities and helpers
│   └── config/            # Configuration management
├── tests/                 # Test suites
├── scripts/               # Training and deployment scripts
├── data/                  # Training data storage
├── models/                # Saved model checkpoints
├── logs/                  # Training logs
├── docker/                # Docker configuration
└── configs/               # Configuration files
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd antichess_engine

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For GPU support
pip install -e ".[gpu]"
```

## Quick Start

```bash
# Start training with default configuration
python scripts/train.py --config configs/default.yaml

# Run distributed training
python scripts/distributed_train.py --workers 4

# Deploy locally with Docker
python scripts/deploy.py local
```

## Development

This project follows professional development practices:

- **Type checking** with mypy
- **Code formatting** with black and isort
- **Testing** with pytest
- **Experiment tracking** with Weights & Biases
- **Containerization** with Docker

## License

MIT License - see LICENSE file for details.
