# Antichess Reinforcement Learning with PPO

This project implements a sophisticated reinforcement learning agent for the chess variant "Antichess" using Proximal Policy Optimization (PPO). In Antichess, the goal is to lose all your pieces or have no legal moves, with captures being mandatory when available.

## Key Features

- **Advanced Curriculum Learning**: Multi-phase training progression from random opponents to self-play
- **Hyperparameter Optimization**: Automated Bayesian optimization using Optuna for optimal training configuration
- **Adaptive Learning Rate Scheduling**: Combined linear and cosine annealing with curriculum-aware transitions
- **Action Masking**: Ensures only legal moves are selected, dramatically improving sample efficiency
- **Self-Play Training**: Progressive opponent strategies including model-based self-play
- **Custom CNN Architecture**: AlphaZero-inspired convolutional neural network for board evaluation
- **Comprehensive Evaluation System**: Built-in performance monitoring and model comparison tools
- **Flexible Training Modes**: Multiple opponent strategies and training configurations

## Project Structure

```
antichess_engine/
├── antichess/           # Game logic and rules
│   ├── __init__.py
│   ├── board.py         # Core Antichess game implementation
│   └── utils.py         # Board encoding and move conversion utilities
├── envs/                # Gym-compatible environment
│   ├── __init__.py
│   └── antichess_env.py # OpenAI Gym environment with action masking
├── models/              # Custom neural network models
│   ├── __init__.py
│   └── custom_policy.py # CNN feature extractor and masked policy
├── train/               # Training logic
│   ├── __init__.py
│   └── train_ppo.py     # Main training script with full CLI
├── evaluate/            # Evaluation scripts
│   ├── __init__.py
│   └── evaluate_policy.py # Model evaluation and performance analysis
├── callbacks/           # Training callbacks
│   ├── __init__.py
│   └── callbacks.py     # Curriculum, self-play, and monitoring callbacks
├── schedules/           # Learning rate schedules
│   ├── __init__.py
│   └── schedules.py     # Linear, cosine, and curriculum-aware schedules
├── hyperopt/            # Hyperparameter optimization
│   ├── __init__.py
│   ├── optimize.py      # Optuna-based hyperparameter optimization
│   ├── manage.py        # Results management and comparison
│   └── defaults.json    # Default configuration parameters
├── config.py            # Centralized configuration and hyperparameter loading
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore patterns
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/antichess_engine.git
cd antichess_engine

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Basic Training (Recommended)

Start with the enhanced curriculum learning approach, which provides the most robust training:

```bash
python -m train.train_ppo --use-enhanced-curriculum
```

This automatically progresses through four training phases:

1. **Phase 1 (15% of training)**: Random opponents only
2. **Phase 2 (20% of training)**: Mixed random (40%) and heuristic (60%) opponents
3. **Phase 3 (25% of training)**: Mixed random (25%), heuristic (25%), and self-play (50%)
4. **Phase 4 (40% of training)**: Primarily self-play (80%) with some random/heuristic mix

### Evaluating a Trained Model

```bash
python -m evaluate.evaluate_policy --model trained_models/final_model.zip --episodes 100
```

## Training Strategies

### 1. Enhanced Curriculum Learning (Recommended)

```bash
python -m train.train_ppo --use-enhanced-curriculum --total-timesteps 2000000
```

**Features:**

- **Four-phase progression** with automatic transitions
- **Curriculum-aware learning rate scheduling** (linear → linear → cosine annealing)
- **Smooth phase transitions** with blended opponent strategies
- **Adaptive self-play integration** starting in phase 3

**Benefits:**

- Most robust learning progression
- Handles the explore-exploit tradeoff effectively
- Prevents overfitting to specific opponent types
- Optimized learning rates for each training phase

### 2. Simple Curriculum Learning

```bash
python -m train.train_ppo --opponent curriculum --self-play-start 400000
```

**Features:**

- **Two-phase training**: Random opponents → Self-play
- **Configurable transition point** and self-play probability
- **Mixed self-play**: Combines model and random moves

### 3. Pure Self-Play

```bash
python -m train.train_ppo --opponent self_play --self-play-prob 0.8
```

**Features:**

- **Immediate self-play training** from the start
- **Configurable model vs random ratio** to prevent overfitting
- **Periodic model updates** for opponent diversity

### 4. Mixed Opponent Training

```bash
python -m train.train_ppo --opponent mixed --random-prob 0.3 --heuristic-prob 0.5
```

**Features:**

- **Customizable opponent mix** with flexible probability weights
- **Episode-level opponent switching** for diverse experience
- **Supports random, heuristic, and self-play opponents**

### 5. Single Opponent Training

```bash
# Random opponents only
python -m train.train_ppo --opponent random

# Heuristic opponents only
python -m train.train_ppo --opponent heuristic
```

## Advanced Training Options

### Learning Rate Scheduling

The system includes sophisticated learning rate scheduling:

**Combined Linear + Cosine Schedule (Default):**

```bash
python -m train.train_ppo  # Uses 60% linear, 40% cosine decay
```

**Curriculum-Aware Schedule:**

```bash
python -m train.train_ppo --use-enhanced-curriculum
# Automatically aligns learning rate with curriculum phases
```

### High-Performance Training

```bash
# Extended training with more parallel environments
python -m train.train_ppo --total-timesteps 5000000 --num-envs 16 --use-enhanced-curriculum

# GPU training with custom learning rate
python -m train.train_ppo --device cuda --total-timesteps 3000000
```

### Training Control and Resumption

```bash
# Resume from latest checkpoint automatically
python -m train.train_ppo --opponent curriculum

# Resume from specific checkpoint
python -m train.train_ppo --resume-from trained_models/checkpoint_1000000_steps.zip

# Start fresh training (ignore checkpoints)
python -m train.train_ppo --no-resume --opponent self_play

# Custom output directories
python -m train.train_ppo --log-dir ./experiment_1 --model-dir ./models_1
```

### Self-Play Configuration

```bash
# Conservative self-play (50% model, 50% random)
python -m train.train_ppo --opponent self_play --self-play-prob 0.5

# Aggressive self-play (90% model, 10% random)
python -m train.train_ppo --opponent self_play --self-play-prob 0.9

# Frequent model updates for self-play diversity
python -m train.train_ppo --opponent curriculum --self-play-update-freq 25000
```

## Hyperparameter Optimization

The project includes a comprehensive hyperparameter optimization system using **Optuna** for Bayesian optimization. This can significantly improve training performance by finding optimal hyperparameters automatically.

### Quick Start with Hyperparameter Optimization

```bash
# 1. Run hyperparameter optimization
python -m hyperopt.optimize --n-trials 50 --training-timesteps 200000

# 2. Train with optimized hyperparameters (automatically loaded)
python -m train.train_ppo --opponent curriculum --total-timesteps 2000000

# 3. Or manually specify optimization results
python -m train.train_ppo --hyperopt-path hyperopt_results/optimization_results_20231208_143022.json
```

### Hyperparameter Optimization Features

- **Bayesian Optimization**: Uses TPE (Tree-structured Parzen Estimator) for efficient search
- **Early Stopping**: Prunes poor trials using median pruning for faster optimization
- **Extensive Search Space**: Optimizes 15+ hyperparameters including:
  - Learning rates (initial, final, schedule type)
  - Network architecture (layer sizes, feature dimensions)
  - PPO parameters (batch size, epochs, regularization)
  - Training parameters (gamma, GAE lambda, clipping)
- **Automatic Integration**: Optimized parameters are automatically loaded when available
- **Trial Management**: Save, load, and compare optimization results

### Running Hyperparameter Optimization

```bash
# Full optimization with 100 trials (recommended)
python -m hyperopt.optimize

# Quick optimization with fewer trials
python -m hyperopt.optimize --n-trials 25 --training-timesteps 100000

# Parallel optimization (if you have multiple GPUs/cores)
python -m hyperopt.optimize --n-jobs 2

# Resume existing optimization study
python -m hyperopt.optimize --study-name my_study --load-study

# View results of completed optimization
python -m hyperopt.optimize --show-results --study-name antichess_hyperopt
```

### Managing Optimization Results

```bash
# List all optimization results
python -c "from hyperopt.manage import list_optimization_results; print(list_optimization_results())"

# Compare multiple optimization runs
python -c "from hyperopt.manage import compare_optimization_results; compare_optimization_results(['result1.json', 'result2.json'])"

# Set environment variable to always use optimized hyperparameters
export ANTICHESS_HYPEROPT_PATH="hyperopt_results/best_results.json"
```

### Hyperparameter Search Spaces

The optimization searches over these parameter ranges:

- **Learning Rate**: 1e-6 to 1e-3 (log scale)
- **Network Architecture**:
  - Feature dimensions: 256, 512, 1024
  - Policy layers: 64-1024 neurons per layer
  - Value layers: 64-1024 neurons per layer
- **PPO Parameters**:
  - Batch size: 32, 64, 128, 256
  - Training epochs: 3-20
  - Discount factor: 0.9-0.9999
- **Regularization**:
  - Entropy coefficient: 1e-6 to 1e-1 (log scale)
  - Value function coefficient: 0.1-2.0
  - Gradient clipping: 0.1-2.0

### Optimization Tips

1. **Start Small**: Use `--n-trials 10-25` for initial exploration
2. **Short Training**: Use `--training-timesteps 100000-200000` for faster trials
3. **Monitor Progress**: Results are saved incrementally in `hyperopt_results/`
4. **Multiple Runs**: Run optimization multiple times with different random seeds
5. **Integration**: Optimized hyperparameters are automatically loaded by `config.py`

## Complete Command Reference

### Training Configuration

- `--opponent`: Opponent strategy (`random`, `heuristic`, `self_play`, `curriculum`, `mixed`)
- `--total-timesteps`: Total training timesteps (default: 2,000,000)
- `--num-envs`: Number of parallel environments (default: 8)
- `--use-enhanced-curriculum`: Enable advanced multi-phase curriculum learning

### Curriculum and Self-Play Options

- `--self-play-start`: When to start self-play (curriculum mode, default: 400,000)
- `--self-play-prob`: Model vs random probability in self-play (default: 0.8)
- `--self-play-update-freq`: Update self-play model every N timesteps (default: 50,000)

### Mixed Opponent Configuration

- `--random-prob`: Random opponent probability (mixed mode, default: 0.5)
- `--heuristic-prob`: Heuristic opponent probability (mixed mode, default: 0.3)
- Remaining probability automatically assigned to self-play

### Training Control

- `--no-resume`: Start fresh, ignore existing checkpoints
- `--resume-from`: Resume from specific checkpoint file
- `--eval-freq`: Evaluation frequency (default: 25,000)
- `--checkpoint-freq`: Checkpoint save frequency (default: 400,000)

### System Configuration

- `--device`: Training device (`auto`, `cpu`, `cuda`)
- `--seed`: Random seed for reproducible results
- `--verbose`: Enable detailed training output

### Output Options

- `--log-dir`: Custom log directory path
- `--model-dir`: Custom model save directory path

## Architecture Details

### CNN Feature Extractor

**Board Representation:**

- **13-plane input**: 12 piece type planes (6 white + 6 black) + 1 turn plane
- **8x8 spatial dimensions** preserving chess board structure

**Network Architecture:**

- **Initial convolution**: 64 filters, 3x3 kernels with batch normalization
- **Residual blocks**: Two 64-filter residual blocks with ReLU activation
- **Feature extraction**: 512-dimensional learned features
- **Separate networks**: Independent policy (π) and value (V) networks

### Action Space and Masking

**Action Representation:**

- **4096 discrete actions**: 64×64 source-target square combinations
- **Promotion handling**: Automatic queen promotion for pawn advancement
- **Legal move masking**: Invalid actions have probability ≈ 0

**Action Masking Benefits:**

- **Eliminates illegal moves** during training and evaluation
- **Dramatically improves sample efficiency** by focusing on valid actions
- **Prevents training instability** from illegal move penalties

### PPO Hyperparameters

**Core PPO Settings:**

- **Learning rate**: Adaptive scheduling (1e-4 → 1e-6)
- **Batch size**: 128 samples per update
- **Training epochs**: 10 epochs per batch
- **Discount factor (γ)**: 0.99
- **GAE lambda**: 0.95
- **Clip range**: 0.2
- **Entropy coefficient**: 0.01 (encourages exploration)

**Network Architecture:**

- **Policy network**: [512, 256, 128] fully connected layers
- **Value network**: [512, 256, 128] fully connected layers
- **Feature extractor**: 512-dimensional CNN features

## Monitoring and Evaluation

### TensorBoard Integration

```bash
# View training progress
tensorboard --logdir logs/

# Monitor specific experiment
tensorboard --logdir logs/antichess_curriculum_20241201_143022/
```

**Available Metrics:**

- Training rewards and episode lengths
- Policy and value function losses
- Gradient norms and entropy
- Evaluation performance against different opponents
- Learning rate schedules and curriculum phases

### Training Logs

**Automatic Logging:**

- **Progress metrics**: `logs/*/progress.csv`
- **Model checkpoints**: `trained_models/*/checkpoint_*.zip`
- **Evaluation results**: `logs/*/evaluations.npz`
- **Best model saves**: `trained_models/*/best_model.zip`

### Performance Evaluation

```bash
# Comprehensive evaluation
python -m evaluate.evaluate_policy --model trained_models/final_model.zip --episodes 100

# Quick performance check
python -m evaluate.evaluate_policy --model trained_models/best_model.zip --episodes 50
```

**Evaluation Metrics:**

- Win rate against random, heuristic, and previous model versions
- Average episode length and decision quality
- Illegal action rate (should be ~0% with action masking)
- Strategic strength assessment

## Antichess Rules Implementation

### Core Rules

**Objective**: Lose all pieces or achieve a position with no legal moves

**Mandatory Captures**: When captures are available, they must be made (no other moves allowed)

**No Royal Restrictions**:

- No check, checkmate, or castling rules
- King moves like any other piece
- King can be captured like any other piece

**Pawn Promotion**: Pawns reaching the back rank promote (default: Queen)

**Win Conditions**:

- Player with no remaining pieces wins
- Player with no legal moves wins

### Strategic Considerations

**Piece Sacrifice**: Actively losing pieces is often advantageous

**Forced Sequences**: Mandatory captures create forcing move sequences

**Endgame Complexity**: Few pieces can lead to complex tactical positions

**Opening Strategy**: Standard chess openings may be counterproductive

## Example Training Workflows

### Research and Development

```bash
# Quick baseline experiment (500k steps)
python -m train.train_ppo --opponent random --total-timesteps 500000

# Full curriculum comparison
python -m train.train_ppo --use-enhanced-curriculum --total-timesteps 2000000 --seed 42

# Self-play strength analysis
python -m train.train_ppo --opponent self_play --self-play-prob 0.7 --total-timesteps 1000000
```

### Production Training

```bash
# High-performance training setup
python -m train.train_ppo \
    --use-enhanced-curriculum \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --device cuda \
    --eval-freq 50000 \
    --checkpoint-freq 200000 \
    --verbose

# Reproducible research run
python -m train.train_ppo \
    --use-enhanced-curriculum \
    --seed 12345 \
    --total-timesteps 3000000 \
    --log-dir ./research_run_1 \
    --model-dir ./research_models_1
```

### Hyperparameter Experimentation

```bash
# Conservative self-play curriculum
python -m train.train_ppo --opponent curriculum --self-play-start 600000 --self-play-prob 0.6

# Aggressive early self-play
python -m train.train_ppo --opponent curriculum --self-play-start 200000 --self-play-prob 0.9

# Frequent model updates
python -m train.train_ppo --opponent self_play --self-play-update-freq 25000
```

## Technical Implementation

### Multiprocessing Architecture

**Parallel Environments**: Uses `SubprocVecEnv` for efficient parallel training

**Shared Memory Self-Play**: Efficient model weight sharing between processes

**Gradient Monitoring**: Real-time gradient norm tracking to detect training instabilities

### Callback System

**Modular Design**: Extensible callback architecture for custom training logic

**Available Callbacks:**

- `EnhancedCurriculumCallback`: Multi-phase curriculum management
- `SelfPlayCallback`: Simple curriculum with self-play transition
- `UpdateSelfPlayModelCallback`: Periodic model updates for opponent diversity
- `MaskedEvalCallback`: Action-masked evaluation with comprehensive metrics
- `GradientMonitorCallback`: Training stability monitoring

## Future Development

### Planned Enhancements

**Advanced Architectures**: Possible integration of attention mechanisms

**Distributed Training**: Ray framework integration for large-scale distributed training

**Human Interface**: GUI integration for human vs AI gameplay

**Tournament System**: Automated model comparison and ranking system

**Hyperparameter Optimization**: Automated tuning with Optuna integration

**Interpretability**: Analysis of learned strategies and decision patterns

## Contributing

We welcome contributions to improve the Antichess engine:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Implement your changes** with appropriate tests
4. **Commit changes**: `git commit -am 'Add feature description'`
5. **Push to branch**: `git push origin feature-name`
6. **Submit a pull request** with detailed description

## License

This project is open source. Please refer to the LICENSE file for specific terms and conditions.

## Acknowledgments

**Stable-Baselines3**: Robust PPO implementation and training infrastructure

**OpenAI Gym**: Standard reinforcement learning environment interface

**PyTorch**: Deep learning framework for neural network implementation

**AlphaZero**: Inspiration for CNN architecture and self-play training methodology
