import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.antichess_env import AntichessEnv
from models.custom_policy import ChessCNN, MaskedActorCriticPolicy
from config import EVALUATION_PARAMS

def evaluate_policy(model_path, n_episodes=100, render=False, opponent="random"):
    """
    Evaluate a trained model against different opponents.
    
    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        opponent: Opponent strategy
        
    Returns:
        Evaluation statistics
    """
    # Load the trained model with correct policy class
    print(f"Loading model from {model_path}")
    try:
        model = PPO.load(
            model_path, 
            custom_objects={
                "policy_class": MaskedActorCriticPolicy,
                "policy": MaskedActorCriticPolicy
            }
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create the environment
    env = AntichessEnv(opponent=opponent)
    
    # Statistics
    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    episode_lengths = []
    rewards = []
    
    print(f"Evaluating model against {opponent} opponent for {n_episodes} episodes...")
    
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Debug first episode
        if episode == 0:
            print(f"First episode observation type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"Observation keys: {obs.keys()}")
                print(f"Observation shape: {obs['observation'].shape}")
                print(f"Action mask shape: {obs['action_mask'].shape}")
                print(f"Legal moves count: {np.sum(obs['action_mask'])}")
        
        while not done:
            # Apply action masking manually for evaluation
            action = predict_with_mask(model, obs, deterministic=True)
            
            # Debug first few moves
            if episode == 0 and steps < 3:
                action_mask = obs['action_mask'] if isinstance(obs, dict) else None
                if action_mask is not None:
                    is_legal = action_mask[action] == 1.0
                    legal_count = np.sum(action_mask)
                    print(f"Step {steps}: Action {action}, Legal: {is_legal}, Legal moves: {legal_count}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.3)  # Slow down rendering
            
            # Safety check - prevent infinite episodes
            if steps > 200:  # Reasonable upper bound for Antichess
                print(f"Episode {episode} exceeded 200 steps, terminating")
                done = True
                draws += 1
                break
            
            if done:
                if "illegal_move" in info and info["illegal_move"]:
                    illegal_moves += 1
                    if episode < 5:  # Debug first few episodes
                        print(f"Episode {episode}: Illegal move detected at step {steps}")
                elif "winner" in info:
                    if info["winner"] == env.player_color:
                        wins += 1
                    elif info["winner"] is None:
                        draws += 1
                    else:
                        losses += 1
                else:
                    draws += 1
        
        episode_lengths.append(steps)
        rewards.append(episode_reward)
        
        # Debug first few episodes
        if episode < 5:
            print(f"Episode {episode}: {steps} steps, reward: {episode_reward}")
    
    # Calculate statistics
    win_rate = wins / n_episodes * 100
    loss_rate = losses / n_episodes * 100
    draw_rate = draws / n_episodes * 100
    illegal_rate = illegal_moves / n_episodes * 100
    avg_episode_length = np.mean(episode_lengths)
    avg_reward = np.mean(rewards)
    
    print(f"\nEvaluation against {opponent} complete:")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Loss rate: {loss_rate:.2f}%")
    print(f"Draw rate: {draw_rate:.2f}%")
    print(f"Illegal moves: {illegal_rate:.2f}%")
    print(f"Average episode length: {avg_episode_length:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Episode length range: {min(episode_lengths)} - {max(episode_lengths)}")
    
    stats = {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "illegal_rate": illegal_rate,
        "avg_episode_length": avg_episode_length,
        "avg_reward": avg_reward,
        "episode_lengths": episode_lengths,
        "rewards": rewards
    }
    
    return stats

def predict_with_mask(model, obs, deterministic=True):
    """
    Make a prediction using the model while respecting action masks.
    
    Args:
        model: The trained PPO model
        obs: Observation (should be a dict with 'observation' and 'action_mask')
        deterministic: Whether to use deterministic action selection
        
    Returns:
        Selected action index
    """
    if not isinstance(obs, dict) or 'action_mask' not in obs:
        # Fallback to standard prediction if no mask available
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    
    # Convert observations to tensors
    device = next(model.policy.parameters()).device
    
    # Handle single observation (not batched)
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            # Add batch dimension if needed
            if value.ndim == len(model.observation_space[key].shape):
                value = np.expand_dims(value, axis=0)
            obs_tensor[key] = torch.from_numpy(value).float().to(device)
        else:
            obs_tensor[key] = value
    
    # Use the policy's forward method directly
    with torch.no_grad():
        actions, values, log_probs = model.policy.forward(obs_tensor, deterministic=deterministic)
    
    # Convert back to numpy and remove batch dimension
    action = actions.cpu().numpy()
    if action.ndim > 0:
        action = action[0]
    
    return int(action)

def plot_results(stats_dict):
    """
    Plot evaluation results.
    
    Args:
        stats_dict: Dictionary of stats for each opponent
    """
    opponents = list(stats_dict.keys())
    win_rates = [stats["win_rate"] for stats in stats_dict.values()]
    loss_rates = [stats["loss_rate"] for stats in stats_dict.values()]
    illegal_rates = [stats["illegal_rate"] for stats in stats_dict.values()]
    
    # Plot win rates
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.bar(opponents, win_rates, color='green')
    plt.title('Win Rates')
    plt.ylabel('Percentage')
    
    # Plot loss rates
    plt.subplot(2, 2, 2)
    plt.bar(opponents, loss_rates, color='red')
    plt.title('Loss Rates')
    
    # Plot illegal move rates
    plt.subplot(2, 2, 3)
    plt.bar(opponents, illegal_rates, color='orange')
    plt.title('Illegal Move Rates')
    plt.ylabel('Percentage')
    
    # Plot average episode lengths
    avg_lengths = [stats["avg_episode_length"] for stats in stats_dict.values()]
    plt.subplot(2, 2, 4)
    plt.bar(opponents, avg_lengths, color='blue')
    plt.title('Average Episode Length')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()

def main():
    """Evaluate a trained model against multiple opponents."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Antichess agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--episodes", type=int, default=EVALUATION_PARAMS["n_episodes"],
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    # Test with just one opponent first
    print("Testing with random opponent first...")
    stats = evaluate_policy(
        args.model, 
        n_episodes=min(10, args.episodes),  # Start with fewer episodes for testing
        render=args.render,
        opponent="random"
    )
    
    if stats is None:
        print("Failed to load model or evaluate. Exiting.")
        return
    
    if stats["illegal_rate"] > 50:
        print(f"WARNING: High illegal move rate ({stats['illegal_rate']:.1f}%). Check action masking.")
    
    if stats["avg_episode_length"] < 5:
        print(f"WARNING: Very short episodes ({stats['avg_episode_length']:.1f} moves). Check policy prediction.")
    
    # If basic test passes, run full evaluation
    if stats["illegal_rate"] < 50 and stats["avg_episode_length"] > 3:
        print("Basic test passed. Running full evaluation...")
        
        # Evaluate against different opponents
        stats_dict = {}
        for opponent in EVALUATION_PARAMS["opponents"]:
            stats = evaluate_policy(
                args.model, 
                n_episodes=args.episodes, 
                render=args.render,
                opponent=opponent
            )
            if stats is not None:
                stats_dict[opponent] = stats
        
        if stats_dict:
            # Plot results
            plot_results(stats_dict)
            
            # Save results
            np.save("evaluation_results.npy", stats_dict)
    else:
        print("Basic test failed. Please check your model and environment setup.")

if __name__ == "__main__":
    main()