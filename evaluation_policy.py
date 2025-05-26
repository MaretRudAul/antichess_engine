import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO

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
    # Load the trained model
    model = PPO.load(model_path, custom_objects={"policy": MaskedActorCriticPolicy})
    
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
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.3)  # Slow down rendering
            
            if done:
                if "illegal_move" in info and info["illegal_move"]:
                    illegal_moves += 1
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
    
    # Evaluate against different opponents
    stats_dict = {}
    for opponent in EVALUATION_PARAMS["opponents"]:
        stats = evaluate_policy(
            args.model, 
            n_episodes=args.episodes, 
            render=args.render,
            opponent=opponent
        )
        stats_dict[opponent] = stats
    
    # Plot results
    plot_results(stats_dict)
    
    # Save results
    np.save("evaluation_results.npy", stats_dict)

if __name__ == "__main__":
    main()