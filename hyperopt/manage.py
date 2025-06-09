"""
Utility script for managing hyperparameter optimization results.

This script provides utilities for:
- Finding and listing available optimization results
- Comparing different optimization runs
- Setting environment variables for using specific results
- Converting results between formats
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from hyperopt.config_loader import find_latest_hyperopt_results


def list_optimization_results(results_dir: str = "hyperopt_results") -> List[Dict[str, Any]]:
    """
    List all available hyperparameter optimization results.
    
    Args:
        results_dir: Directory containing optimization results
        
    Returns:
        List of result summaries
    """
    if not os.path.exists(results_dir):
        return []
    
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("hyperopt_results_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                results.append({
                    "filename": filename,
                    "filepath": filepath,
                    "study_name": data.get("study_name", "Unknown"),
                    "best_value": data.get("best_value", None),
                    "n_trials": data.get("n_trials", 0),
                    "timestamp": data.get("timestamp", "Unknown"),
                    "modified_time": datetime.fromtimestamp(os.path.getmtime(filepath))
                })
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified_time"], reverse=True)
    return results


def display_results_summary(results: List[Dict[str, Any]]):
    """Display a formatted summary of optimization results."""
    if not results:
        print("No hyperparameter optimization results found.")
        return
    
    print(f"Found {len(results)} optimization result(s):")
    print()
    print(f"{'#':<3} {'Study Name':<20} {'Best Value':<12} {'Trials':<8} {'Date':<12} {'Filename'}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        date_str = result["modified_time"].strftime("%Y-%m-%d")
        print(f"{i+1:<3} {result['study_name']:<20} {result['best_value']:<12.4f} "
              f"{result['n_trials']:<8} {date_str:<12} {result['filename']}")


def show_detailed_results(filepath: str):
    """Show detailed information about a specific optimization result."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    print(f"Hyperparameter Optimization Results")
    print(f"=" * 50)
    print(f"Study Name: {data.get('study_name', 'Unknown')}")
    print(f"Number of Trials: {data.get('n_trials', 0)}")
    print(f"Best Value: {data.get('best_value', 'N/A')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    print()
    
    print("Best Parameters:")
    print("-" * 20)
    best_params = data.get("best_params", {})
    for key, value in best_params.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    print()
    print("Optimization Configuration:")
    print("-" * 30)
    opt_config = data.get("optimization_config", {})
    for key, value in opt_config.items():
        print(f"  {key}: {value}")
    
    # Show trial history summary
    trial_history = data.get("trial_history", [])
    if trial_history:
        print()
        print("Trial History Summary:")
        print("-" * 25)
        completed_trials = [t for t in trial_history if t.get("value") is not None]
        if completed_trials:
            values = [t["value"] for t in completed_trials]
            print(f"  Completed trials: {len(completed_trials)}")
            print(f"  Best value: {max(values):.4f}")
            print(f"  Average value: {sum(values)/len(values):.4f}")
            print(f"  Worst value: {min(values):.4f}")


def set_environment_variable(filepath: str):
    """Display command to set environment variable for using specific results."""
    abs_path = os.path.abspath(filepath)
    
    print(f"To use these hyperparameters in training, set the environment variable:")
    print()
    print(f"Windows (PowerShell):")
    print(f'$env:ANTICHESS_HYPEROPT_PATH="{abs_path}"')
    print()
    print(f"Windows (CMD):")
    print(f'set ANTICHESS_HYPEROPT_PATH="{abs_path}"')
    print()
    print(f"Linux/macOS:")
    print(f'export ANTICHESS_HYPEROPT_PATH="{abs_path}"')
    print()
    print(f"Then run your training script normally:")
    print(f"python -m train.train_ppo --use-enhanced-curriculum")


def compare_results(filepaths: List[str]):
    """Compare multiple optimization results."""
    if len(filepaths) < 2:
        print("Need at least 2 result files to compare.")
        return
    
    results_data = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results_data.append((os.path.basename(filepath), data))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return
    
    print("Comparison of Optimization Results")
    print("=" * 50)
    
    # Compare best values
    print("\nBest Values:")
    for filename, data in results_data:
        best_value = data.get("best_value", "N/A")
        n_trials = data.get("n_trials", 0)
        print(f"  {filename}: {best_value} ({n_trials} trials)")
    
    # Compare key hyperparameters
    print("\nKey Hyperparameters:")
    key_params = ["learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "ent_coef"]
    
    for param in key_params:
        print(f"\n  {param}:")
        for filename, data in results_data:
            best_params = data.get("best_params", {})
            if param in best_params:
                value = best_params[param]
                if isinstance(value, dict):
                    # Handle complex parameters like learning_rate
                    if "initial_value" in value and "final_value" in value:
                        print(f"    {filename}: {value['initial_value']} -> {value['final_value']}")
                    else:
                        print(f"    {filename}: {value}")
                else:
                    print(f"    {filename}: {value}")
            else:
                print(f"    {filename}: N/A")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage hyperparameter optimization results"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available optimization results")
    list_parser.add_argument(
        "--results-dir", 
        type=str, 
        default="hyperopt_results",
        help="Directory containing optimization results"
    )
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed results")
    show_parser.add_argument(
        "filepath", 
        type=str, 
        nargs="?",
        help="Path to results file (uses latest if not specified)"
    )
    show_parser.add_argument(
        "--results-dir", 
        type=str, 
        default="hyperopt_results",
        help="Directory to search for results if filepath not specified"
    )
    
    # Use command
    use_parser = subparsers.add_parser("use", help="Set environment variable for using results")
    use_parser.add_argument(
        "filepath", 
        type=str, 
        nargs="?",
        help="Path to results file (uses latest if not specified)"
    )
    use_parser.add_argument(
        "--results-dir", 
        type=str, 
        default="hyperopt_results",
        help="Directory to search for results if filepath not specified"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple optimization results")
    compare_parser.add_argument(
        "filepaths", 
        type=str, 
        nargs="+",
        help="Paths to results files to compare"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "list":
        results = list_optimization_results(args.results_dir)
        display_results_summary(results)
    
    elif args.command == "show":
        filepath = args.filepath
        if not filepath:
            # Use latest results
            filepath = find_latest_hyperopt_results(args.results_dir)
            if not filepath:
                print(f"No optimization results found in {args.results_dir}")
                return
            print(f"Using latest results: {filepath}")
            print()
        
        if not os.path.exists(filepath):
            print(f"Results file not found: {filepath}")
            return
        
        show_detailed_results(filepath)
    
    elif args.command == "use":
        filepath = args.filepath
        if not filepath:
            # Use latest results
            filepath = find_latest_hyperopt_results(args.results_dir)
            if not filepath:
                print(f"No optimization results found in {args.results_dir}")
                return
            print(f"Using latest results: {filepath}")
            print()
        
        if not os.path.exists(filepath):
            print(f"Results file not found: {filepath}")
            return
        
        set_environment_variable(filepath)
    
    elif args.command == "compare":
        # Validate all files exist
        for filepath in args.filepaths:
            if not os.path.exists(filepath):
                print(f"Results file not found: {filepath}")
                return
        
        compare_results(args.filepaths)
    
    else:
        print("Please specify a command. Use --help for available commands.")
        print("Available commands: list, show, use, compare")


if __name__ == "__main__":
    main()
