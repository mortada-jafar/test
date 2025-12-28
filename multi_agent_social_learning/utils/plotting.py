"""
Plotting utilities for analyzing multi-agent social learning results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")


def smooth_curve(data, weight=0.9):
    """Exponential moving average smoothing"""
    smoothed = []
    last = data[0] if len(data) > 0 else 0
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def load_results(results_dir, method, seeds):
    """
    Load results from multiple seeds

    Args:
        results_dir: Base results directory
        method: Method name (independent, concat, proposed)
        seeds: List of seed numbers

    Returns:
        Dictionary with aggregated results
    """
    all_eval_rewards = [[] for _ in range(3)]  # 3 agents
    all_episode_rewards = [[] for _ in range(3)]

    for seed in seeds:
        result_file = Path(results_dir) / method / f"seed_{seed}" / "results.json"
        if not result_file.exists():
            print(f"Warning: {result_file} not found")
            continue

        with open(result_file, 'r') as f:
            data = json.load(f)

        for agent_id in range(3):
            all_eval_rewards[agent_id].append(data['eval_rewards'][agent_id])
            all_episode_rewards[agent_id].append(data['episode_rewards'][agent_id])

    return {
        'eval_rewards': all_eval_rewards,
        'episode_rewards': all_episode_rewards,
        'method': method,
        'seeds': seeds
    }


def compute_regret(rewards, optimal_reward_rate, total_steps):
    """
    Compute regret given rewards and optimal reward rate

    Regret(T) = T * ρ* - Σ r_t

    Args:
        rewards: List of rewards per episode
        optimal_reward_rate: ρ* (optimal average reward rate)
        total_steps: Total number of steps

    Returns:
        Cumulative regret over time
    """
    cumulative_rewards = np.cumsum(rewards)
    timesteps = np.arange(1, len(rewards) + 1)
    optimal_cumulative = optimal_reward_rate * timesteps
    regret = optimal_cumulative - cumulative_rewards
    return regret


def plot_learning_curves(results_dict, agent_id=0, save_path=None):
    """
    Plot learning curves comparing different methods for a specific agent

    Args:
        results_dict: Dict of {method_name: results}
        agent_id: Which agent to plot (0, 1, or 2)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'independent': 'blue', 'concat': 'orange', 'proposed': 'green'}

    for method, results in results_dict.items():
        eval_rewards = np.array(results['eval_rewards'][agent_id])  # (n_seeds, n_evals)

        # Compute mean and std across seeds
        mean_rewards = np.mean(eval_rewards, axis=0)
        std_rewards = np.std(eval_rewards, axis=0)
        n_seeds = eval_rewards.shape[0]

        # 95% confidence interval
        ci = 1.96 * std_rewards / np.sqrt(n_seeds)

        x = np.arange(len(mean_rewards))

        # Plot mean
        ax.plot(x, mean_rewards, label=method.capitalize(),
               color=colors.get(method, 'black'), linewidth=2)

        # Plot confidence interval
        ax.fill_between(x, mean_rewards - ci, mean_rewards + ci,
                        alpha=0.2, color=colors.get(method, 'black'))

    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.set_title(f'Learning Curves - Agent {agent_id}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_regret_curves(results_dict, optimal_rewards, agent_id=0, save_path=None):
    """
    Plot regret curves for different methods

    Args:
        results_dict: Dict of {method_name: results}
        optimal_rewards: Dict of {agent_id: optimal_reward_rate}
        agent_id: Which agent to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'independent': 'blue', 'concat': 'orange', 'proposed': 'green'}
    optimal_rate = optimal_rewards[agent_id]

    for method, results in results_dict.items():
        # Get episode rewards for this agent across all seeds
        episode_rewards_all_seeds = results['episode_rewards'][agent_id]

        regrets = []
        for seed_rewards in episode_rewards_all_seeds:
            regret = compute_regret(seed_rewards, optimal_rate, len(seed_rewards))
            regrets.append(regret)

        # Find minimum length to handle different episode counts
        min_len = min(len(r) for r in regrets)
        regrets = [r[:min_len] for r in regrets]
        regrets = np.array(regrets)

        # Compute mean and confidence interval
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)
        ci = 1.96 * std_regret / np.sqrt(len(regrets))

        x = np.arange(len(mean_regret))

        # Plot
        ax.plot(x, mean_regret, label=method.capitalize(),
               color=colors.get(method, 'black'), linewidth=2)
        ax.fill_between(x, mean_regret - ci, mean_regret + ci,
                        alpha=0.2, color=colors.get(method, 'black'))

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title(f'Regret Curves - Agent {agent_id}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_all_agents_comparison(results_dict, save_path=None):
    """
    Plot comparison of all three agents across methods

    Args:
        results_dict: Dict of {method_name: results}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'independent': 'blue', 'concat': 'orange', 'proposed': 'green'}
    agent_names = ['Agent 0\n(position only, novice)',
                  'Agent 1\n(full obs, novice)',
                  'Agent 2\n(full obs, expert)']

    for agent_id in range(3):
        ax = axes[agent_id]

        for method, results in results_dict.items():
            eval_rewards = np.array(results['eval_rewards'][agent_id])

            mean_rewards = np.mean(eval_rewards, axis=0)
            std_rewards = np.std(eval_rewards, axis=0)
            ci = 1.96 * std_rewards / np.sqrt(eval_rewards.shape[0])

            x = np.arange(len(mean_rewards))

            ax.plot(x, mean_rewards, label=method.capitalize(),
                   color=colors.get(method, 'black'), linewidth=2)
            ax.fill_between(x, mean_rewards - ci, mean_rewards + ci,
                           alpha=0.2, color=colors.get(method, 'black'))

        ax.set_xlabel('Evaluation Step', fontsize=10)
        ax.set_ylabel('Average Return', fontsize=10)
        ax.set_title(agent_names[agent_id], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def create_comparison_table(results_dict, save_path=None):
    """
    Create a table comparing final performance of all methods

    Args:
        results_dict: Dict of {method_name: results}
        save_path: Path to save table
    """
    methods = list(results_dict.keys())
    agent_names = ['Agent 0\n(pos only)', 'Agent 1\n(full, novice)', 'Agent 2\n(full, expert)']

    # Create table data
    table_data = []
    for method in methods:
        results = results_dict[method]
        row = [method.capitalize()]

        for agent_id in range(3):
            eval_rewards = np.array(results['eval_rewards'][agent_id])
            # Take final evaluation across all seeds
            final_rewards = eval_rewards[:, -1]
            mean = np.mean(final_rewards)
            std = np.std(final_rewards)
            row.append(f"{mean:.2f} ± {std:.2f}")

        table_data.append(row)

    # Print table
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    header = ["Method"] + agent_names
    print(f"{header[0]:<15} {header[1]:<20} {header[2]:<20} {header[3]:<20}")
    print("-"*80)
    for row in table_data:
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    print("="*80 + "\n")

    # Save to file
    if save_path:
        with open(save_path, 'w') as f:
            f.write("FINAL PERFORMANCE COMPARISON\n")
            f.write("="*80 + "\n")
            f.write(f"{header[0]:<15} {header[1]:<20} {header[2]:<20} {header[3]:<20}\n")
            f.write("-"*80 + "\n")
            for row in table_data:
                f.write(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}\n")
            f.write("="*80 + "\n")
        print(f"Table saved to {save_path}")


def plot_noise_sensitivity(results_by_noise, agent_id=0, method="proposed", save_path=None):
    """
    Plot sensitivity to noise levels in social observations

    Args:
        results_by_noise: Dict of {noise_std: results_dict}
        agent_id: Which agent to plot
        method: Which method to analyze
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    noise_levels = sorted(results_by_noise.keys())
    final_rewards = []
    final_stds = []

    for noise in noise_levels:
        results = results_by_noise[noise][method]
        eval_rewards = np.array(results['eval_rewards'][agent_id])
        final = eval_rewards[:, -1]
        final_rewards.append(np.mean(final))
        final_stds.append(np.std(final) / np.sqrt(len(final)))

    ax.errorbar(noise_levels, final_rewards, yerr=final_stds,
               marker='o', linewidth=2, markersize=8, capsize=5)

    ax.set_xlabel('Noise Standard Deviation', fontsize=12)
    ax.set_ylabel('Final Average Return', fontsize=12)
    ax.set_title(f'Noise Sensitivity - {method.capitalize()} - Agent {agent_id}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def analyze_results(results_dir, methods, seeds, save_dir='analysis'):
    """
    Comprehensive analysis of results

    Args:
        results_dir: Base directory with results
        methods: List of methods to compare
        seeds: List of seeds to aggregate
        save_dir: Directory to save analysis outputs
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load all results
    print("Loading results...")
    results_dict = {}
    for method in methods:
        results_dict[method] = load_results(results_dir, method, seeds)

    print(f"Loaded results for methods: {list(results_dict.keys())}")

    # Create plots
    print("\nGenerating plots...")

    # Learning curves for each agent
    for agent_id in range(3):
        plot_learning_curves(
            results_dict,
            agent_id=agent_id,
            save_path=save_dir / f"learning_curves_agent_{agent_id}.png"
        )

    # All agents comparison
    plot_all_agents_comparison(
        results_dict,
        save_path=save_dir / "all_agents_comparison.png"
    )

    # Performance table
    create_comparison_table(
        results_dict,
        save_path=save_dir / "performance_table.txt"
    )

    print(f"\nAnalysis complete! Results saved to {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze multi-agent learning results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Base directory with results")
    parser.add_argument("--methods", type=str, nargs='+',
                       default=["independent", "concat", "proposed"],
                       help="Methods to compare")
    parser.add_argument("--seeds", type=int, nargs='+',
                       default=list(range(10)),
                       help="Seeds to aggregate")
    parser.add_argument("--save-dir", type=str, default="analysis",
                       help="Directory to save analysis")

    args = parser.parse_args()

    analyze_results(args.results_dir, args.methods, args.seeds, args.save_dir)
