"""
Visualization script for trained agents with rendering

This script loads a trained agent and visualizes its behavior in the environment.
"""

import os
import sys
import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.navigation_env import Navigation2DEnv
from agents.sac_agent import SACAgent
from utils.social_channel import create_default_agent_configs


def visualize_agent(
    checkpoint_path,
    agent_id=0,
    method="independent",
    n_episodes=5,
    deterministic=True,
    render=True,
    save_video=False
):
    """
    Visualize a trained agent

    Args:
        checkpoint_path: Path to checkpoint directory
        agent_id: Which agent to visualize
        method: Learning method used
        n_episodes: Number of episodes to run
        deterministic: Use deterministic policy
        render: Whether to render the environment
        save_video: Whether to save video frames
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    render_mode = "human" if render else None
    env = Navigation2DEnv(
        dt=0.1,
        v_max=2.0,
        world_size=10.0,
        goal_threshold=0.5,
        max_steps=200,
        random_goal=True,
        render_mode=render_mode
    )

    # Get agent config
    agent_configs = create_default_agent_configs()
    config = agent_configs[agent_id]

    obs_dim = config.get_obs_dim(env)
    action_dim = env.action_space.shape[0]

    # Create agent
    n_agents = len(agent_configs)
    if method == "independent":
        social_state_dim = 0
        social_action_dim = 0
        n_other_agents = 0
    else:
        social_state_dim = 6
        social_action_dim = action_dim
        n_other_agents = n_agents - 1

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        method=method,
        social_state_dim=social_state_dim,
        social_action_dim=social_action_dim,
        n_other_agents=n_other_agents
    )

    # Load checkpoint
    checkpoint_file = Path(checkpoint_path) / f"agent_{agent_id}.pt"
    if not checkpoint_file.exists():
        print(f"Checkpoint not found: {checkpoint_file}")
        return

    agent.load(checkpoint_file)
    print(f"Loaded agent from {checkpoint_file}")

    # Run episodes
    episode_rewards = []
    episode_lengths = []
    frames = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        obs = env.get_partial_observation(config.obs_type)
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}/{n_episodes}")

        while not done:
            # Select action (no social obs for visualization)
            action = agent.select_action(obs, deterministic=deterministic)

            # Step
            next_obs_full, reward, terminated, truncated, info = env.step(action)
            obs = env.get_partial_observation(config.obs_type)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render
            if render:
                frame = env.render()
                if save_video and frame is not None:
                    frames.append(frame)
                time.sleep(0.03)  # Slow down for visualization

            if done:
                print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}, "
                      f"Final distance: {info['distance']:.2f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Agent {agent_id} ({config.obs_type}, {config.expertise_level})")
    print(f"Method: {method}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"{'='*50}")

    # Save video if requested
    if save_video and frames:
        save_path = Path(checkpoint_path).parent / f"agent_{agent_id}_video.gif"
        save_frames_as_gif(frames, save_path)
        print(f"Video saved to {save_path}")


def save_frames_as_gif(frames, path, duration=50):
    """Save frames as animated GIF"""
    try:
        from PIL import Image
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
    except ImportError:
        print("PIL not available. Install with: pip install Pillow")


def visualize_multi_agent_comparison(checkpoint_dirs, agent_id=0, n_episodes=10):
    """
    Compare multiple methods for the same agent side-by-side

    Args:
        checkpoint_dirs: Dict of {method_name: checkpoint_path}
        agent_id: Which agent to compare
        n_episodes: Number of episodes for evaluation
    """
    fig, axes = plt.subplots(1, len(checkpoint_dirs), figsize=(6 * len(checkpoint_dirs), 6))
    if len(checkpoint_dirs) == 1:
        axes = [axes]

    for idx, (method, checkpoint_path) in enumerate(checkpoint_dirs.items()):
        print(f"\nEvaluating {method}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create environment
        env = Navigation2DEnv(
            dt=0.1,
            v_max=2.0,
            world_size=10.0,
            goal_threshold=0.5,
            max_steps=200,
            random_goal=True
        )

        # Get agent config
        agent_configs = create_default_agent_configs()
        config = agent_configs[agent_id]

        obs_dim = config.get_obs_dim(env)
        action_dim = env.action_space.shape[0]

        # Create agent
        n_agents = len(agent_configs)
        if method == "independent":
            social_state_dim = 0
            social_action_dim = 0
            n_other_agents = 0
        else:
            social_state_dim = 6
            social_action_dim = action_dim
            n_other_agents = n_agents - 1

        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            method=method,
            social_state_dim=social_state_dim,
            social_action_dim=social_action_dim,
            n_other_agents=n_other_agents
        )

        # Load checkpoint
        checkpoint_file = Path(checkpoint_path) / f"agent_{agent_id}.pt"
        agent.load(checkpoint_file)

        # Evaluate and collect trajectory
        obs, _ = env.reset(seed=42)
        obs = env.get_partial_observation(config.obs_type)
        done = False
        trajectory = []

        while not done:
            trajectory.append([env.state[0], env.state[1]])
            action = agent.select_action(obs, deterministic=True)
            next_obs_full, reward, terminated, truncated, info = env.step(action)
            obs = env.get_partial_observation(config.obs_type)
            done = terminated or truncated

        trajectory = np.array(trajectory)

        # Plot trajectory
        ax = axes[idx]
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{method.capitalize()}")

        # Goal
        goal_x, goal_y = env.state[4], env.state[5]
        circle = plt.Circle((goal_x, goal_y), 0.5, color='green', alpha=0.3)
        ax.add_patch(circle)

        # Trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='blue', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='X', label='End')
        ax.scatter(goal_x, goal_y, c='green', s=100, marker='*', label='Goal')

        ax.legend()
        env.close()

    plt.tight_layout()
    plt.savefig("agent_comparison.png", dpi=150)
    print(f"\nComparison saved to agent_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize trained agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--agent-id", type=int, default=0,
                       help="Agent ID to visualize (0, 1, or 2)")
    parser.add_argument("--method", type=str, default="independent",
                       choices=["independent", "concat", "proposed"],
                       help="Learning method")
    parser.add_argument("--n-episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering")
    parser.add_argument("--save-video", action="store_true",
                       help="Save video as GIF")

    args = parser.parse_args()

    visualize_agent(
        checkpoint_path=args.checkpoint,
        agent_id=args.agent_id,
        method=args.method,
        n_episodes=args.n_episodes,
        deterministic=True,
        render=not args.no_render,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
