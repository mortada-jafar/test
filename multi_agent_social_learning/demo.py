"""
Quick demo script to test the navigation environment and social learning system
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.navigation_env import Navigation2DEnv
from utils.social_channel import SocialObservationChannel, create_default_agent_configs


def demo_environment():
    """Demonstrate the navigation environment with rendering"""
    print("="*60)
    print("DEMO: 2D Navigation Environment")
    print("="*60)

    env = Navigation2DEnv(
        dt=0.1,
        v_max=2.0,
        world_size=10.0,
        goal_threshold=0.5,
        max_steps=200,
        random_goal=True,
        render_mode="human"
    )

    obs, _ = env.reset(seed=42)
    print(f"\nInitial state: {obs}")
    print(f"State format: [x, y, vx, vy, gx, gy]")
    print(f"Action format: [ax, ay] in [-1, 1]^2")

    episode_reward = 0
    step = 0
    done = False

    print("\nRunning random policy for one episode...")
    print("Watch the agent (blue circle) try to reach the goal (green area)")

    while not done and step < 50:  # Limit to 50 steps for demo
        # Random action
        action = env.action_space.sample()

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        step += 1

        # Render
        env.render()

        if done:
            print(f"\nEpisode finished!")
            print(f"  Steps: {step}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Final distance to goal: {info['distance']:.2f}")
            print(f"  Success: {info['distance'] < env.goal_threshold}")

    env.close()
    print("\nEnvironment demo complete!")


def demo_heterogeneous_observations():
    """Demonstrate different observation types for heterogeneous agents"""
    print("\n" + "="*60)
    print("DEMO: Heterogeneous Agent Observations")
    print("="*60)

    env = Navigation2DEnv()
    obs, _ = env.reset(seed=42)

    agent_configs = create_default_agent_configs()

    print(f"\nFull state: {obs}")
    print(f"Format: [x, y, vx, vy, gx, gy]")

    for config in agent_configs:
        partial_obs = env.get_partial_observation(config.obs_type)
        print(f"\n{config}:")
        print(f"  Observation: {partial_obs}")
        print(f"  Dimension: {len(partial_obs)}")

    env.close()


def demo_social_channel():
    """Demonstrate social observation channel"""
    print("\n" + "="*60)
    print("DEMO: Social Observation Channel")
    print("="*60)

    n_agents = 3
    social_channel = SocialObservationChannel(
        n_agents=n_agents,
        state_noise_std=0.1,
        action_noise_std=0.05
    )

    # Simulate agents' states and actions
    print("\nSimulating 3 agents with different states and actions:")

    states = [
        np.array([1.0, 2.0, 0.5, 0.3, 5.0, 5.0]),  # Agent 0
        np.array([-1.0, 3.0, -0.2, 0.1, 5.0, 5.0]),  # Agent 1
        np.array([2.0, -1.0, 0.0, 0.5, 5.0, 5.0])  # Agent 2
    ]

    actions = [
        np.array([0.5, 0.3]),  # Agent 0
        np.array([-0.2, 0.8]),  # Agent 1
        np.array([0.1, -0.4])  # Agent 2
    ]

    # Update social channel
    for i in range(n_agents):
        social_channel.update(i, states[i], actions[i])
        print(f"\nAgent {i} state: {states[i][:2]}  action: {actions[i]}")

    # Get social observations
    print("\n" + "-"*60)
    print("Social observations (what each agent sees about others):")

    for i in range(n_agents):
        social_obs = social_channel.get_social_observation(i, add_noise=True)
        print(f"\nAgent {i} observes {n_agents - 1} other agents:")
        if social_obs is not None:
            for j, obs in enumerate(social_obs):
                print(f"  Other agent {j}: state+action (with noise) = {obs[:4]}...")

    # Show effect of noise
    print("\n" + "-"*60)
    print("Noise effect (comparing clean vs noisy observations):")

    social_obs_clean = social_channel.get_social_observation(0, add_noise=False)
    social_obs_noisy = social_channel.get_social_observation(0, add_noise=True)

    print("\nAgent 0 observing Agent 1 (first two state dimensions):")
    print(f"  Clean:  {social_obs_clean[0][:2]}")
    print(f"  Noisy:  {social_obs_noisy[0][:2]}")
    print(f"  True state: {states[1][:2]}")


def demo_multi_agent_navigation():
    """Demonstrate multiple agents navigating simultaneously"""
    print("\n" + "="*60)
    print("DEMO: Multi-Agent Navigation (Visualization)")
    print("="*60)

    n_agents = 3
    envs = []

    # Create environments
    for i in range(n_agents):
        env = Navigation2DEnv(
            dt=0.1,
            v_max=2.0,
            world_size=10.0,
            goal_threshold=0.5,
            max_steps=100,
            random_goal=True
        )
        env.reset(seed=i)
        envs.append(env)

    # Collect trajectories
    trajectories = [[] for _ in range(n_agents)]
    goals = []

    for i, env in enumerate(envs):
        trajectories[i].append([env.state[0], env.state[1]])
        goals.append([env.state[4], env.state[5]])

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            trajectories[i].append([env.state[0], env.state[1]])

            if terminated or truncated:
                break

    # Plot all trajectories
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['blue', 'red', 'green']
    agent_names = ['Agent 0 (pos only)', 'Agent 1 (full, novice)', 'Agent 2 (full, expert)']

    for i in range(n_agents):
        traj = np.array(trajectories[i])
        goal = goals[i]

        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.6,
               linewidth=2, label=agent_names[i])

        # Start point
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, marker='o')

        # End point
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=100, marker='X')

        # Goal
        circle = plt.Circle(goal, 0.5, color=colors[i], alpha=0.2)
        ax.add_patch(circle)
        ax.scatter(goal[0], goal[1], color=colors[i], s=150, marker='*')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Multi-Agent Navigation (Random Policies)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('multi_agent_demo.png', dpi=150)
    print("\nVisualization saved to: multi_agent_demo.png")
    plt.show()

    # Clean up
    for env in envs:
        env.close()


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("MULTI-AGENT SOCIAL LEARNING - DEMO")
    print("="*60)
    print("\nThis demo will showcase:")
    print("1. The 2D navigation environment")
    print("2. Heterogeneous agent observations")
    print("3. Social observation channel")
    print("4. Multi-agent navigation visualization")
    print("\n" + "="*60)

    input("\nPress Enter to start Demo 1: Environment...")
    demo_environment()

    input("\nPress Enter to start Demo 2: Heterogeneous Observations...")
    demo_heterogeneous_observations()

    input("\nPress Enter to start Demo 3: Social Channel...")
    demo_social_channel()

    input("\nPress Enter to start Demo 4: Multi-Agent Visualization...")
    demo_multi_agent_navigation()

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train agents: python train.py --method proposed --total-steps 100000")
    print("2. Visualize: python visualize_agent.py --checkpoint results/proposed/seed_0/checkpoint_100000")
    print("3. Analyze: python utils/plotting.py --results-dir results")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
