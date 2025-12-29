# ============================================================
# Render Agent Trajectories (Add after loading models)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

def visualize_agent_trajectory(agent, agent_config, n_episodes=4, seed_start=0):
    """
    Visualize agent trajectories in the navigation environment

    Args:
        agent: Trained SAC agent
        agent_config: Agent configuration
        n_episodes: Number of episodes to visualize
        seed_start: Starting seed for episodes
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for ep_idx in range(min(n_episodes, 4)):
        ax = axes[ep_idx]

        # Create environment
        env = Navigation2DEnv(random_goal=True, world_size=10.0)
        obs, _ = env.reset(seed=seed_start + ep_idx)
        obs = env.get_partial_observation(agent_config.obs_type)

        # Collect trajectory
        trajectory = []
        velocities = []
        total_reward = 0
        done = False
        steps = 0
        max_steps = 200

        # Get initial position and goal
        start_pos = env.state[:2].copy()
        goal_pos = env.state[4:6].copy()

        while not done and steps < max_steps:
            trajectory.append(env.state[:2].copy())
            velocities.append(env.state[2:4].copy())

            action = agent.select_action(obs, deterministic=True)
            next_obs_full, reward, terminated, truncated, info = env.step(action)
            obs = env.get_partial_observation(agent_config.obs_type)

            total_reward += reward
            done = terminated or truncated
            steps += 1

        trajectory = np.array(trajectory)

        # Plot
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                'b-', linewidth=2, alpha=0.6, label='Trajectory')

        # Plot start position
        ax.plot(start_pos[0], start_pos[1],
                'go', markersize=15, label='Start', zorder=5)

        # Plot goal
        goal_circle = patches.Circle(goal_pos, 0.5,
                                     color='red', alpha=0.3, label='Goal')
        ax.add_patch(goal_circle)
        ax.plot(goal_pos[0], goal_pos[1],
                'r*', markersize=20, zorder=5)

        # Plot end position
        ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                'bs', markersize=12, label='End', zorder=5)

        # Add direction arrows every 10 steps
        arrow_interval = max(1, len(trajectory) // 10)
        for i in range(0, len(trajectory) - 1, arrow_interval):
            dx = trajectory[i+1, 0] - trajectory[i, 0]
            dy = trajectory[i+1, 1] - trajectory[i, 1]
            ax.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy,
                    head_width=0.3, head_length=0.2,
                    fc='blue', ec='blue', alpha=0.4)

        # Title with metrics
        success = "✓ SUCCESS" if terminated else "✗ TIMEOUT"
        final_dist = np.linalg.norm(trajectory[-1] - goal_pos)
        ax.set_title(f'Episode {ep_idx + 1} | Steps: {steps} | Reward: {total_reward:.1f}\n'
                    f'{success} | Final Distance: {final_dist:.2f}',
                    fontsize=11, fontweight='bold')

        if ep_idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig

# ============================================================
# Visualize All Agents
# ============================================================

print("="*60)
print("Rendering Agent Trajectories")
print("="*60)

agent_names = [
    'Agent 0 (Position Only)',
    'Agent 1 (Full Obs, Novice)',
    'Agent 2 (Full Obs, Expert)'
]

for i, (agent, config, name) in enumerate(zip(agents, agent_configs, agent_names)):
    print(f"\n{name}:")
    fig = visualize_agent_trajectory(agent, config, n_episodes=4, seed_start=i*10)
    plt.savefig(f'agent_{i}_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: agent_{i}_trajectories.png")

print("\n" + "="*60)
print("Visualization Complete!")
print("="*60)
