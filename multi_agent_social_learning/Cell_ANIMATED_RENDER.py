# ============================================================
# Animated Agent Rendering (Shows agent moving in real-time)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

def create_animated_episode(agent, agent_config, seed=42):
    """
    Create animated visualization of a single episode

    Args:
        agent: Trained SAC agent
        agent_config: Agent configuration
        seed: Random seed for reproducibility
    """

    # Create environment and run episode
    env = Navigation2DEnv(random_goal=True, world_size=10.0)
    obs, _ = env.reset(seed=seed)
    obs = env.get_partial_observation(agent_config.obs_type)

    # Collect full trajectory
    states = [env.state.copy()]
    actions_taken = []
    rewards_list = []

    done = False
    steps = 0
    max_steps = 200

    while not done and steps < max_steps:
        action = agent.select_action(obs, deterministic=True)
        next_obs_full, reward, terminated, truncated, info = env.step(action)
        obs = env.get_partial_observation(agent_config.obs_type)

        states.append(env.state.copy())
        actions_taken.append(action)
        rewards_list.append(reward)

        done = terminated or truncated
        steps += 1

    states = np.array(states)
    goal_pos = states[0, 4:6]

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def init():
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        # Draw goal
        goal_circle = patches.Circle(goal_pos, 0.5, color='red', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=25)

        # Draw start position
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=15, label='Start')

        return ax.patches + ax.lines

    def update(frame):
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        # Draw goal
        goal_circle = patches.Circle(goal_pos, 0.5, color='red', alpha=0.3, label='Goal')
        ax.add_patch(goal_circle)
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=25)

        # Draw trajectory so far
        if frame > 0:
            ax.plot(states[:frame+1, 0], states[:frame+1, 1],
                   'b-', linewidth=2, alpha=0.5, label='Path')

        # Draw start
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=12)

        # Draw current position (agent)
        current_pos = states[frame, :2]
        current_vel = states[frame, 2:4]

        # Agent as circle
        agent_circle = patches.Circle(current_pos, 0.3, color='blue', zorder=10)
        ax.add_patch(agent_circle)

        # Velocity arrow
        if np.linalg.norm(current_vel) > 0.1:
            ax.arrow(current_pos[0], current_pos[1],
                    current_vel[0], current_vel[1],
                    head_width=0.3, head_length=0.2,
                    fc='cyan', ec='cyan', zorder=9)

        # Distance to goal
        dist = np.linalg.norm(current_pos - goal_pos)
        total_reward = sum(rewards_list[:frame]) if frame > 0 else 0

        # Title
        ax.set_title(f'Step: {frame}/{len(states)-1} | Distance to Goal: {dist:.2f} | Reward: {total_reward:.1f}',
                    fontsize=13, fontweight='bold')

        if frame == 0:
            ax.legend(loc='upper right', fontsize=10)

        return ax.patches + ax.lines

    anim = FuncAnimation(fig, update, frames=len(states),
                        init_func=init, interval=50, blit=False, repeat=True)

    plt.close()  # Prevent duplicate static plot
    return anim, states, rewards_list

# ============================================================
# Run Animation
# ============================================================

print("="*60)
print("Creating Animated Visualization")
print("="*60)

# Select agent to animate (change index to see different agents)
agent_idx = 2  # 0 = pos only, 1 = novice, 2 = expert
agent_name = ['Agent 0 (Position Only)', 'Agent 1 (Novice)', 'Agent 2 (Expert)'][agent_idx]

print(f"\nAnimating: {agent_name}")
print("This may take a moment...\n")

anim, states, rewards = create_animated_episode(
    agents[agent_idx],
    agent_configs[agent_idx],
    seed=42
)

# Display animation
print(f"Animation created with {len(states)} frames")
print(f"Final reward: {sum(rewards):.2f}")
print(f"Final distance to goal: {np.linalg.norm(states[-1, :2] - states[0, 4:6]):.2f}\n")

# Show the animation
HTML(anim.to_jshtml())
