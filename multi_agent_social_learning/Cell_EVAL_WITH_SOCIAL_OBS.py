# ============================================================
# Multi-Agent Evaluation WITH Social Observations
# ============================================================

import numpy as np

def evaluate_multi_agent_with_social(agents, agent_configs, n_episodes=20, seed=0):
    """
    Evaluate agents together with social observations (proper for social learning methods)

    Args:
        agents: List of trained agents
        agent_configs: List of agent configurations
        n_episodes: Number of episodes to evaluate
        seed: Random seed
    """
    n_agents = len(agents)
    envs = [Navigation2DEnv(random_goal=True) for _ in range(n_agents)]
    social_channel = SocialObservationChannel(n_agents, state_noise_std=0.0, action_noise_std=0.0)

    all_episode_rewards = [[] for _ in range(n_agents)]

    for ep in range(n_episodes):
        # Reset all environments
        observations = []
        for i, (env, config) in enumerate(zip(envs, agent_configs)):
            obs, _ = env.reset(seed=seed + ep)
            observations.append(env.get_partial_observation(config.obs_type))

        social_channel.reset()
        episode_rewards = [0.0] * n_agents
        dones = [False] * n_agents
        steps = 0
        max_steps = 200

        while not all(dones) and steps < max_steps:
            for i in range(n_agents):
                if dones[i]:
                    continue

                env = envs[i]
                agent = agents[i]
                config = agent_configs[i]
                obs = observations[i]

                # Get social observation (with other agents present!)
                social_obs = social_channel.get_social_observation(i, add_noise=False)

                # Select action (agent uses social observations)
                action = agent.select_action(obs, social_obs, deterministic=True)

                # Take step
                next_obs_full, reward, terminated, truncated, _ = env.step(action)
                next_obs = env.get_partial_observation(config.obs_type)

                # Update social channel (share with others)
                social_channel.update(i, next_obs_full, action)

                episode_rewards[i] += reward
                observations[i] = next_obs
                dones[i] = terminated or truncated

            steps += 1

        for i in range(n_agents):
            all_episode_rewards[i].append(episode_rewards[i])

    return all_episode_rewards

# Run proper evaluation
print("\n" + "="*60)
print("Multi-Agent Evaluation (WITH Social Observations)")
print("="*60 + "\n")

eval_rewards = evaluate_multi_agent_with_social(agents, agent_configs, n_episodes=20, seed=42)

agent_names = [
    'Agent 0 (position_only, novice)',
    'Agent 1 (full, novice)',
    'Agent 2 (full, expert)'
]

print("Results:")
print("-" * 60)
for i, (name, rewards) in enumerate(zip(agent_names, eval_rewards)):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"{name}:")
    print(f"  Mean: {mean_reward:7.2f} ± {std_reward:6.2f}")
    print(f"  Min:  {min_reward:7.2f}")
    print(f"  Max:  {max_reward:7.2f}")
    print()

print("="*60)
print("\nComparison with Training Performance:")
print("-" * 60)
print("Training vs Evaluation:")
for i in range(3):
    training_avg = np.mean(episode_rewards[i][-10:]) if len(episode_rewards[i]) >= 10 else 0
    eval_avg = np.mean(eval_rewards[i])
    diff = eval_avg - training_avg

    print(f"{agent_names[i]}:")
    print(f"  Training:   {training_avg:7.2f}")
    print(f"  Evaluation: {eval_avg:7.2f} ({diff:+.2f})")
    print()

print("="*60)
print("\nNote: This evaluation includes social observations,")
print("which is the correct way to evaluate social learning agents!")
print("="*60)
