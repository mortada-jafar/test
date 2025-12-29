# ============================================================
# Cell 19: Proper Evaluation for Social Learning Agents
# ============================================================

import numpy as np

def evaluate_multi_agent(agents, agent_configs, method="proposed", n_episodes=20, seed=0):
    """
    Evaluate agents with or without social observations based on method

    Args:
        agents: List of trained agents
        agent_configs: List of agent configurations
        method: Training method ("independent", "concat", or "proposed")
        n_episodes: Number of episodes to evaluate
        seed: Random seed
    """
    n_agents = len(agents)

    if method == "independent":
        # For independent method, evaluate each agent alone
        print(f"Evaluating {method} method (agents alone)")
        all_rewards = []

        for agent_id, (agent, config) in enumerate(zip(agents, agent_configs)):
            env = Navigation2DEnv(random_goal=True)
            episode_rewards = []

            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                obs = env.get_partial_observation(config.obs_type)
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 200:
                    action = agent.select_action(obs, social_obs=None, deterministic=True)
                    next_obs_full, reward, terminated, truncated, _ = env.step(action)
                    obs = env.get_partial_observation(config.obs_type)
                    episode_reward += reward
                    done = terminated or truncated
                    steps += 1

                episode_rewards.append(episode_reward)

            all_rewards.append(episode_rewards)
            env.close()

    else:
        # For social learning methods (concat, proposed), evaluate together with social observations
        print(f"Evaluating {method} method (multi-agent with social observations)")
        envs = [Navigation2DEnv(random_goal=True) for _ in range(n_agents)]
        social_channel = SocialObservationChannel(n_agents, state_noise_std=0.0, action_noise_std=0.0)
        all_rewards = [[] for _ in range(n_agents)]

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

                    # Get social observation
                    social_obs = social_channel.get_social_observation(i, add_noise=False)

                    # Select action with social observations
                    if method == "concat":
                        action = agent.select_action(obs, social_obs.flatten(), deterministic=True)
                    else:  # proposed
                        action = agent.select_action(obs, social_obs, deterministic=True)

                    # Take step
                    next_obs_full, reward, terminated, truncated, _ = env.step(action)
                    next_obs = env.get_partial_observation(config.obs_type)

                    # Update social channel
                    social_channel.update(i, next_obs_full, action)

                    episode_rewards[i] += reward
                    observations[i] = next_obs
                    dones[i] = terminated or truncated

                steps += 1

            for i in range(n_agents):
                all_rewards[i].append(episode_rewards[i])

        for env in envs:
            env.close()

    return all_rewards

# Run evaluation
print("\n" + "="*60)
print("Evaluation Results")
print("="*60 + "\n")

# Detect method (same as save cell)
if hasattr(agents[0], 'social_encoder'):
    first_agent_input_dim = agents[0].policy.fc1.in_features
    obs_dim = agent_configs[0].get_obs_dim(Navigation2DEnv())

    if first_agent_input_dim == obs_dim:
        detected_method = "independent"
    elif first_agent_input_dim > obs_dim + 20:
        detected_method = "proposed"
    else:
        detected_method = "concat"
else:
    detected_method = "independent"

print(f"Detected method: {detected_method}\n")

# Evaluate
eval_rewards = evaluate_multi_agent(agents, agent_configs, method=detected_method, n_episodes=20, seed=42)

# Print results
agent_names = [
    'Agent 0 (position_only, novice)',
    'Agent 1 (full, novice)',
    'Agent 2 (full, expert)'
]

print("\n" + "="*60)
print("Results:")
print("="*60)
for i, (name, rewards) in enumerate(zip(agent_names, eval_rewards)):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"\n{name}:")
    print(f"  Mean: {mean_reward:7.2f} ± {std_reward:6.2f}")
    print(f"  Min:  {min_reward:7.2f}")
    print(f"  Max:  {max_reward:7.2f}")

print("\n" + "="*60)
