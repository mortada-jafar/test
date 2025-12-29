# ============================================================
# Analyze Goal-Reaching Success Rate
# ============================================================

import numpy as np

def evaluate_with_success_tracking(agents, agent_configs, method="proposed", n_episodes=50):
    """
    Evaluate agents and track goal-reaching success

    Goal is reached if final distance < 0.5 (goal_threshold)
    """
    n_agents = len(agents)
    envs = [Navigation2DEnv(random_goal=True, goal_threshold=0.5) for _ in range(n_agents)]
    social_channel = SocialObservationChannel(n_agents, state_noise_std=0.0, action_noise_std=0.0)

    all_rewards = [[] for _ in range(n_agents)]
    all_successes = [[] for _ in range(n_agents)]
    all_final_distances = [[] for _ in range(n_agents)]

    for ep in range(n_episodes):
        # Reset all environments
        observations = []
        for i, (env, config) in enumerate(zip(envs, agent_configs)):
            obs, _ = env.reset(seed=ep)
            observations.append(env.get_partial_observation(config.obs_type))

        social_channel.reset()
        episode_rewards = [0.0] * n_agents
        dones = [False] * n_agents
        terminateds = [False] * n_agents
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

                social_obs = social_channel.get_social_observation(i, add_noise=False)
                action = agent.select_action(obs, social_obs, deterministic=True)

                next_obs_full, reward, terminated, truncated, info = env.step(action)
                next_obs = env.get_partial_observation(config.obs_type)

                social_channel.update(i, next_obs_full, action)

                episode_rewards[i] += reward
                observations[i] = next_obs
                done = terminated or truncated
                dones[i] = done

                if done:
                    terminateds[i] = terminated
                    final_distance = info.get('distance', 999)
                    all_final_distances[i].append(final_distance)

            steps += 1

        # Record results
        for i in range(n_agents):
            all_rewards[i].append(episode_rewards[i])
            all_successes[i].append(terminateds[i])

            # If didn't finish, record final distance
            if not dones[i]:
                current_state = envs[i].state
                goal_pos = current_state[4:6]
                agent_pos = current_state[:2]
                final_distance = np.linalg.norm(agent_pos - goal_pos)
                all_final_distances[i].append(final_distance)

    return all_rewards, all_successes, all_final_distances

# Run detailed evaluation
print("\n" + "="*60)
print("Detailed Performance Analysis (50 episodes)")
print("="*60 + "\n")

rewards, successes, distances = evaluate_with_success_tracking(
    agents, agent_configs, method="proposed", n_episodes=50
)

agent_names = [
    'Agent 0 (position_only, novice)',
    'Agent 1 (full, novice)',
    'Agent 2 (full, expert)'
]

print("="*60)
for i, name in enumerate(agent_names):
    success_rate = np.mean(successes[i]) * 100
    mean_reward = np.mean(rewards[i])
    std_reward = np.std(rewards[i])
    mean_distance = np.mean(distances[i])
    min_distance = np.min(distances[i])

    print(f"\n{name}:")
    print(f"  Success Rate:    {success_rate:5.1f}% ({int(np.sum(successes[i]))}/{len(successes[i])} episodes)")
    print(f"  Mean Reward:     {mean_reward:7.2f} ± {std_reward:6.2f}")
    print(f"  Mean Final Dist: {mean_distance:7.2f}")
    print(f"  Best Distance:   {min_distance:7.2f}")

    if success_rate > 0:
        successful_rewards = [r for r, s in zip(rewards[i], successes[i]) if s]
        print(f"  Reward when successful: {np.mean(successful_rewards):.2f} ± {np.std(successful_rewards):.2f}")

print("\n" + "="*60)
print("Analysis:")
print("="*60)

# Compare agents
expert_success = np.mean(successes[2]) * 100
novice_success = np.mean(successes[1]) * 100

if expert_success > novice_success + 5:
    print(f"✓ Expert significantly better: {expert_success:.1f}% vs {novice_success:.1f}%")
elif expert_success > novice_success:
    print(f"⚠ Expert slightly better: {expert_success:.1f}% vs {novice_success:.1f}%")
else:
    print(f"⚠ Expert not better than novice: {expert_success:.1f}% vs {novice_success:.1f}%")

print("\nRecommendations:")
if expert_success < 50:
    print("  - Current success rate is low (<50%)")
    print("  - For better performance, consider:")
    print("    1. Increase pretraining to 200K-500K steps")
    print("    2. Increase main training to 1M steps")
    print("    3. Tune hyperparameters (learning rate, network size)")
else:
    print("  - Success rate is reasonable (>50%)")
    print("  - This is good performance for a complex continuous control task!")

print("\n" + "="*60)
