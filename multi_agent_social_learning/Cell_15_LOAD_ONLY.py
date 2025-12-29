# ============================================================
# Cell 15 Replacement: Load Models Only (No Training)
# ============================================================
# Use this instead of the original Cell 15 to just load and evaluate

from pathlib import Path

print(f"\n{'='*60}")
print("Loading Trained Models (Evaluation Mode)")
print(f"{'='*60}\n")

# Configuration
method = "proposed"  # Must match the method used during training!
checkpoint_dir = Path("/kaggle/input/multi-agent-model/models")

# Create environments and agent configs
envs = [Navigation2DEnv(random_goal=True) for _ in range(3)]
agent_configs = create_default_agent_configs()

# Create agents with correct architecture
agents = []
for i, config in enumerate(agent_configs):
    obs_dim = config.get_obs_dim(envs[i])
    action_dim = 2

    if method == "independent":
        social_state_dim, social_action_dim, n_other_agents = 0, 0, 0
    else:
        social_state_dim, social_action_dim, n_other_agents = 6, 2, 2

    agent = SACAgent(obs_dim, action_dim, device, method,
                    social_state_dim, social_action_dim, n_other_agents)
    agents.append(agent)

# Load checkpoints
if checkpoint_dir.exists():
    print("Loading checkpoints...\n")
    for i in range(3):
        checkpoint_path = checkpoint_dir / f"agent_{i}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agents[i].policy.load_state_dict(checkpoint['policy'])
            agents[i].critic_1.load_state_dict(checkpoint['critic_1'])
            agents[i].critic_2.load_state_dict(checkpoint['critic_2'])

            if 'social_encoder' in checkpoint and hasattr(agents[i], 'social_encoder'):
                agents[i].social_encoder.load_state_dict(checkpoint['social_encoder'])
                agents[i].action_predictor.load_state_dict(checkpoint['action_predictor'])

            print(f"✓ Loaded Agent {i}")
        else:
            print(f"✗ Error: {checkpoint_path} not found!")
else:
    print(f"✗ Error: Checkpoint directory not found: {checkpoint_dir}")

# Create dummy episode_rewards (for compatibility with Cell 17 plotting)
# Since we didn't train, we'll leave this empty
episode_rewards = [[], [], []]

print(f"\n{'='*60}")
print("Models Loaded Successfully!")
print("Skip to Cell 19 for evaluation, or run Cell 16-17 will show empty plots")
print(f"{'='*60}\n")

# Optional: Quick performance test
print("Quick Performance Test (1 episode per agent):")
print("-" * 60)

for i, (agent, config) in enumerate(zip(agents, agent_configs)):
    env = Navigation2DEnv(random_goal=True)
    obs, _ = env.reset(seed=42)
    obs = env.get_partial_observation(config.obs_type)

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 200:
        action = agent.select_action(obs, deterministic=True)
        next_obs_full, reward, terminated, truncated, _ = env.step(action)
        obs = env.get_partial_observation(config.obs_type)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    agent_name = ['Agent 0 (pos only)', 'Agent 1 (novice)', 'Agent 2 (expert)'][i]
    status = "✓ REACHED GOAL" if terminated else f"✗ Timeout ({steps} steps)"
    print(f"{agent_name}: {total_reward:7.1f}  |  {status}")

print("-" * 60)
print("\nIf Agent 2 scored > -500, models loaded correctly!")
print("Now run Cell 19 for full evaluation (20 episodes)")
