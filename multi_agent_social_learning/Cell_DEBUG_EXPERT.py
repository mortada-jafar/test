# ============================================================
# Diagnostic: Why isn't the expert agent reaching the goal?
# ============================================================

import torch
import numpy as np

print("="*60)
print("DEBUGGING EXPERT AGENT PERFORMANCE")
print("="*60)

# Test 1: Compare random vs loaded policy
print("\n1. Testing Random Policy vs Loaded Policy")
print("-" * 60)

agent_idx = 2  # Expert agent
config = agent_configs[agent_idx]

# Test with loaded agent
env = Navigation2DEnv(random_goal=True)
obs, _ = env.reset(seed=42)
obs = env.get_partial_observation(config.obs_type)

total_reward_loaded = 0
steps = 0
done = False

while not done and steps < 200:
    action = agents[agent_idx].select_action(obs, deterministic=True)
    next_obs_full, reward, terminated, truncated, _ = env.step(action)
    obs = env.get_partial_observation(config.obs_type)
    total_reward_loaded += reward
    done = terminated or truncated
    steps += 1

print(f"Loaded Agent Performance: {total_reward_loaded:.2f} in {steps} steps")
print(f"Reached Goal: {'✓ YES' if terminated else '✗ NO'}")

# Test with random policy
env.reset(seed=42)
obs = env.get_partial_observation(config.obs_type)
total_reward_random = 0
done = False
steps_random = 0

while not done and steps_random < 200:
    action = env.action_space.sample()  # Random actions
    next_obs_full, reward, terminated, truncated, _ = env.step(action)
    obs = env.get_partial_observation(config.obs_type)
    total_reward_random += reward
    done = terminated or truncated
    steps_random += 1

print(f"Random Policy Performance: {total_reward_random:.2f} in {steps_random} steps")
print(f"\nImprovement: {total_reward_loaded - total_reward_random:.2f}")

if total_reward_loaded < -1500:
    print("⚠ WARNING: Loaded agent performing poorly (like random policy)!")
    print("   Possible issues:")
    print("   - Checkpoint not loaded correctly")
    print("   - Wrong method specified")
    print("   - Architecture mismatch")

# Test 2: Check if weights are actually different from initialization
print("\n2. Checking if Weights Were Actually Loaded")
print("-" * 60)

# Create a fresh agent with random weights
obs_dim = config.get_obs_dim(env)
fresh_agent = SACAgent(obs_dim, 2, device, "proposed", 6, 2, 2)

# Compare some weights
loaded_weights = agents[agent_idx].policy.fc1.weight.data.cpu().numpy()
fresh_weights = fresh_agent.policy.fc1.weight.data.cpu().numpy()

weight_diff = np.abs(loaded_weights - fresh_weights).mean()
print(f"Average weight difference: {weight_diff:.6f}")

if weight_diff < 0.01:
    print("⚠ WARNING: Weights are nearly identical to random initialization!")
    print("   The checkpoint may not have loaded correctly.")
else:
    print("✓ Weights are different from random initialization (checkpoint loaded)")

# Test 3: Check agent configuration
print("\n3. Checking Agent Configuration")
print("-" * 60)
print(f"Agent ID: {config.agent_id}")
print(f"Observation Type: {config.obs_type}")
print(f"Observation Dim: {obs_dim}")
print(f"Expertise Level: {config.expertise_level}")
print(f"Pretrain Steps: {config.pretrain_steps}")

if config.obs_type != "full":
    print("⚠ WARNING: Expert agent should have 'full' observations!")

# Test 4: Multiple episodes with different seeds
print("\n4. Testing Multiple Episodes")
print("-" * 60)

rewards_list = []
successes = []

for seed in range(10):
    env = Navigation2DEnv(random_goal=True)
    obs, _ = env.reset(seed=seed)
    obs = env.get_partial_observation(config.obs_type)

    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 200:
        action = agents[agent_idx].select_action(obs, deterministic=True)
        next_obs_full, reward, terminated, truncated, _ = env.step(action)
        obs = env.get_partial_observation(config.obs_type)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    rewards_list.append(total_reward)
    successes.append(terminated)
    print(f"Episode {seed}: {total_reward:7.2f} | Steps: {steps:3d} | {'✓ Goal' if terminated else '✗ Timeout'}")

avg_reward = np.mean(rewards_list)
success_rate = np.mean(successes) * 100

print(f"\nAverage Reward: {avg_reward:.2f}")
print(f"Success Rate: {success_rate:.0f}%")

# Test 5: Check what actions the agent is taking
print("\n5. Sample Actions from Agent")
print("-" * 60)

env = Navigation2DEnv(random_goal=True)
obs, _ = env.reset(seed=42)
obs = env.get_partial_observation(config.obs_type)

print("First 10 actions:")
for i in range(10):
    action = agents[agent_idx].select_action(obs, deterministic=True)
    print(f"  Step {i}: [{action[0]:6.3f}, {action[1]:6.3f}]")
    next_obs_full, reward, terminated, truncated, _ = env.step(action)
    obs = env.get_partial_observation(config.obs_type)

    if np.all(action == 0.0):
        print("⚠ WARNING: Agent is outputting zero actions!")
        break
    if terminated or truncated:
        break

# Final Diagnosis
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if avg_reward > -500:
    print("✓ Agent is performing well! Expert agent is working correctly.")
elif avg_reward > -1000:
    print("⚠ Moderate performance. Agent has learned something but not optimal.")
    print("  Possible reasons:")
    print("  - Training was insufficient")
    print("  - Checkpoints from early in training")
else:
    print("✗ Poor performance. Agent is NOT working correctly.")
    print("  MOST LIKELY ISSUES:")
    print("  1. Checkpoint didn't load (check Cell 15 output)")
    print("  2. Wrong method specified (should be 'proposed')")
    print("  3. Checkpoint files are from untrained/early training")
    print("\n  SOLUTION:")
    print("  - Verify checkpoint path is correct")
    print("  - Check that method='proposed' in load script")
    print("  - Ensure checkpoint files are from completed training run")

print("="*60)
