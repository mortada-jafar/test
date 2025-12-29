# ============================================================
# Cell 23: Train and Compare All Methods (with proper saving)
# ============================================================

from pathlib import Path
import torch
import json

# Train with all three methods for comparison
comparison_results = {}

for method in ['independent', 'concat', 'proposed']:
    print(f"\n{'='*60}")
    print(f"Training with method: {method}")
    print(f"{'='*60}\n")

    agents_m, rewards_m = train_multi_agent(
        method=method,
        total_steps=50000,  # Adjust as needed (50K for quick, 500K for full)
        device=device,
        seed=0
    )

    comparison_results[method] = rewards_m

    # Save each method with prefix
    print(f"\nSaving {method} models...")
    Path("models").mkdir(exist_ok=True)

    for i, agent in enumerate(agents_m):
        checkpoint = {
            'policy': agent.policy.state_dict(),
            'critic_1': agent.critic_1.state_dict(),
            'critic_2': agent.critic_2.state_dict(),
            'method': method,
        }

        if hasattr(agent, 'social_encoder'):
            checkpoint['social_encoder'] = agent.social_encoder.state_dict()
            checkpoint['action_predictor'] = agent.action_predictor.state_dict()

        filename = f"models/{method}_agent_{i}.pt"
        torch.save(checkpoint, filename)
        print(f"  ✓ Saved {filename}")

    # Save results
    results = {
        'episode_rewards': [list(map(float, er)) for er in rewards_m],
        'device': device,
        'method': method
    }
    with open(f'models/{method}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Saved models/{method}_results.json")

print(f"\n{'='*60}")
print("All Methods Trained and Saved!")
print(f"{'='*60}")

# Plot comparison
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
colors = {'independent': 'blue', 'concat': 'orange', 'proposed': 'green'}
agent_names = ['Agent 0 (pos only)', 'Agent 1 (full, novice)', 'Agent 2 (full, expert)']

for agent_id in range(3):
    for method, rewards in comparison_results.items():
        if len(rewards[agent_id]) > 0:
            # Smooth the rewards
            smoothed = []
            window = 10
            for j in range(len(rewards[agent_id])):
                start = max(0, j - window)
                smoothed.append(np.mean(rewards[agent_id][start:j+1]))

            axes[agent_id].plot(smoothed, label=method.capitalize(),
                              color=colors[method], alpha=0.7, linewidth=2)

    axes[agent_id].set_title(agent_names[agent_id], fontsize=12, fontweight='bold')
    axes[agent_id].set_xlabel('Episode')
    axes[agent_id].set_ylabel('Return')
    axes[agent_id].legend()
    axes[agent_id].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Saved method_comparison.png")

# Print final performance summary
print(f"\n{'='*60}")
print("Final Performance Summary")
print(f"{'='*60}\n")

for method in ['independent', 'concat', 'proposed']:
    print(f"{method.upper()}:")
    for agent_id in range(3):
        if len(comparison_results[method][agent_id]) >= 10:
            final_rewards = comparison_results[method][agent_id][-10:]
            mean_reward = np.mean(final_rewards)
            std_reward = np.std(final_rewards)
            print(f"  {agent_names[agent_id]}: {mean_reward:7.2f} ± {std_reward:6.2f}")
    print()
