# ============================================================
# Cell 17: Visualize Results (Works with both training and loading)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# Check if we have training data or just loaded models
has_training_data = any(len(er) > 0 for er in episode_rewards)

if has_training_data:
    # Original visualization code (when models were trained)
    print("Plotting learning curves from training...\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    agent_names = ['Agent 0 (pos only)', 'Agent 1 (full, novice)', 'Agent 2 (full, expert)']

    for i in range(3):
        if len(episode_rewards[i]) > 0:
            # Smooth rewards
            smoothed = []
            window = 10
            for j in range(len(episode_rewards[i])):
                start = max(0, j - window)
                smoothed.append(np.mean(episode_rewards[i][start:j+1]))

            axes[i].plot(smoothed, alpha=0.8, linewidth=2)
            axes[i].set_title(agent_names[i], fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel('Return')
            axes[i].grid(True, alpha=0.3)

            # Show final performance
            final_avg = np.mean(episode_rewards[i][-10:]) if len(episode_rewards[i]) >= 10 else 0
            axes[i].axhline(y=final_avg, color='r', linestyle='--', alpha=0.5,
                           label=f'Final avg: {final_avg:.1f}')
            axes[i].legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFinal Performance:")
    print("="*50)
    for i in range(3):
        if len(episode_rewards[i]) >= 10:
            final_rewards = episode_rewards[i][-10:]
            print(f"{agent_names[i]}: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print("="*50)

else:
    # Models were loaded, not trained
    print("="*60)
    print("SKIPPING LEARNING CURVES")
    print("="*60)
    print("\nNo training data available (models were loaded from checkpoint).")
    print("Learning curves can only be plotted when training is performed.")
    print("\nOptions:")
    print("1. Run full training to see learning curves")
    print("2. Skip to Cell 19 for evaluation of loaded models")
    print("3. Use visualization cells to see agent trajectories")
    print("="*60)
