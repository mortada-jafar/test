# ============================================================
# Cell 21: Save Models with Method Prefix
# ============================================================

from pathlib import Path
import torch
import json

# Determine which method was used for training
# This should match the method used in the training call
training_method = "proposed"  # Change this to "independent", "concat", or "proposed"

print(f"\n{'='*60}")
print(f"Saving Models (Method: {training_method})")
print(f"{'='*60}\n")

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save each agent with method prefix
for i, agent in enumerate(agents):
    checkpoint = {
        'policy': agent.policy.state_dict(),
        'critic_1': agent.critic_1.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'method': training_method,  # Save method info for loading
    }

    # Add social learning networks if they exist
    if hasattr(agent, 'social_encoder'):
        checkpoint['social_encoder'] = agent.social_encoder.state_dict()
        checkpoint['action_predictor'] = agent.action_predictor.state_dict()

    # Save with method prefix
    filename = f"models/{training_method}_agent_{i}.pt"
    torch.save(checkpoint, filename)
    print(f"✓ Saved {filename}")

# Save results with method prefix
results = {
    'episode_rewards': [list(map(float, er)) for er in episode_rewards],
    'device': device,
    'method': training_method
}

results_filename = f'models/{training_method}_results.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved {results_filename}")
print(f"\n{'='*60}")
print("All models and results saved!")
print(f"{'='*60}")
print(f"\nSaved files:")
print(f"  - {training_method}_agent_0.pt")
print(f"  - {training_method}_agent_1.pt")
print(f"  - {training_method}_agent_2.pt")
print(f"  - {training_method}_results.json")
print(f"\nTo load these models, update checkpoint path to:")
print(f"  checkpoint_dir / '{training_method}_agent_{{i}}.pt'")
