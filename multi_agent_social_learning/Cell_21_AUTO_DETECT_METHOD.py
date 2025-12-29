# ============================================================
# Cell 21: Save Models (Auto-detect Method)
# ============================================================

from pathlib import Path
import torch
import json

print(f"\n{'='*60}")
print("Saving Models")
print(f"{'='*60}\n")

# Auto-detect which method was used by checking agent architecture
# Check if agents have social learning components
if hasattr(agents[0], 'social_encoder'):
    # Check input dimension to distinguish concat from proposed
    first_agent_input_dim = agents[0].policy.fc1.in_features
    obs_dim = agent_configs[0].get_obs_dim(Navigation2DEnv())

    # For Agent 0 (position_only): obs_dim=4
    # - independent: 4
    # - concat: 4 + 16 (2 other agents × 8) = 20
    # - proposed: 4 + 64 (social_embed_dim) = 68

    if first_agent_input_dim == obs_dim:
        training_method = "independent"
    elif first_agent_input_dim > obs_dim + 20:
        training_method = "proposed"
    else:
        training_method = "concat"
else:
    training_method = "independent"

print(f"Auto-detected training method: {training_method}")
print()

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save each agent with method prefix
for i, agent in enumerate(agents):
    checkpoint = {
        'policy': agent.policy.state_dict(),
        'critic_1': agent.critic_1.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'method': training_method,
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
    'device': str(device),
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
