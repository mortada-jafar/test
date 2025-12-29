# Guide: Resume Training from Saved Models on Kaggle

## Step 1: Save Models as Kaggle Dataset

After training completes, download these files from your Kaggle notebook output:
```
models/
├── agent_0.pt
├── agent_1.pt
├── agent_2.pt
└── results.json
```

Upload them as a **Kaggle Dataset** (e.g., "my-trained-agents").

---

## Step 2: Add Dataset to Your Notebook

In Kaggle notebook:
1. Click **"+ Add Data"** (right sidebar)
2. Search for your dataset (e.g., "my-trained-agents")
3. Click **"Add"**

Your models will be available at: `/kaggle/input/my-trained-agents/`

---

## Step 3: Modify Training Code

### **OPTION A: Load and Continue Training**

**Find this section in Cell 13 (around line 35-50 in the training function):**

```python
# Pretrain expert
for i, config in enumerate(agent_configs):
    if config.pretrain_steps > 0:
        print(f"Pretraining Agent {i} for {config.pretrain_steps} steps...")
        # ... pretraining code ...
```

**Replace with this to load checkpoints:**

```python
# ==================== MODIFIED: LOAD CHECKPOINTS ====================
from pathlib import Path

checkpoint_dir = Path("/kaggle/input/my-trained-agents")  # Change to your dataset name
load_checkpoints = checkpoint_dir.exists()  # Auto-detect if checkpoints available

if load_checkpoints:
    print(f"\n{'='*60}")
    print("Loading saved checkpoints...")
    print(f"{'='*60}\n")

    for i in range(n_agents):
        checkpoint_path = checkpoint_dir / f"agent_{i}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agents[i].policy.load_state_dict(checkpoint['policy'])
            agents[i].critic_1.load_state_dict(checkpoint['critic_1'])
            agents[i].critic_2.load_state_dict(checkpoint['critic_2'])
            # Update target networks
            agents[i].critic_target_1.load_state_dict(checkpoint['critic_1'])
            agents[i].critic_target_2.load_state_dict(checkpoint['critic_2'])

            # Load social learning networks if they exist
            if 'social_encoder' in checkpoint and hasattr(agents[i], 'social_encoder'):
                agents[i].social_encoder.load_state_dict(checkpoint['social_encoder'])
                agents[i].action_predictor.load_state_dict(checkpoint['action_predictor'])

            print(f"✓ Loaded Agent {i} from checkpoint")
    print()
else:
    # Original pretraining code (if no checkpoints found)
    print("No checkpoints found. Training from scratch...")
    for i, config in enumerate(agent_configs):
        if config.pretrain_steps > 0:
            print(f"Pretraining Agent {i} for {config.pretrain_steps} steps...")
            # ... keep original pretraining code ...
# ==================== END MODIFICATION ====================
```

---

### **OPTION B: Just Load Models (No Additional Training)**

**Add this new cell right before Cell 19 (Evaluation):**

```python
# ==================== LOAD TRAINED MODELS ====================
from pathlib import Path

checkpoint_dir = Path("/kaggle/input/my-trained-agents")  # Your dataset name

if checkpoint_dir.exists():
    print("Loading trained models from dataset...")

    # Create agents first (same as in training)
    envs = [Navigation2DEnv(random_goal=True) for _ in range(3)]
    agent_configs = create_default_agent_configs()
    method = "proposed"  # Must match the method used during training!

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

        # Load checkpoint
        checkpoint = torch.load(checkpoint_dir / f"agent_{i}.pt", map_location=device)
        agent.policy.load_state_dict(checkpoint['policy'])
        agent.critic_1.load_state_dict(checkpoint['critic_1'])
        agent.critic_2.load_state_dict(checkpoint['critic_2'])

        if 'social_encoder' in checkpoint and hasattr(agent, 'social_encoder'):
            agent.social_encoder.load_state_dict(checkpoint['social_encoder'])
            agent.action_predictor.load_state_dict(checkpoint['action_predictor'])

        agents.append(agent)
        print(f"✓ Loaded Agent {i}")

    print("\n✓ All models loaded! Ready for evaluation.")
else:
    print("ERROR: Checkpoint directory not found!")
# ==================== END LOAD MODELS ====================
```

---

## Summary of Changes

### To **Continue Training**:
Modify **Cell 13** (training function) as shown in Option A

### To **Just Evaluate** Pretrained Models:
Add new cell before **Cell 19** as shown in Option B

### Key Variables to Match:
```python
checkpoint_dir = "/kaggle/input/YOUR-DATASET-NAME"  # ← Change this!
method = "proposed"  # ← Must match training method!
```

---

## Quick Reference: Line Numbers to Edit

**Cell 13 - Training Function:**
- **Lines 35-50**: Pretrain expert section → Add checkpoint loading logic

**Cell 15 - Run Training:**
- **Line 5**: Modify `total_steps` to continue training (e.g., add another 100000 steps)

**Cell 21 - Save Models:**
- Already correct - no changes needed

---

## Example: Full Resume Training Workflow

```python
# 1. In Cell 13, add checkpoint loading (see Option A above)

# 2. In Cell 15, run with additional steps:
agents, episode_rewards = train_multi_agent(
    method="proposed",
    total_steps=100000,  # Will continue from loaded checkpoint
    device=device,
    seed=0  # Use same seed for consistency
)

# 3. Models will continue training from where they left off!
```
