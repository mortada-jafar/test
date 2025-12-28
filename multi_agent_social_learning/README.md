# Multi-Agent Social Learning

**تسریع یادگیری عامل مستقل با استفاده از تخمین رفتار سایر عامل های ناهمگن در چارچوب یادگیری تعاملی اجتماعی**

Accelerating Independent Agent Learning through Behavioral Estimation of Other Heterogeneous Agents in Social Interactive Learning Framework

---

## 📋 Overview

This project implements a multi-agent reinforcement learning system where heterogeneous agents learn to navigate in independent environments while sharing noisy state/action observations (but NOT rewards) through a social observation channel. The goal is to demonstrate that agents can accelerate their learning by observing others, even without access to their rewards.

### Key Features

- **2D Continuous Navigation Environment**: Agents navigate to goal positions using continuous control
- **Heterogeneous Agents**: Three agents with different observation capabilities and expertise levels
- **Social Learning Framework**: Agents observe noisy state/action pairs from other agents
- **Three Learning Methods**:
  - **Independent**: No social learning (baseline)
  - **Concat**: Direct concatenation of social observations
  - **Proposed**: Embedding network with auxiliary action prediction loss
- **Comprehensive Evaluation**: Regret analysis, learning curves, statistical comparisons

---

## 🏗️ Project Structure

```
multi_agent_social_learning/
├── envs/
│   └── navigation_env.py          # 2D navigation environment
├── agents/
│   ├── networks.py                # Neural network architectures
│   └── sac_agent.py               # SAC agent with social learning
├── utils/
│   ├── social_channel.py          # Social observation channel
│   └── plotting.py                # Analysis and plotting utilities
├── train.py                       # Main training script
├── visualize_agent.py             # Visualization script with rendering
├── demo.py                        # Quick demo of all components
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd multi_agent_social_learning

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Run interactive demo to see all components
python demo.py
```

This will demonstrate:
1. The navigation environment with rendering
2. Heterogeneous agent observations
3. Social observation channel with noise
4. Multi-agent navigation visualization

### Train Agents

Train agents using different methods:

```bash
# Independent learning (baseline)
python train.py --method independent --total-steps 500000 --seed 0

# Concatenation method
python train.py --method concat --total-steps 500000 --seed 0

# Proposed method (with embedding network)
python train.py --method proposed --total-steps 500000 --seed 0
```

For full statistical analysis, train with multiple seeds:

```bash
# Example: Train with 10 seeds
for seed in {0..9}; do
    python train.py --method independent --total-steps 500000 --seed $seed &
    python train.py --method concat --total-steps 500000 --seed $seed &
    python train.py --method proposed --total-steps 500000 --seed $seed &
done
wait
```

### Visualize Trained Agents

```bash
# Visualize a trained agent with rendering
python visualize_agent.py \
    --checkpoint results/proposed/seed_0/checkpoint_500000 \
    --agent-id 0 \
    --method proposed \
    --n-episodes 5
```

Options:
- `--agent-id`: Which agent to visualize (0, 1, or 2)
- `--method`: Learning method used (independent, concat, proposed)
- `--save-video`: Save visualization as GIF

### Analyze Results

```bash
# Generate comprehensive analysis plots
python utils/plotting.py \
    --results-dir results \
    --methods independent concat proposed \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --save-dir analysis
```

This generates:
- Learning curves for each agent
- Comparison across all methods
- Performance tables with mean ± std
- Regret analysis (optional)

---

## 🤖 Agent Configuration

The system uses three heterogeneous agents as specified in the project proposal:

| Agent | Observation Type | Expertise | Description |
|-------|-----------------|-----------|-------------|
| **Agent 0** | Position only (`[x, y, gx, gy]`) | Novice | Limited observation, learns from scratch |
| **Agent 1** | Full state (`[x, y, vx, vy, gx, gy]`) | Novice | Complete observation, learns from scratch |
| **Agent 2** | Full state (`[x, y, vx, vy, gx, gy]`) | Expert | Pre-trained for 50k steps |

---

## 🧪 Environment Details

### State Space
```
s_t = [x_t, y_t, v_x_t, v_y_t, g_x, g_y]
```
- `(x, y)`: Agent position
- `(v_x, v_y)`: Agent velocity
- `(g_x, g_y)`: Goal position

### Action Space
```
a_t = [a_x_t, a_y_t] ∈ [-1, 1]²
```
Continuous acceleration in x and y directions.

### Dynamics
```
v_{t+1} = clip(v_t + Δt · a_t, -v_max, v_max)
p_{t+1} = p_t + Δt · v_{t+1}
```

### Reward
```
r_t = -||p_t - g||_2
```
Negative distance to goal (shared across all agents).

---

## 📊 Social Observation Channel

Each agent receives noisy observations of other agents:

```
social_obs_{i,t} = {(ŝ_{j,t} + ε^s, â_{j,t} + ε^a)} for j ≠ i

ε^s ~ N(0, σ_s²)
ε^a ~ N(0, σ_a²)
```

**Key constraints**:
- ✅ State and action observations shared
- ❌ Rewards are NOT shared (privacy)
- ❌ Agents don't affect each other's environments

---

## 🎯 Methods Comparison

### 1. Independent Learner
- **Input**: Own observation only
- **Architecture**: Standard SAC
- **Purpose**: Baseline without social learning

### 2. Concat Baseline
- **Input**: Own observation + flattened social observations
- **Architecture**: SAC with larger input dimension
- **Purpose**: Simple social learning baseline

### 3. Proposed Method
- **Input**: Own observation + embedded social observations
- **Architecture**: SAC + Social Embedding Network + Action Prediction Head
- **Loss**: L = L_SAC + λ · L_aux
- **Auxiliary Loss**: Predicts actions of other agents
- **Purpose**: Learned social representation without using rewards

---

## 📈 Evaluation Metrics

### Average Reward
```
ρ_i(π) = (1/T) Σ r_{i,t}
```

### Regret
```
Regret_i(T) = T · ρ_i* - Σ r_{i,t}
```
where `ρ_i*` is the optimal reward rate under observation constraints.

### Statistical Analysis
- Mean ± Standard Deviation across seeds
- 95% Confidence Intervals
- Convergence speed analysis

---

## 🔬 Expected Results

Based on the project proposal, we expect:

1. **Proposed > Concat > Independent** in terms of:
   - Lower cumulative regret
   - Faster convergence
   - Higher final performance

2. **Agent 0 (novice, limited obs)** benefits most from social learning

3. **Agent 2 (expert)** helps others but may not benefit much

4. **Robustness to noise**: Proposed method should be more stable with increasing noise

---

## 🛠️ Customization

### Change Environment Parameters

Edit in `train.py` or `envs/navigation_env.py`:
```python
env = Navigation2DEnv(
    dt=0.1,              # Time step
    v_max=2.0,          # Maximum velocity
    world_size=10.0,    # World boundaries
    goal_threshold=0.5, # Success threshold
    max_steps=200       # Episode length
)
```

### Modify Agent Configuration

Edit `utils/social_channel.py` → `create_default_agent_configs()`:
```python
AgentConfig(
    agent_id=0,
    obs_type="position_only",  # or "full"
    expertise_level="novice",   # or "expert"
    pretrain_steps=0            # 0 for novice, >0 for expert
)
```

### Adjust Social Noise

In `train.py`:
```python
social_channel = SocialObservationChannel(
    n_agents=3,
    state_noise_std=0.1,   # Increase for more noise
    action_noise_std=0.05
)
```

### Tune Hyperparameters

In `agents/sac_agent.py` or `train.py`:
```python
agent = SACAgent(
    lr=3e-4,                    # Learning rate
    gamma=0.99,                 # Discount factor
    tau=0.005,                  # Target network update rate
    alpha=0.2,                  # Entropy coefficient
    hidden_dim=256,             # Network hidden size
    social_embed_dim=64,        # Social embedding size
    aux_loss_weight=0.1         # Auxiliary loss weight
)
```

---

## 📁 Results Structure

After training, results are saved in:
```
results/
├── independent/
│   ├── seed_0/
│   │   ├── results.json
│   │   └── checkpoint_X/
│   │       ├── agent_0.pt
│   │       ├── agent_1.pt
│   │       ├── agent_2.pt
│   │       └── metrics.json
│   └── seed_1/
│       └── ...
├── concat/
│   └── ...
└── proposed/
    └── ...
```

---

## 🎨 Rendering Features

The environment supports:
- **Real-time visualization** during training
- **Trajectory plotting** for analysis
- **Multi-agent comparison** side-by-side
- **Video export** as GIF (requires Pillow)

Example with rendering:
```python
env = Navigation2DEnv(render_mode="human")
obs, _ = env.reset()

for _ in range(100):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()  # Display visualization
```

---

## 🧮 Mathematical Framework

### MDP Definition
- **States**: S = ℝ⁶ (continuous)
- **Actions**: A = [-1, 1]² (continuous)
- **Transition**: p(s'|s,a) deterministic with noise
- **Reward**: r(s,a) = -||p - g||₂
- **Policy**: π_θ: S × Z → Δ(A) (stochastic Gaussian)

### Social Learning
- **Social Observation**: o^social_i = {(s_j, a_j) + noise}_{j≠i}
- **Embedding**: z_i = g_φ(o^social_i)
- **Policy**: π_θ(a | s_i, z_i)

### Auxiliary Loss
```
L_aux = 𝔼[(â_j - ã_j)²]
```
where â_j is predicted action, ã_j is observed action.

---

## 🔍 Ablation Studies

To run ablation studies, modify training parameters:

### Noise Sensitivity
```bash
for noise in 0.0 0.05 0.1 0.2 0.5; do
    # Edit train.py to set state_noise_std = $noise
    python train.py --method proposed --seed 0
done
```

### Number of Agents
Edit `train.py` to use 2, 3, or 4 agents:
```python
trainer = MultiAgentTrainer(n_agents=4, ...)
```

### Expert vs No Expert
Remove expert agent from configuration or set `pretrain_steps=0`.

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `train.py`
- Reduce `hidden_dim` in agent configuration
- Use CPU: Set `device="cpu"`

### Slow Training
- Decrease `total_steps`
- Reduce `eval_interval`
- Use fewer agents

### Rendering Issues
- Check matplotlib backend: `export MPLBACKEND=TkAgg`
- For headless servers, use `render_mode=None`

---

## 📚 References

This project implements concepts from:
- **Soft Actor-Critic (SAC)**: Haarnoja et al., 2018
- **Multi-Agent RL**: Survey by Zhang et al., 2021
- **Social Learning**: Inspired by collaborative learning literature

---

## 📝 Citation

If you use this code for your project, please cite:

```
Multi-Agent Social Learning: Accelerating Independent Agent Learning
through Behavioral Estimation of Heterogeneous Agents
Reinforcement Learning Course Project, 2024
```

---

## 🤝 Contributing

This is a course project. For questions or improvements:
1. Check existing issues
2. Test changes thoroughly
3. Maintain code structure and documentation style

---

## 📧 Contact

For questions about this implementation, please refer to the code comments and this README.

---

## ⚖️ License

This project is for educational purposes. Feel free to use and modify for learning.

---

**Happy Learning! 🎓🤖**
