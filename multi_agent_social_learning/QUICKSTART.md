# 🚀 Quick Start Guide

## Installation (5 minutes)

```bash
cd multi_agent_social_learning

# Install dependencies
pip install torch numpy gymnasium matplotlib seaborn scipy tqdm Pillow

# Verify installation
python test_basic.py
```

## Running Your First Experiment (10 minutes)

### 1. Quick Demo (No Training)
```bash
# Interactive demo of all components
python demo.py
```

This shows:
- ✓ Navigation environment with rendering
- ✓ Heterogeneous agents with different observations
- ✓ Social observation channel with noise
- ✓ Multi-agent visualization

### 2. Quick Training (100K steps, ~10 minutes on CPU)
```bash
# Train with proposed method
python train.py --method proposed --total-steps 100000 --seed 0

# Train with baseline methods for comparison
python train.py --method independent --total-steps 100000 --seed 0
python train.py --method concat --total-steps 100000 --seed 0
```

### 3. Visualize Results
```bash
# Visualize trained agent (with rendering)
python visualize_agent.py \
    --checkpoint results/proposed/seed_0/checkpoint_100000 \
    --agent-id 0 \
    --method proposed \
    --n-episodes 3
```

### 4. Analyze Performance
```bash
# Generate comparison plots
python utils/plotting.py \
    --results-dir results \
    --methods independent concat proposed \
    --seeds 0 \
    --save-dir analysis
```

## Full Experiment (For Course Project)

For statistically significant results, run with 10 seeds:

```bash
# Create a simple run script
cat > run_all.sh << 'EOF'
#!/bin/bash
for seed in {0..9}; do
    echo "Training seed $seed..."
    python train.py --method independent --total-steps 500000 --seed $seed &
    python train.py --method concat --total-steps 500000 --seed $seed &
    python train.py --method proposed --total-steps 500000 --seed $seed &
done
wait
echo "All training complete!"
EOF

chmod +x run_all.sh
./run_all.sh
```

Then analyze:
```bash
python utils/plotting.py \
    --results-dir results \
    --methods independent concat proposed \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --save-dir analysis
```

## Understanding the Output

### Training Output
```
Training (proposed): 100%|██████████| 500000/500000 [1:23:45<00:00]
  Agent0: -45.23, Agent1: -32.15, Agent2: -18.45
```
- Progress bar shows training steps
- Numbers are evaluation rewards (higher is better, less negative = closer to goal)

### Result Files
```
results/
├── proposed/seed_0/
│   ├── results.json          # Full training metrics
│   └── checkpoint_X/
│       ├── agent_0.pt        # Trained model
│       └── metrics.json      # Checkpoint metrics
```

### Analysis Output
```
analysis/
├── learning_curves_agent_0.png    # Learning progress
├── all_agents_comparison.png      # All agents side-by-side
└── performance_table.txt          # Final results table
```

## Expected Results

Based on the project design:

1. **Performance Order**: Proposed > Concat > Independent
   - Proposed method should converge faster
   - Lower cumulative regret
   - Higher final rewards

2. **Agent Differences**:
   - Agent 0 (limited obs) benefits most from social learning
   - Agent 2 (expert) helps others but already performs well
   - Agent 1 (full obs, novice) shows moderate improvement

3. **Typical Rewards** (after 500K steps):
   - Agent 0: -60 to -40
   - Agent 1: -40 to -25
   - Agent 2: -25 to -15

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Training too slow
```bash
# Use fewer steps
python train.py --method proposed --total-steps 50000

# Or use GPU if available
python train.py --method proposed --device cuda
```

### Out of memory
Edit `train.py` and reduce:
- `batch_size=128` (default 256)
- `hidden_dim=128` (default 256)

### No rendering window
```bash
# Use without rendering
python visualize_agent.py --checkpoint <path> --no-render
```

## Next Steps

1. **Customize Agents**: Edit `utils/social_channel.py` → `create_default_agent_configs()`
2. **Change Environment**: Edit `envs/navigation_env.py` parameters
3. **Tune Hyperparameters**: See `agents/sac_agent.py` and `train.py`
4. **Add More Agents**: Set `n_agents=4` in `train.py`

## File Overview

| File | Purpose |
|------|---------|
| `envs/navigation_env.py` | 2D navigation environment |
| `agents/networks.py` | Neural network architectures |
| `agents/sac_agent.py` | SAC with social learning |
| `utils/social_channel.py` | Social observation system |
| `utils/plotting.py` | Analysis and visualization |
| `train.py` | Main training script |
| `visualize_agent.py` | Render trained agents |
| `demo.py` | Interactive demo |

## Tips for Best Results

1. **Always train with multiple seeds** (at least 5-10) for statistical validity
2. **Monitor evaluation rewards** during training - should gradually increase
3. **Check tensorboard logs** if you add logging (recommended for longer runs)
4. **Save checkpoints regularly** - training can take hours for 500K steps
5. **Compare all three methods** to show the value of social learning

## Common Questions

**Q: How long does training take?**
A: ~2-3 hours for 500K steps on CPU, ~30 minutes on GPU

**Q: Can I stop and resume training?**
A: Yes, modify `train.py` to load from checkpoint and continue

**Q: How do I know if it's working?**
A: Evaluation rewards should improve over time. Agent 2 (expert) should perform best.

**Q: What if agents don't learn?**
A: Check that rewards are increasing, reduce learning rate, or increase exploration

**Q: Can I visualize during training?**
A: Yes, set `render_mode="human"` in environment creation (slows training significantly)

## Support

- Check `README.md` for detailed documentation
- Review code comments for implementation details
- Test components individually with `test_basic.py`

---

**Ready to start?** Run `python demo.py` now! 🚀
