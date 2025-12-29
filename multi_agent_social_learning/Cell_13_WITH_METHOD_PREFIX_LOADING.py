from tqdm.notebook import tqdm
from pathlib import Path

def train_multi_agent(method="proposed", total_steps=100000, n_agents=3,
                     batch_size=256, device="cuda", seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environments and agents
    envs = [Navigation2DEnv(random_goal=True) for _ in range(n_agents)]
    agent_configs = create_default_agent_configs()
    social_channel = SocialObservationChannel(n_agents, 0.1, 0.05)

    agents = []
    buffers = []

    # ==================== STEP 1: CREATE ALL AGENTS ====================
    for i, config in enumerate(agent_configs):
        obs_dim = config.get_obs_dim(envs[i])
        action_dim = 2

        if method == "independent":
            social_state_dim, social_action_dim, n_other_agents = 0, 0, 0
        else:
            social_state_dim, social_action_dim, n_other_agents = 6, 2, n_agents - 1

        agent = SACAgent(obs_dim, action_dim, device, method,
                        social_state_dim, social_action_dim, n_other_agents)
        agents.append(agent)

        if method == "independent":
            buffer = ReplayBuffer(1000000, obs_dim, action_dim, device)
        else:
            social_obs_dim = n_other_agents * (social_state_dim + social_action_dim)
            buffer = SocialReplayBuffer(1000000, obs_dim, action_dim, social_obs_dim, device)
        buffers.append(buffer)

    # ==================== STEP 2: LOAD METHOD-SPECIFIC CHECKPOINTS ====================
    checkpoint_dir = Path("/kaggle/input/multi-agent-model/models")
    load_from_checkpoint = False

    if checkpoint_dir.exists():
        # Look for method-specific checkpoint files
        print(f"\nCheckpoint directory found: {checkpoint_dir}")
        print(f"Current training method: {method}")

        # Try to find checkpoint with method prefix
        method_checkpoint_path = checkpoint_dir / f"{method}_agent_0.pt"

        if method_checkpoint_path.exists():
            print(f"✓ Found method-specific checkpoint: {method}_agent_*.pt")
            load_from_checkpoint = True
            checkpoint_pattern = f"{method}_agent_{{}}.pt"
        else:
            # Fallback: Try loading from generic checkpoint (backward compatibility)
            generic_checkpoint_path = checkpoint_dir / "agent_0.pt"

            if generic_checkpoint_path.exists():
                print(f"Found generic checkpoint: agent_*.pt")
                print("Checking if method matches...")

                try:
                    test_checkpoint = torch.load(generic_checkpoint_path, map_location=device)

                    # Check if checkpoint has method info
                    if 'method' in test_checkpoint:
                        checkpoint_method = test_checkpoint['method']
                    else:
                        # Detect from architecture
                        checkpoint_fc1_shape = test_checkpoint['policy']['fc1.weight'].shape[1]
                        agent_0_obs_dim = agent_configs[0].get_obs_dim(envs[0])

                        if checkpoint_fc1_shape == agent_0_obs_dim:
                            checkpoint_method = "independent"
                        elif checkpoint_fc1_shape == agent_0_obs_dim + 16:
                            checkpoint_method = "concat"
                        else:
                            checkpoint_method = "proposed"

                    print(f"  Detected checkpoint method: {checkpoint_method}")

                    if checkpoint_method == method:
                        load_from_checkpoint = True
                        checkpoint_pattern = "agent_{}.pt"
                        print("✓ Method matches! Loading checkpoints...")
                    else:
                        print(f"✗ Method mismatch!")
                        print(f"  Checkpoint: {checkpoint_method}, Current: {method}")
                        print(f"  → Training from scratch")

                except Exception as e:
                    print(f"⚠ Warning: Could not detect checkpoint method: {e}")
                    print("  → Training from scratch")
            else:
                print(f"No checkpoints found for method '{method}'")
                print("→ Training from scratch")

    if load_from_checkpoint:
        print(f"\n{'='*60}")
        print("Loading saved checkpoints...")
        print(f"{'='*60}\n")

        for i in range(n_agents):
            checkpoint_path = checkpoint_dir / checkpoint_pattern.format(i)

            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agents[i].policy.load_state_dict(checkpoint['policy'])
                agents[i].critic_1.load_state_dict(checkpoint['critic_1'])
                agents[i].critic_2.load_state_dict(checkpoint['critic_2'])
                agents[i].critic_target_1.load_state_dict(checkpoint['critic_1'])
                agents[i].critic_target_2.load_state_dict(checkpoint['critic_2'])

                if 'social_encoder' in checkpoint and hasattr(agents[i], 'social_encoder'):
                    agents[i].social_encoder.load_state_dict(checkpoint['social_encoder'])
                    agents[i].action_predictor.load_state_dict(checkpoint['action_predictor'])

                print(f"✓ Loaded Agent {i} from {checkpoint_path.name}")
            else:
                print(f"⚠ Warning: {checkpoint_path} not found")
        print()
    else:
        # ==================== STEP 3: PRETRAIN IF NO CHECKPOINTS ====================
        print("\nTraining from scratch...")

        for i, config in enumerate(agent_configs):
            if config.pretrain_steps > 0:
                print(f"\nPretraining Agent {i} for {config.pretrain_steps} steps...")
                env = envs[i]
                agent = agents[i]
                buffer = buffers[i]
                obs, _ = env.reset()
                obs = env.get_partial_observation(config.obs_type)

                for step in tqdm(range(config.pretrain_steps), desc=f"Pretrain Agent {i}"):
                    action = env.action_space.sample() if step < 10000 else agent.select_action(obs)
                    next_obs_full, reward, terminated, truncated, _ = env.step(action)
                    next_obs = env.get_partial_observation(config.obs_type)
                    done = terminated or truncated
                    buffer.add(obs, action, reward, next_obs, float(done))
                    obs = next_obs
                    if step >= 10000 and step % 1 == 0:
                        agent.update(buffer.sample(batch_size), step)
                    if done:
                        obs, _ = env.reset()
                        obs = env.get_partial_observation(config.obs_type)

    # ==================== STEP 4: MAIN TRAINING ====================
    observations = [env.get_partial_observation(agent_configs[i].obs_type)
                   for i, (env, _) in enumerate([(e, e.reset()) for e in envs])]
    social_channel.reset()

    episode_rewards = [[] for _ in range(n_agents)]
    total_rewards = [0.0] * n_agents

    pbar = tqdm(total=total_steps, desc=f"Training ({method})")
    for global_step in range(total_steps):
        for i in range(n_agents):
            env = envs[i]
            agent = agents[i]
            buffer = buffers[i]
            config = agent_configs[i]
            obs = observations[i]

            social_obs = None if method == "independent" else social_channel.get_social_observation(i)

            if global_step < 10000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, social_obs.flatten() if method == "concat" and social_obs is not None else social_obs)

            next_obs_full, reward, terminated, truncated, _ = env.step(action)
            next_obs = env.get_partial_observation(config.obs_type)
            done = terminated or truncated

            social_channel.update(i, next_obs_full, action)
            next_social_obs = None if method == "independent" else social_channel.get_social_observation(i)

            if method == "independent":
                buffer.add(obs, action, reward, next_obs, float(done))
            else:
                buffer.add(obs, action, reward, next_obs, float(done),
                          social_obs.flatten(), next_social_obs.flatten())

            total_rewards[i] += reward
            observations[i] = next_obs

            if global_step >= 10000 and global_step % 1 == 0:
                agent.update(buffer.sample(batch_size), global_step)

            if done:
                episode_rewards[i].append(total_rewards[i])
                total_rewards[i] = 0.0
                obs, _ = env.reset()
                observations[i] = env.get_partial_observation(config.obs_type)

        pbar.update(1)
        if global_step % 10000 == 0 and global_step > 0:
            avg_rewards = [np.mean(er[-10:]) if len(er) > 0 else 0 for er in episode_rewards]
            pbar.set_postfix({f"A{i}": f"{avg_rewards[i]:.1f}" for i in range(n_agents)})

    pbar.close()
    return agents, episode_rewards

print("✓ Training function ready")
