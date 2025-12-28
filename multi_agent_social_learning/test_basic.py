"""
Basic test script to verify all components work correctly
"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test the navigation environment"""
    print("Testing Navigation Environment...")
    from envs.navigation_env import Navigation2DEnv

    env = Navigation2DEnv()
    obs, _ = env.reset(seed=42)

    assert env.observation_space.shape == (6,), "Observation space should be 6D"
    assert env.action_space.shape == (2,), "Action space should be 2D"
    assert len(obs) == 6, "Initial observation should be 6D"

    # Take a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert len(obs) == 6
        assert isinstance(reward, (int, float))

    # Test partial observations
    partial_obs = env.get_partial_observation("position_only")
    assert len(partial_obs) == 4, "Position-only obs should be 4D"

    full_obs = env.get_partial_observation("full")
    assert len(full_obs) == 6, "Full obs should be 6D"

    env.close()
    print("✓ Environment test passed!")


def test_social_channel():
    """Test the social observation channel"""
    print("\nTesting Social Observation Channel...")
    from utils.social_channel import SocialObservationChannel, create_default_agent_configs

    n_agents = 3
    channel = SocialObservationChannel(n_agents, state_noise_std=0.1, action_noise_std=0.05)

    # Update with dummy data
    for i in range(n_agents):
        state = np.random.randn(6).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        channel.update(i, state, action)

    # Get social observations
    for i in range(n_agents):
        social_obs = channel.get_social_observation(i, add_noise=True)
        assert social_obs is not None
        assert social_obs.shape[0] == n_agents - 1, "Should observe n-1 other agents"
        assert social_obs.shape[1] == 8, "Each observation should be 6 (state) + 2 (action)"

    # Test agent configs
    configs = create_default_agent_configs()
    assert len(configs) == 3, "Should have 3 agent configs"
    assert configs[0].obs_type == "position_only"
    assert configs[1].obs_type == "full"
    assert configs[2].expertise_level == "expert"

    print("✓ Social channel test passed!")


def test_networks():
    """Test neural network architectures"""
    print("\nTesting Neural Networks...")
    from agents.networks import (
        QNetwork, GaussianPolicy,
        SocialEmbeddingNetwork, ActionPredictionHead
    )

    device = "cpu"
    batch_size = 32
    obs_dim = 6
    action_dim = 2
    social_embed_dim = 64

    # Test Q-Network
    q_net = QNetwork(obs_dim, action_dim, hidden_dim=256).to(device)
    obs = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)
    q_value = q_net(obs, action)
    assert q_value.shape == (batch_size, 1)

    # Test Policy
    policy = GaussianPolicy(obs_dim, action_dim, hidden_dim=256).to(device)
    action, log_prob, mean = policy.sample(obs)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)

    # Test Social Embedding Network
    social_net = SocialEmbeddingNetwork(6, 2, embed_dim=64).to(device)
    social_obs = torch.randn(batch_size, 2, 8)  # 2 other agents, 8 features each
    embedding = social_net(social_obs)
    assert embedding.shape == (batch_size, social_embed_dim)

    # Test Action Prediction Head
    action_pred = ActionPredictionHead(social_embed_dim, action_dim).to(device)
    predicted_action = action_pred(embedding)
    assert predicted_action.shape == (batch_size, action_dim)

    print("✓ Network test passed!")


def test_sac_agent():
    """Test SAC agent creation and basic operations"""
    print("\nTesting SAC Agent...")
    from agents.sac_agent import SACAgent, ReplayBuffer, SocialReplayBuffer

    device = "cpu"
    obs_dim = 6
    action_dim = 2

    # Test independent agent
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        method="independent"
    )

    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.select_action(obs, deterministic=False)
    assert action.shape == (action_dim,)

    # Test replay buffer
    buffer = ReplayBuffer(1000, obs_dim, action_dim, device)
    for _ in range(100):
        buffer.add(
            np.random.randn(obs_dim),
            np.random.randn(action_dim),
            np.random.randn(),
            np.random.randn(obs_dim),
            0.0
        )

    batch = buffer.sample(32)
    assert batch['obs'].shape == (32, obs_dim)

    # Test update
    if buffer.size >= 32:
        losses = agent.update(batch, step=0)
        assert 'policy_loss' in losses
        assert 'critic_1_loss' in losses

    # Test proposed method agent
    agent_proposed = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        method="proposed",
        social_state_dim=6,
        social_action_dim=2,
        n_other_agents=2
    )

    social_obs = np.random.randn(2, 8).astype(np.float32)
    action = agent_proposed.select_action(obs, social_obs, deterministic=True)
    assert action.shape == (action_dim,)

    print("✓ SAC agent test passed!")


def test_integration():
    """Integration test: brief training loop"""
    print("\nTesting Integration (Mini Training Loop)...")
    from envs.navigation_env import Navigation2DEnv
    from agents.sac_agent import SACAgent, ReplayBuffer
    from utils.social_channel import AgentConfig

    device = "cpu"
    env = Navigation2DEnv()
    config = AgentConfig(0, obs_type="full", expertise_level="novice")

    obs_dim = config.get_obs_dim(env)
    action_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, action_dim, device, method="independent")
    buffer = ReplayBuffer(10000, obs_dim, action_dim, device)

    obs, _ = env.reset(seed=42)
    obs = env.get_partial_observation(config.obs_type)

    total_reward = 0
    steps = 0

    # Run for 100 steps
    for step in range(100):
        if step < 50:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs_full, reward, terminated, truncated, _ = env.step(action)
        next_obs = env.get_partial_observation(config.obs_type)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        total_reward += reward
        steps += 1

        # Update agent
        if buffer.size >= 32 and step % 10 == 0:
            batch = buffer.sample(32)
            losses = agent.update(batch, step)

        if done:
            obs, _ = env.reset()
            obs = env.get_partial_observation(config.obs_type)
            total_reward = 0

    env.close()
    print(f"✓ Integration test passed! Completed {steps} steps")


def main():
    """Run all tests"""
    print("="*60)
    print("RUNNING BASIC TESTS")
    print("="*60)

    try:
        test_environment()
        test_social_channel()
        test_networks()
        test_sac_agent()
        test_integration()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe system is ready to use!")
        print("\nNext steps:")
        print("1. Run demo: python demo.py")
        print("2. Train agents: python train.py --method proposed --total-steps 10000")
        print("3. Visualize: python visualize_agent.py --checkpoint <path>")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
