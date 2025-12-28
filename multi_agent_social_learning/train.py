"""
Multi-Agent Social Learning Training Script

Trains multiple heterogeneous agents using different methods:
    - Independent: No social learning
    - Concat: Direct concatenation of social observations
    - Proposed: Embedding network with auxiliary loss
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.navigation_env import Navigation2DEnv
from agents.sac_agent import SACAgent, ReplayBuffer, SocialReplayBuffer
from utils.social_channel import SocialObservationChannel, create_default_agent_configs


class MultiAgentTrainer:
    """Trainer for multiple agents with social learning"""

    def __init__(
        self,
        method="independent",
        n_agents=3,
        agent_configs=None,
        total_steps=500000,
        batch_size=256,
        buffer_size=1000000,
        start_steps=10000,
        update_interval=1,
        eval_interval=10000,
        eval_episodes=10,
        state_noise_std=0.1,
        action_noise_std=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        save_dir="results"
    ):
        self.method = method
        self.n_agents = n_agents
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.device = device
        self.seed = seed
        self.save_dir = Path(save_dir) / method / f"seed_{seed}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create agent configurations
        if agent_configs is None:
            self.agent_configs = create_default_agent_configs()
        else:
            self.agent_configs = agent_configs

        # Create environments for each agent
        self.envs = []
        for i in range(n_agents):
            env = Navigation2DEnv(
                dt=0.1,
                v_max=2.0,
                world_size=10.0,
                goal_threshold=0.5,
                max_steps=200,
                random_goal=True
            )
            env.reset(seed=seed + i)
            self.envs.append(env)

        # Social observation channel
        self.social_channel = SocialObservationChannel(
            n_agents, state_noise_std, action_noise_std
        )

        # Create agents
        self.agents = []
        self.buffers = []

        for i, config in enumerate(self.agent_configs):
            obs_dim = config.get_obs_dim(self.envs[i])
            action_dim = self.envs[i].action_space.shape[0]

            # Determine social observation dimensions
            if method == "independent":
                social_state_dim = 0
                social_action_dim = 0
                n_other_agents = 0
            else:
                # Social obs includes state and action from other agents
                # We need to know the maximum state dim among all agents for social obs
                social_state_dim = 6  # Full state dimension
                social_action_dim = action_dim
                n_other_agents = n_agents - 1

            # Create agent
            agent = SACAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device,
                method=method,
                social_state_dim=social_state_dim,
                social_action_dim=social_action_dim,
                n_other_agents=n_other_agents,
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                auto_entropy_tuning=True,
                hidden_dim=256,
                social_embed_dim=64,
                aux_loss_weight=0.1
            )

            self.agents.append(agent)

            # Create replay buffer
            if method == "independent":
                buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, device)
            else:
                # Calculate social obs dimension
                if method == "concat":
                    social_obs_dim = n_other_agents * (social_state_dim + social_action_dim)
                else:  # proposed - stored as structured array
                    social_obs_dim = n_other_agents * (social_state_dim + social_action_dim)

                buffer = SocialReplayBuffer(
                    buffer_size, obs_dim, action_dim, social_obs_dim, device
                )

            self.buffers.append(buffer)

        # Pretrain expert agents
        self._pretrain_experts()

        # Tracking metrics
        self.episode_rewards = [[] for _ in range(n_agents)]
        self.episode_lengths = [[] for _ in range(n_agents)]
        self.eval_rewards = [[] for _ in range(n_agents)]
        self.total_rewards = [0.0 for _ in range(n_agents)]
        self.episode_steps = [0 for _ in range(n_agents)]

    def _pretrain_experts(self):
        """Pretrain expert agents"""
        for i, config in enumerate(self.agent_configs):
            if config.pretrain_steps > 0:
                print(f"\nPretraining Agent {i} (Expert) for {config.pretrain_steps} steps...")
                self._train_single_agent(i, config.pretrain_steps, is_pretrain=True)
                print(f"Agent {i} pretraining complete!")

    def _train_single_agent(self, agent_id, steps, is_pretrain=False):
        """Train a single agent independently"""
        env = self.envs[agent_id]
        agent = self.agents[agent_id]
        buffer = self.buffers[agent_id]
        config = self.agent_configs[agent_id]

        obs, _ = env.reset()
        obs = env.get_partial_observation(config.obs_type)
        episode_reward = 0
        episode_step = 0

        for step in tqdm(range(steps), desc=f"Pretraining Agent {agent_id}"):
            # Random actions for initial exploration
            if step < self.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs_full, reward, terminated, truncated, _ = env.step(action)
            next_obs = env.get_partial_observation(config.obs_type)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_reward += reward
            episode_step += 1

            # Update agent
            if step >= self.start_steps and step % self.update_interval == 0:
                batch = buffer.sample(self.batch_size)
                agent.update(batch, step)

            if done:
                obs, _ = env.reset()
                obs = env.get_partial_observation(config.obs_type)
                episode_reward = 0
                episode_step = 0

    def train(self):
        """Main training loop for all agents"""
        # Reset all environments
        observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            obs = env.get_partial_observation(self.agent_configs[i].obs_type)
            observations.append(obs)

        # Reset social channel
        self.social_channel.reset()

        global_step = 0
        pbar = tqdm(total=self.total_steps, desc=f"Training ({self.method})")

        while global_step < self.total_steps:
            # Each agent takes a step
            for i in range(self.n_agents):
                env = self.envs[i]
                agent = self.agents[i]
                buffer = self.buffers[i]
                config = self.agent_configs[i]
                obs = observations[i]

                # Get social observation
                social_obs = None
                if self.method != "independent":
                    social_obs = self.social_channel.get_social_observation(i, add_noise=True)

                # Select action
                if global_step < self.start_steps:
                    action = env.action_space.sample()
                else:
                    if self.method == "concat" and social_obs is not None:
                        action = agent.select_action(obs, social_obs.flatten())
                    else:
                        action = agent.select_action(obs, social_obs)

                # Take step in environment
                next_obs_full, reward, terminated, truncated, _ = env.step(action)
                next_obs = env.get_partial_observation(config.obs_type)
                done = terminated or truncated

                # Update social channel with full state (for other agents to observe)
                self.social_channel.update(i, next_obs_full, action)

                # Get next social observation
                next_social_obs = None
                if self.method != "independent":
                    next_social_obs = self.social_channel.get_social_observation(i, add_noise=True)

                # Store transition
                if self.method == "independent":
                    buffer.add(obs, action, reward, next_obs, float(done))
                else:
                    # Flatten or structure social obs for storage
                    social_flat = social_obs.flatten() if social_obs is not None else np.zeros(buffer.social_obs.shape[1])
                    next_social_flat = next_social_obs.flatten() if next_social_obs is not None else np.zeros(buffer.social_obs.shape[1])
                    buffer.add(obs, action, reward, next_obs, float(done), social_flat, next_social_flat)

                # Track metrics
                self.total_rewards[i] += reward
                self.episode_steps[i] += 1

                # Update observation
                observations[i] = next_obs

                # Update agent
                if global_step >= self.start_steps and global_step % self.update_interval == 0:
                    batch = buffer.sample(self.batch_size)
                    agent.update(batch, global_step)

                # Episode end
                if done:
                    self.episode_rewards[i].append(self.total_rewards[i])
                    self.episode_lengths[i].append(self.episode_steps[i])

                    # Reset
                    obs, _ = env.reset()
                    observations[i] = env.get_partial_observation(config.obs_type)
                    self.total_rewards[i] = 0.0
                    self.episode_steps[i] = 0

            global_step += 1
            pbar.update(1)

            # Evaluation
            if global_step % self.eval_interval == 0:
                eval_results = self.evaluate()
                self._save_checkpoint(global_step, eval_results)
                pbar.set_postfix({f"Agent{i}": f"{eval_results[i]:.2f}" for i in range(self.n_agents)})

        pbar.close()
        self._save_results()

    def evaluate(self):
        """Evaluate all agents"""
        eval_rewards = []

        for i in range(self.n_agents):
            agent = self.agents[i]
            config = self.agent_configs[i]
            env = self.envs[i]

            episode_rewards = []
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                obs = env.get_partial_observation(config.obs_type)
                episode_reward = 0
                done = False

                while not done:
                    # No social obs during evaluation (or use available ones)
                    action = agent.select_action(obs, deterministic=True)
                    next_obs_full, reward, terminated, truncated, _ = env.step(action)
                    obs = env.get_partial_observation(config.obs_type)
                    episode_reward += reward
                    done = terminated or truncated

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)
            eval_rewards.append(mean_reward)
            self.eval_rewards[i].append(mean_reward)

        return eval_rewards

    def _save_checkpoint(self, step, eval_results):
        """Save checkpoint"""
        checkpoint_dir = self.save_dir / f"checkpoint_{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent.save(checkpoint_dir / f"agent_{i}.pt")

        # Save metrics
        metrics = {
            'step': step,
            'eval_rewards': eval_results,
            'episode_rewards': [list(r) for r in self.episode_rewards],
            'eval_history': [list(r) for r in self.eval_rewards]
        }

        with open(checkpoint_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    def _save_results(self):
        """Save final results"""
        results = {
            'method': self.method,
            'seed': self.seed,
            'total_steps': self.total_steps,
            'episode_rewards': [list(r) for r in self.episode_rewards],
            'episode_lengths': [list(r) for r in self.episode_lengths],
            'eval_rewards': [list(r) for r in self.eval_rewards],
            'agent_configs': [
                {
                    'agent_id': c.agent_id,
                    'obs_type': c.obs_type,
                    'expertise_level': c.expertise_level,
                    'pretrain_steps': c.pretrain_steps
                }
                for c in self.agent_configs
            ]
        }

        with open(self.save_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {self.save_dir}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Social Learning")
    parser.add_argument("--method", type=str, default="independent",
                       choices=["independent", "concat", "proposed"],
                       help="Learning method")
    parser.add_argument("--total-steps", type=int, default=500000,
                       help="Total training steps")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save results")

    args = parser.parse_args()

    trainer = MultiAgentTrainer(
        method=args.method,
        total_steps=args.total_steps,
        seed=args.seed,
        save_dir=args.save_dir
    )

    trainer.train()


if __name__ == "__main__":
    main()
