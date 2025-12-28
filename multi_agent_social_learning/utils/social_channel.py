"""
Social Observation Channel for Multi-Agent Learning

Handles the exchange of noisy state/action information between agents.
"""

import numpy as np


class SocialObservationChannel:
    """
    Manages social observations between agents.

    Each agent can observe noisy state/action pairs from other agents.
    Rewards are NOT shared (privacy constraint).
    """

    def __init__(self, n_agents, state_noise_std=0.1, action_noise_std=0.05):
        """
        Args:
            n_agents: Number of agents
            state_noise_std: Standard deviation of noise added to state observations
            action_noise_std: Standard deviation of noise added to action observations
        """
        self.n_agents = n_agents
        self.state_noise_std = state_noise_std
        self.action_noise_std = action_noise_std

        # Store recent states and actions for each agent
        self.states = [None] * n_agents
        self.actions = [None] * n_agents

    def update(self, agent_id, state, action):
        """
        Update the state and action for a specific agent.

        Args:
            agent_id: ID of the agent (0 to n_agents-1)
            state: Current state of the agent
            action: Last action taken by the agent
        """
        self.states[agent_id] = state
        self.actions[agent_id] = action

    def get_social_observation(self, agent_id, add_noise=True):
        """
        Get social observation for a specific agent.

        Returns noisy state/action pairs from all other agents.

        Args:
            agent_id: ID of the requesting agent
            add_noise: Whether to add noise to observations

        Returns:
            social_obs: Array of shape (n_other_agents, state_dim + action_dim)
                       or None if not enough data available
        """
        social_obs = []

        for i in range(self.n_agents):
            if i == agent_id:
                continue

            if self.states[i] is None or self.actions[i] is None:
                continue

            state = self.states[i].copy()
            action = self.actions[i].copy()

            # Add noise
            if add_noise:
                state_noise = np.random.normal(0, self.state_noise_std, size=state.shape)
                action_noise = np.random.normal(0, self.action_noise_std, size=action.shape)

                state = state + state_noise
                action = action + action_noise
                action = np.clip(action, -1.0, 1.0)  # Keep action in valid range

            # Concatenate state and action
            sa_pair = np.concatenate([state, action])
            social_obs.append(sa_pair)

        if len(social_obs) == 0:
            return None

        return np.array(social_obs, dtype=np.float32)

    def get_flattened_social_observation(self, agent_id, add_noise=True):
        """
        Get flattened social observation (for concat method).

        Returns:
            Flattened array of all other agents' state/action pairs
        """
        social_obs = self.get_social_observation(agent_id, add_noise)

        if social_obs is None:
            return None

        return social_obs.flatten()

    def reset(self):
        """Reset all stored states and actions"""
        self.states = [None] * self.n_agents
        self.actions = [None] * self.n_agents


class AgentConfig:
    """Configuration for a single agent"""

    def __init__(
        self,
        agent_id,
        obs_type="full",
        expertise_level="novice",
        pretrain_steps=0
    ):
        """
        Args:
            agent_id: Unique ID for the agent
            obs_type: Type of observation ("position_only" or "full")
            expertise_level: Level of expertise ("novice" or "expert")
            pretrain_steps: Number of pretraining steps (0 for novice, >0 for expert)
        """
        self.agent_id = agent_id
        self.obs_type = obs_type
        self.expertise_level = expertise_level
        self.pretrain_steps = pretrain_steps

    def get_obs_dim(self, env):
        """Get observation dimension based on observation type"""
        if self.obs_type == "position_only":
            return 4  # [x, y, gx, gy]
        else:  # full
            return 6  # [x, y, vx, vy, gx, gy]

    def __repr__(self):
        return f"Agent{self.agent_id}(obs={self.obs_type}, expertise={self.expertise_level})"


def create_default_agent_configs():
    """
    Create default heterogeneous agent configurations as per project spec.

    Agent 1: position_only observation, novice
    Agent 2: full observation, novice
    Agent 3: full observation, expert (pretrained)
    """
    return [
        AgentConfig(
            agent_id=0,
            obs_type="position_only",
            expertise_level="novice",
            pretrain_steps=0
        ),
        AgentConfig(
            agent_id=1,
            obs_type="full",
            expertise_level="novice",
            pretrain_steps=0
        ),
        AgentConfig(
            agent_id=2,
            obs_type="full",
            expertise_level="expert",
            pretrain_steps=50000  # Will be pretrained
        )
    ]
