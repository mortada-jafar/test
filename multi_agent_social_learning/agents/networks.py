"""
Neural Network architectures for SAC and Social Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def weights_init_(m):
    """Initialize network weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    """Critic Network (Q-function)"""

    def __init__(self, obs_dim, action_dim, hidden_dim=256, social_embed_dim=0):
        super(QNetwork, self).__init__()

        input_dim = obs_dim + action_dim + social_embed_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, obs, action, social_embed=None):
        if social_embed is not None:
            x = torch.cat([obs, action, social_embed], dim=-1)
        else:
            x = torch.cat([obs, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class GaussianPolicy(nn.Module):
    """Actor Network (Policy)"""

    def __init__(self, obs_dim, action_dim, hidden_dim=256, social_embed_dim=0):
        super(GaussianPolicy, self).__init__()

        input_dim = obs_dim + social_embed_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, obs, social_embed=None):
        if social_embed is not None:
            x = torch.cat([obs, social_embed], dim=-1)
        else:
            x = obs

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs, social_embed=None):
        mean, log_std = self.forward(obs, social_embed)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        mean = torch.tanh(mean)

        return action, log_prob, mean


class SocialEmbeddingNetwork(nn.Module):
    """
    Network to process social observations (other agents' states/actions).

    Input: social_obs = [(s_j, a_j) for j != i]
    Output: embedding vector z
    """

    def __init__(self, state_dim, action_dim, embed_dim=64, hidden_dim=128):
        super(SocialEmbeddingNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Per-agent encoder
        per_agent_input = state_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(per_agent_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Aggregator (mean pooling + FC)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        self.apply(weights_init_)

    def forward(self, social_obs):
        """
        Args:
            social_obs: tensor of shape (batch, n_other_agents, state_dim + action_dim)

        Returns:
            embedding: tensor of shape (batch, embed_dim)
        """
        batch_size, n_agents, _ = social_obs.shape

        # Encode each agent
        # Reshape to (batch * n_agents, state_dim + action_dim)
        social_flat = social_obs.reshape(-1, self.state_dim + self.action_dim)
        encoded = self.encoder(social_flat)  # (batch * n_agents, hidden_dim)

        # Reshape back
        encoded = encoded.reshape(batch_size, n_agents, -1)

        # Aggregate (mean pooling)
        aggregated = encoded.mean(dim=1)  # (batch, hidden_dim)

        # Final embedding
        embedding = self.aggregator(aggregated)  # (batch, embed_dim)

        return embedding


class ActionPredictionHead(nn.Module):
    """
    Auxiliary head to predict actions of other agents from social observations.
    Used for auxiliary loss without using rewards.
    """

    def __init__(self, embed_dim, action_dim):
        super(ActionPredictionHead, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Action is in [-1, 1]
        )

        self.apply(weights_init_)

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, embed_dim)

        Returns:
            predicted_action: (batch, action_dim) - mean predicted action
        """
        return self.predictor(embedding)
