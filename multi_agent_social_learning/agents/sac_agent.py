"""
Soft Actor-Critic (SAC) Agent with Social Learning capabilities
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .networks import QNetwork, GaussianPolicy, SocialEmbeddingNetwork, ActionPredictionHead


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.FloatTensor(self.obs[idx]).to(self.device),
            action=torch.FloatTensor(self.action[idx]).to(self.device),
            reward=torch.FloatTensor(self.reward[idx]).to(self.device),
            next_obs=torch.FloatTensor(self.next_obs[idx]).to(self.device),
            done=torch.FloatTensor(self.done[idx]).to(self.device)
        )
        return batch


class SocialReplayBuffer(ReplayBuffer):
    """Extended Replay Buffer with social observations"""

    def __init__(self, capacity, obs_dim, action_dim, social_obs_dim, device):
        super().__init__(capacity, obs_dim, action_dim, device)
        self.social_obs = np.zeros((capacity, social_obs_dim), dtype=np.float32)
        self.next_social_obs = np.zeros((capacity, social_obs_dim), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, social_obs=None, next_social_obs=None):
        super().add(obs, action, reward, next_obs, done)
        if social_obs is not None:
            self.social_obs[self.ptr - 1] = social_obs
        if next_social_obs is not None:
            self.next_social_obs[self.ptr - 1] = next_social_obs

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        idx = np.random.randint(0, self.size, size=batch_size)

        batch['social_obs'] = torch.FloatTensor(self.social_obs[idx]).to(self.device)
        batch['next_social_obs'] = torch.FloatTensor(self.next_social_obs[idx]).to(self.device)

        return batch


class SACAgent:
    """
    Soft Actor-Critic Agent with optional Social Learning

    Methods:
        - "independent": No social observations
        - "concat": Concatenate social obs directly to policy input
        - "proposed": Use embedding network + auxiliary loss
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        method="independent",
        social_state_dim=0,
        social_action_dim=0,
        n_other_agents=0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        hidden_dim=256,
        social_embed_dim=64,
        aux_loss_weight=0.1
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.method = method
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.aux_loss_weight = aux_loss_weight

        # Store social dimensions for later use
        self.social_state_dim = social_state_dim
        self.social_action_dim = social_action_dim
        self.n_other_agents = n_other_agents

        # Compute dimensions based on method
        policy_input_dim = obs_dim
        social_embedding_dim = 0

        if method == "concat":
            # Direct concatenation
            policy_input_dim += n_other_agents * (social_state_dim + social_action_dim)
        elif method == "proposed":
            # Use embedding
            social_embedding_dim = social_embed_dim
            self.social_encoder = SocialEmbeddingNetwork(
                social_state_dim, social_action_dim, social_embed_dim
            ).to(device)
            self.social_encoder_optimizer = optim.Adam(
                self.social_encoder.parameters(), lr=lr
            )

            # Auxiliary action prediction head
            self.action_predictor = ActionPredictionHead(
                social_embed_dim, social_action_dim
            ).to(device)
            self.action_predictor_optimizer = optim.Adam(
                self.action_predictor.parameters(), lr=lr
            )

        # Create networks
        self.policy = GaussianPolicy(
            policy_input_dim, action_dim, hidden_dim, social_embedding_dim
        ).to(device)

        self.critic_1 = QNetwork(
            policy_input_dim, action_dim, hidden_dim, social_embedding_dim
        ).to(device)
        self.critic_2 = QNetwork(
            policy_input_dim, action_dim, hidden_dim, social_embedding_dim
        ).to(device)

        # Target networks
        self.critic_target_1 = QNetwork(
            policy_input_dim, action_dim, hidden_dim, social_embedding_dim
        ).to(device)
        self.critic_target_2 = QNetwork(
            policy_input_dim, action_dim, hidden_dim, social_embedding_dim
        ).to(device)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None

    def select_action(self, obs, social_obs=None, deterministic=False):
        """Select action given observation (and optional social observation)"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Process social observation based on method
        social_embed = None
        if self.method == "concat":
            if social_obs is not None:
                social_flat = social_obs.flatten()
            else:
                # Provide zero-padding if no social observations available
                social_size = self.policy.fc1.in_features - self.obs_dim
                social_flat = np.zeros(social_size, dtype=np.float32)
            obs = torch.cat([obs, torch.FloatTensor(social_flat).unsqueeze(0).to(self.device)], dim=-1)
        elif self.method == "proposed":
            if social_obs is not None:
                social_tensor = torch.FloatTensor(social_obs).unsqueeze(0).to(self.device)
                social_embed = self.social_encoder(social_tensor)
            # If social_obs is None, social_embed stays None (zero embedding will be used)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.policy.sample(obs, social_embed)
            else:
                action, _, _ = self.policy.sample(obs, social_embed)

        return action.cpu().numpy()[0]

    def update(self, batch, step):
        """Update agent parameters"""
        obs = batch['obs']
        action = batch['action']
        reward = batch['reward']
        next_obs = batch['next_obs']
        done = batch['done']

        # Process social observations
        social_embed = None
        next_social_embed = None
        aux_loss = torch.tensor(0.0).to(self.device)

        if self.method == "proposed" and 'social_obs' in batch:
            social_obs = batch['social_obs']
            next_social_obs = batch['next_social_obs']

            # Reshape social observations from flat to (batch, n_agents, state_dim + action_dim)
            batch_size = obs.shape[0]
            features_per_agent = self.social_state_dim + self.social_action_dim

            # Reshape: (batch, n_agents * features) -> (batch, n_agents, features)
            social_obs = social_obs.reshape(batch_size, self.n_other_agents, features_per_agent)
            next_social_obs = next_social_obs.reshape(batch_size, self.n_other_agents, features_per_agent)

            # Get embeddings
            social_embed = self.social_encoder(social_obs)
            next_social_embed = self.social_encoder(next_social_obs)

            # Compute auxiliary loss (action prediction)
            # Predict actions and compare with ground truth from social observations
            predicted_actions = self.action_predictor(social_embed)
            # Extract actual actions from social_obs (last action_dim dimensions)
            actual_actions = social_obs[:, :, self.social_state_dim:]  # (batch, n_agents, action_dim)
            # Average across agents for single prediction
            actual_actions_mean = actual_actions.mean(dim=1)  # (batch, action_dim)
            aux_loss = F.mse_loss(predicted_actions, actual_actions_mean)

        elif self.method == "concat" and 'social_obs' in batch:
            obs = torch.cat([obs, batch['social_obs']], dim=-1)
            next_obs = torch.cat([next_obs, batch['next_social_obs']], dim=-1)

        # Update critics
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs, next_social_embed)
            target_q1 = self.critic_target_1(next_obs, next_action, next_social_embed)
            target_q2 = self.critic_target_2(next_obs, next_action, next_social_embed)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(obs, action, social_embed)
        current_q2 = self.critic_2(obs, action, social_embed)

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update policy
        new_action, log_prob, _ = self.policy.sample(obs, social_embed)
        q1 = self.critic_1(obs, new_action, social_embed)
        q2 = self.critic_2(obs, new_action, social_embed)
        q = torch.min(q1, q2)

        policy_loss = (self.alpha * log_prob - q).mean()

        # Add auxiliary loss for proposed method
        if self.method == "proposed":
            total_loss = policy_loss + self.aux_loss_weight * aux_loss
        else:
            total_loss = policy_loss

        self.policy_optimizer.zero_grad()
        if self.method == "proposed":
            self.social_encoder_optimizer.zero_grad()
            self.action_predictor_optimizer.zero_grad()

        total_loss.backward()

        self.policy_optimizer.step()
        if self.method == "proposed":
            self.social_encoder_optimizer.step()
            self.action_predictor_optimizer.step()

        # Update alpha (entropy coefficient)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
        }

    def save(self, path):
        """Save agent parameters"""
        checkpoint = {
            'policy': self.policy.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_target_1': self.critic_target_1.state_dict(),
            'critic_target_2': self.critic_target_2.state_dict(),
        }
        if self.method == "proposed":
            checkpoint['social_encoder'] = self.social_encoder.state_dict()
            checkpoint['action_predictor'] = self.action_predictor.state_dict()

        torch.save(checkpoint, path)

    def load(self, path):
        """Load agent parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
        self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])

        if self.method == "proposed" and 'social_encoder' in checkpoint:
            self.social_encoder.load_state_dict(checkpoint['social_encoder'])
            self.action_predictor.load_state_dict(checkpoint['action_predictor'])
