"""
Hamiltonian-Regularized Actor-Critic Policy for Stable Baselines3

Simplified approach: Use H as regularization loss instead of ∇H guidance
- More stable and SB3-compatible
- Still physics-informed via energy minimization

Author: 小A 🤖
Date: 2026-03-23 (simplified version)
Milestone: v2.1-hamiltonian
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class ObservationEncoder(BaseFeaturesExtractor):
    """Encode high-dim MHD observation → low-dim latent"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, features_dim),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class HamiltonianNetwork(nn.Module):
    """Hamiltonian estimator - learns total energy"""
    
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        
        self.h_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(z)"""
        return self.h_net(latent)


class HamiltonianActorCriticPolicy(ActorCriticPolicy):
    """Hamiltonian-Regularized Policy
    
    Simplified approach:
    - Standard actor-critic architecture
    - Additional H-network for energy estimation
    - H used as regularization term in loss (not in action generation)
    - SB3-compatible, stable, physics-informed
    """
    
    def __init__(self, *args, lambda_h: float = 1.0, **kwargs):
        self.lambda_h = lambda_h
        super().__init__(*args, **kwargs,
                         features_extractor_class=ObservationEncoder,
                         features_extractor_kwargs=dict(features_dim=8))
        
    def _build(self, lr_schedule):
        """Build networks"""
        super()._build(lr_schedule)
        
        # Add Hamiltonian network
        latent_dim = self.features_extractor.features_dim
        self.h_network = HamiltonianNetwork(latent_dim)
        
        # Extend optimizer to include H-network
        self.optimizer = torch.optim.Adam(
            list(self.parameters()) + list(self.h_network.parameters()),
            lr=lr_schedule(1)
        )
        
    def forward(self, obs, deterministic: bool = False):
        """Standard forward pass (no H involved here)"""
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Get action distribution
        mean_actions = self.action_net(latent_pi)
        
        if isinstance(self.action_dist, Normal.__class__):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            distribution = self.action_dist.proba_distribution(mean_actions)
        
        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Compute values
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions - standard SB3 interface"""
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        mean_actions = self.action_net(latent_pi)
        
        if isinstance(self.action_dist, Normal.__class__):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            distribution = self.action_dist.proba_distribution(mean_actions)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
    def get_hamiltonian_loss(self, obs):
        """Compute Hamiltonian regularization loss
        
        To be called during training update (not in forward pass)
        Encourages policy to minimize energy
        
        Args:
            obs: Observations
            
        Returns:
            h_loss: Hamiltonian regularization term
        """
        if self.lambda_h == 0:
            return torch.tensor(0.0)
        
        # Encode observations
        latent = self.features_extractor(obs)
        
        # Compute Hamiltonian
        H = self.h_network(latent)
        
        # Regularization: penalize high energy states
        # Loss = λ * mean(H²) encourages low H
        h_loss = self.lambda_h * (H ** 2).mean()
        
        return h_loss
