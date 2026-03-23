"""
Hamiltonian-Guided Actor-Critic Policy for Stable Baselines3

Implements the Hamiltonian RL approach from Phase A design:
- Encoder: obs (113) → latent (8)
- Pseudo-Hamiltonian network: latent → H estimate
- Actor: constrained by ∇H
- Critic: standard value function

Author: 小A 🤖
Date: 2026-03-23
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
    """Encode high-dim MHD observation → low-dim latent
    
    Maps 113-dim observation to 8-dim latent representation
    capturing essential physics state.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        # Encoder network: 113 → 64 → 32 → 8
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, features_dim),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space
        
        Args:
            observations: (batch, 113) MHD state
            
        Returns:
            latent: (batch, 8) compressed representation
        """
        return self.encoder(observations)


class PseudoHamiltonianNetwork(nn.Module):
    """Pseudo-Hamiltonian estimator
    
    Learns H(z) ≈ Total Energy from latent state z
    Used to guide policy via ∇H constraint.
    """
    
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        
        # H network: 8 → 32 → 16 → 1 (scalar H)
        self.h_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-Hamiltonian
        
        Args:
            latent: (batch, 8) latent state
            
        Returns:
            H: (batch, 1) pseudo-Hamiltonian value
        """
        return self.h_net(latent)
    
    def gradient(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute ∇H w.r.t. latent
        
        Args:
            latent: (batch, 8) latent state (requires_grad=True)
            
        Returns:
            grad_H: (batch, 8) gradient of H
        """
        latent = latent.requires_grad_(True)
        H = self.forward(latent)
        grad_H = torch.autograd.grad(
            H.sum(), latent, 
            create_graph=True,
            retain_graph=True
        )[0]
        return grad_H


class HamiltonianActor(nn.Module):
    """Actor network with Hamiltonian guidance
    
    Policy π(a|s) is constrained by ∇H to respect physics structure.
    """
    
    def __init__(self, latent_dim: int = 8, action_dim: int = 4, lambda_h: float = 1.0):
        super().__init__()
        
        self.lambda_h = lambda_h  # Hamiltonian guidance strength
        
        # Mean network: 8 → 32 → action_dim
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),  # Bounded actions [-1, 1]
        )
        
        # Log std (learned parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, latent: torch.Tensor, grad_h: torch.Tensor = None) -> tuple:
        """Compute action distribution
        
        Args:
            latent: (batch, 8) latent state
            grad_h: (batch, 8) ∇H gradient (optional)
            
        Returns:
            mean: (batch, action_dim) action mean
            std: (batch, action_dim) action std
        """
        # Base policy mean
        mean = self.mean_net(latent)
        
        # Apply Hamiltonian guidance if provided
        if grad_h is not None and self.lambda_h > 0:
            # Project grad_H to action space (simplified: use first 4 dims)
            h_correction = -self.lambda_h * grad_h[:, :mean.shape[1]]
            mean = mean + h_correction
            
            # Re-clip to [-1, 1]
            mean = torch.tanh(mean)
        
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std


class HamiltonianActorCriticPolicy(ActorCriticPolicy):
    """Complete Hamiltonian-guided Actor-Critic Policy
    
    Integrates:
    - Observation encoder
    - Pseudo-Hamiltonian network
    - Hamiltonian-constrained actor
    - Standard critic
    
    Compatible with SB3 PPO/SAC.
    """
    
    def __init__(self, *args, lambda_h: float = 1.0, **kwargs):
        """Initialize policy
        
        Args:
            lambda_h: Hamiltonian guidance strength (0.0 = baseline)
        """
        self.lambda_h = lambda_h
        
        # Call parent init (will trigger _build)
        super().__init__(*args, **kwargs, 
                         features_extractor_class=ObservationEncoder,
                         features_extractor_kwargs=dict(features_dim=8))
        
    def _build(self, lr_schedule):
        """Build policy networks"""
        # Get latent dim from features extractor
        latent_dim = self.features_extractor.features_dim
        
        # Build Hamiltonian network
        self.h_network = PseudoHamiltonianNetwork(latent_dim)
        
        # Build actor with Hamiltonian guidance
        self.hamiltonian_actor = HamiltonianActor(
            latent_dim=latent_dim,
            action_dim=self.action_space.shape[0],
            lambda_h=self.lambda_h
        )
        
        # Build standard critic (value function)
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Optimizer for all networks
        self.optimizer = torch.optim.Adam(
            list(self.features_extractor.parameters()) +
            list(self.h_network.parameters()) +
            list(self.hamiltonian_actor.parameters()) +
            list(self.value_net.parameters()),
            lr=lr_schedule(1)
        )
        
    def forward(self, obs, deterministic: bool = False):
        """Forward pass for action sampling
        
        Args:
            obs: Observations
            deterministic: Use mean action if True
            
        Returns:
            actions, values, log_probs
        """
        # Encode observations
        latent = self.features_extractor(obs)
        
        # Compute Hamiltonian gradient (for guidance)
        if self.lambda_h > 0:
            grad_h = self.h_network.gradient(latent)
        else:
            grad_h = None
        
        # Get action distribution
        mean, std = self.hamiltonian_actor(latent, grad_h)
        distribution = Normal(mean, std)
        
        # Sample action
        if deterministic:
            actions = mean
        else:
            actions = distribution.rsample()
        
        # Compute log prob
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        # Compute value
        values = self.value_net(latent)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update
        
        Args:
            obs: Observations
            actions: Actions taken
            
        Returns:
            values, log_probs, entropy
        """
        # Encode
        latent = self.features_extractor(obs)
        
        # Hamiltonian gradient
        if self.lambda_h > 0:
            grad_h = self.h_network.gradient(latent)
        else:
            grad_h = None
        
        # Distribution
        mean, std = self.hamiltonian_actor(latent, grad_h)
        distribution = Normal(mean, std)
        
        # Evaluate
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.value_net(latent)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        """Predict values for critic
        
        Args:
            obs: Observations
            
        Returns:
            values
        """
        latent = self.features_extractor(obs)
        return self.value_net(latent)
