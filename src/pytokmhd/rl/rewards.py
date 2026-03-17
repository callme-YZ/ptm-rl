"""
Reward computation for MHD control task.

Multi-objective reward function for equilibrium maintenance.
"""

import numpy as np
from typing import Dict, Tuple


class RewardComputer:
    """
    Compute multi-objective reward for MHD control.
    
    Objective: Maintain plasma equilibrium
    
    Reward components:
    - Energy deviation: -w_E * |E - E_eq| / E_eq
    - Constraint violation: -w_B * max|∇·B| / threshold
    - Action penalty: -w_A * |action|²
    
    Parameters
    ----------
    w_energy : float
        Energy weight (default: 1.0)
    w_constraint : float
        Constraint weight (default: 0.1)
    w_action : float
        Action penalty weight (default: 0.01)
    """
    
    def __init__(
        self,
        w_energy: float = 1.0,
        w_constraint: float = 0.1,
        w_action: float = 0.01
    ):
        self.w_energy = w_energy
        self.w_constraint = w_constraint
        self.w_action = w_action
    
    def compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward from observation and action.
        
        Parameters
        ----------
        obs : dict
            Observation dictionary with keys:
            - 'energy_drift': |E - E_eq| / E_eq
            - 'div_B_max': max|∇·B| / threshold
        action : np.ndarray
            Action taken (for penalty)
        
        Returns
        -------
        reward : float
            Total reward
        info : dict
            Component breakdown for logging
        """
        # Energy term (primary objective)
        energy_drift = obs['energy_drift'][0]
        r_energy = -self.w_energy * energy_drift
        
        # Constraint term (∇·B violation)
        div_B_normalized = obs['div_B_max'][0]
        r_constraint = -self.w_constraint * div_B_normalized
        
        # Action penalty (control effort)
        action_norm = np.linalg.norm(action)
        r_action = -self.w_action * (action_norm ** 2)
        
        # Total reward
        reward = r_energy + r_constraint + r_action
        
        # Component breakdown for logging
        info = {
            'reward_energy': r_energy,
            'reward_constraint': r_constraint,
            'reward_action': r_action,
            'reward_total': reward,
            'energy_drift': energy_drift,
            'div_B_max': div_B_normalized,
            'action_norm': action_norm
        }
        
        return reward, info
    
    def check_failure(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if episode should terminate due to physics violation.
        
        Parameters
        ----------
        obs : dict
            Observation
        
        Returns
        -------
        failed : bool
            True if physics constraints violated
        """
        # Failure: ∇·B too large
        div_B_normalized = obs['div_B_max'][0]
        if div_B_normalized > 10.0:  # 10× threshold
            return True
        
        # Failure: Energy exploded
        energy_drift = obs['energy_drift'][0]
        if energy_drift > 10.0:  # 1000% deviation
            return True
        
        return False
    
    def check_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if episode achieved success.
        
        Parameters
        ----------
        obs : dict
            Observation
        
        Returns
        -------
        success : bool
            True if equilibrium well maintained
        """
        # Success: Energy drift < 1%
        energy_drift = obs['energy_drift'][0]
        if energy_drift < 0.01:
            return True
        
        return False


def compute_terminal_reward(
    reason: str,
    bonus_success: float = 10.0,
    penalty_failure: float = -10.0
) -> float:
    """
    Compute terminal reward at episode end.
    
    Parameters
    ----------
    reason : str
        'success', 'failure', or 'timeout'
    bonus_success : float
        Bonus for successful episode
    penalty_failure : float
        Penalty for failed episode
    
    Returns
    -------
    terminal_reward : float
    """
    if reason == 'success':
        return bonus_success
    elif reason == 'failure':
        return penalty_failure
    else:  # timeout
        return 0.0
