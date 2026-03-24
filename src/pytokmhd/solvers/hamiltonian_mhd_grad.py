"""
Hamiltonian Gradient API for RL Integration

Provides JAX-compatible gradient computation for Hamiltonian energy.

Issue #24 Task 4: RL environment API for ∇H

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Tuple

from ..geometry.toroidal import ToroidalGrid


def _compute_derivatives_jax(f, dr, dtheta):
    """
    JAX-compatible derivative computation (same as test version).
    
    2nd-order centered differences with appropriate boundary conditions.
    """
    nr, ntheta = f.shape
    
    # Initialize
    df_dr = jnp.zeros_like(f)
    df_dtheta = jnp.zeros_like(f)
    
    # Radial derivatives (2nd order centered)
    df_dr = df_dr.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2*dr))
    # Boundary: 2nd order one-sided
    df_dr = df_dr.at[0, :].set((-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dr))
    df_dr = df_dr.at[-1, :].set((3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dr))
    
    # Theta derivatives (periodic, 2nd order centered)
    df_dtheta = df_dtheta.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2*dtheta))
    # Periodic BC
    df_dtheta = df_dtheta.at[:, 0].set((f[:, 1] - f[:, -1]) / (2*dtheta))
    df_dtheta = df_dtheta.at[:, -1].set((f[:, 0] - f[:, -2]) / (2*dtheta))
    
    return df_dr, df_dtheta


def hamiltonian_energy_jax(psi: jnp.ndarray, 
                           phi: jnp.ndarray,
                           r_grid: jnp.ndarray,
                           R_grid: jnp.ndarray,
                           dr: float,
                           dtheta: float) -> float:
    """
    Compute Hamiltonian energy (JAX-compatible).
    
    H = ∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²] dV
    
    Parameters
    ----------
    psi : jnp.ndarray (nr, ntheta)
        Poloidal magnetic flux
    phi : jnp.ndarray (nr, ntheta)
        Electrostatic potential (stream function)
    r_grid : jnp.ndarray (nr, ntheta)
        Minor radius grid
    R_grid : jnp.ndarray (nr, ntheta)
        Major radius grid
    dr : float
        Radial grid spacing
    dtheta : float
        Poloidal grid spacing
        
    Returns
    -------
    H : float
        Total Hamiltonian energy
        
    Notes
    -----
    This function is designed for JAX autodiff:
    - Pure function (no side effects)
    - Uses jax.numpy operations
    - JIT-compilable
    """
    # Compute derivatives
    dpsi_dr, dpsi_dtheta = _compute_derivatives_jax(psi, dr, dtheta)
    dphi_dr, dphi_dtheta = _compute_derivatives_jax(phi, dr, dtheta)
    
    # |∇ψ|² = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)²
    grad_psi_sq = dpsi_dr**2 + (dpsi_dtheta / r_grid)**2
    
    # |∇φ|² = (∂φ/∂r)² + (1/r²)(∂φ/∂θ)²
    grad_phi_sq = dphi_dr**2 + (dphi_dtheta / r_grid)**2
    
    # Energy density
    h = 0.5 * (grad_psi_sq + grad_phi_sq)
    
    # Volume element: r*R dr dθ
    jacobian = r_grid * R_grid
    
    # Integrate over poloidal plane
    energy_2d = jnp.sum(h * jacobian) * dr * dtheta
    
    # Multiply by 2π (toroidal direction)
    H = 2 * jnp.pi * energy_2d
    
    return H


class HamiltonianGradientComputer:
    """
    Compute Hamiltonian and its gradients for RL integration.
    
    This class provides an efficient API for computing H and ∇H
    simultaneously, which is needed for Hamiltonian RL algorithms.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Computational grid
        
    Attributes
    ----------
    grad_psi_func : callable
        JIT-compiled gradient function for ∇_ψ H
    grad_phi_func : callable
        JIT-compiled gradient function for ∇_φ H
        
    Examples
    --------
    >>> from pytokmhd.geometry.toroidal import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    >>> grad_computer = HamiltonianGradientComputer(grid)
    >>> 
    >>> # Compute H and gradients
    >>> psi = ...  # State
    >>> phi = ...
    >>> H, grad_psi, grad_phi = grad_computer.compute_all(psi, phi)
    """
    
    def __init__(self, grid: ToroidalGrid):
        """Initialize gradient computer with grid parameters"""
        self.grid = grid
        
        # Convert grid to JAX arrays (once, for efficiency)
        self.r_grid_jax = jnp.array(grid.r_grid)
        self.R_grid_jax = jnp.array(grid.R_grid)
        self.dr = grid.dr
        self.dtheta = grid.dtheta
        
        # Create gradient functions
        def H_func(psi, phi):
            return hamiltonian_energy_jax(
                psi, phi, 
                self.r_grid_jax, self.R_grid_jax,
                self.dr, self.dtheta
            )
        
        # JIT-compile for performance
        self.H_func = jit(H_func)
        self.grad_psi_func = jit(grad(H_func, argnums=0))
        self.grad_phi_func = jit(grad(H_func, argnums=1))
    
    def compute_energy(self, psi: jnp.ndarray, phi: jnp.ndarray) -> float:
        """
        Compute Hamiltonian energy H.
        
        Parameters
        ----------
        psi : jnp.ndarray (nr, ntheta)
            Poloidal flux
        phi : jnp.ndarray (nr, ntheta)
            Stream function
            
        Returns
        -------
        H : float
            Hamiltonian energy
        """
        return self.H_func(psi, phi)
    
    def compute_gradients(self, psi: jnp.ndarray, phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Hamiltonian gradients ∇H.
        
        Parameters
        ----------
        psi : jnp.ndarray (nr, ntheta)
            Poloidal flux
        phi : jnp.ndarray (nr, ntheta)
            Stream function
            
        Returns
        -------
        grad_psi : jnp.ndarray (nr, ntheta)
            ∇_ψ H = δH/δψ
        grad_phi : jnp.ndarray (nr, ntheta)
            ∇_φ H = δH/δφ
            
        Notes
        -----
        Typical execution time: ~8 μs (from Task 3 benchmark)
        """
        grad_psi = self.grad_psi_func(psi, phi)
        grad_phi = self.grad_phi_func(psi, phi)
        return grad_psi, grad_phi
    
    def compute_all(self, psi: jnp.ndarray, phi: jnp.ndarray) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """
        Compute H and ∇H in one call.
        
        This is the recommended method for RL integration,
        as it computes both energy and gradients efficiently.
        
        Parameters
        ----------
        psi : jnp.ndarray (nr, ntheta)
            Poloidal flux
        phi : jnp.ndarray (nr, ntheta)
            Stream function
            
        Returns
        -------
        H : float
            Hamiltonian energy
        grad_psi : jnp.ndarray (nr, ntheta)
            ∇_ψ H
        grad_phi : jnp.ndarray (nr, ntheta)
            ∇_φ H
            
        Examples
        --------
        >>> H, grad_psi, grad_phi = grad_computer.compute_all(psi, phi)
        >>> print(f"Energy: {H:.6e}")
        >>> print(f"Max |∇_ψ H|: {jnp.max(jnp.abs(grad_psi)):.6e}")
        """
        H = self.compute_energy(psi, phi)
        grad_psi, grad_phi = self.compute_gradients(psi, phi)
        return H, grad_psi, grad_phi


# Convenience function for single-use
def compute_hamiltonian_gradient(psi: jnp.ndarray, 
                                 phi: jnp.ndarray,
                                 grid: ToroidalGrid) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Convenience function to compute H and ∇H.
    
    For repeated calls, use HamiltonianGradientComputer instead
    (avoids re-creating JIT-compiled functions).
    
    Parameters
    ----------
    psi : jnp.ndarray (nr, ntheta)
        Poloidal flux
    phi : jnp.ndarray (nr, ntheta)
        Stream function
    grid : ToroidalGrid
        Computational grid
        
    Returns
    -------
    H : float
        Hamiltonian energy
    grad_psi : jnp.ndarray (nr, ntheta)
        ∇_ψ H
    grad_phi : jnp.ndarray (nr, ntheta)
        ∇_φ H
    """
    computer = HamiltonianGradientComputer(grid)
    return computer.compute_all(psi, phi)
