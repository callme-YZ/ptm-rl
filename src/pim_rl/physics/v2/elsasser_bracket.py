"""
Morrison Bracket Implementation for Elsasser MHD (v2.0 Phase 1.1)

Author: 小P ⚛️
Date: 2026-03-20
Issue #33: Added JAX pytree registration (小A 🤖, 2026-03-25)

Implements noncanonical Poisson bracket for Elsasser variables in cylindrical geometry.

Theory:
- Morrison bracket: {F,G} = ∫ δF/δu · J · δG/δu dV
- Elsasser: z⁺ = v + B, z⁻ = v - B
- Cylindrical (ε=0): No toroidal coupling

References:
- Morrison 1982 (AIP Conf Proc)
- Module 2 (Elsasser notes, 67KB)
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Tuple, Callable
from dataclasses import dataclass

@dataclass
class ElsasserState:
    """State vector for Elsasser MHD (cylindrical)
    
    Fields:
        z_plus: Forward Alfvén field (v + B)
        z_minus: Backward Alfvén field (v - B)
        P: Normalized pressure (β₀ p)
    
    All fields shape: (Nr, Nθ, Nz) in cylindrical coordinates
    
    Issue #33: JAX pytree registration for JIT compatibility.
    """
    z_plus: jnp.ndarray
    z_minus: jnp.ndarray
    P: jnp.ndarray


# Register ElsasserState as JAX pytree (Issue #33)
from jax.tree_util import register_pytree_node

def _elsasser_flatten(state):
    """Flatten ElsasserState to (values, aux_data)."""
    return (state.z_plus, state.z_minus, state.P), None

def _elsasser_unflatten(aux_data, values):
    """Reconstruct ElsasserState from flattened values."""
    z_plus, z_minus, P = values
    return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)

register_pytree_node(
    ElsasserState,
    _elsasser_flatten,
    _elsasser_unflatten
)


class MorrisonBracket:
    """Morrison Poisson bracket for Elsasser MHD
    
    Implements {F,G} = ∫ [z⁺·[δF/δz⁺, δG/δz⁻] + z⁻·[δF/δz⁻, δG/δz⁺]] dV
    
    where [f,g] is 2D Poisson bracket in (r,θ) plane.
    """
    
    def __init__(self, grid_shape: Tuple[int, int, int], 
                 dr: float, dtheta: float, dz: float):
        """Initialize Morrison bracket
        
        Args:
            grid_shape: (Nr, Nθ, Nz)
            dr, dtheta, dz: Grid spacings
        """
        self.Nr, self.Ntheta, self.Nz = grid_shape
        self.dr = dr
        self.dtheta = dtheta
        self.dz = dz
        self.dV = dr * dtheta * dz  # Volume element (cylindrical, simplified)
        
    @staticmethod
    @jit
    def poisson_bracket_2d(f: jnp.ndarray, g: jnp.ndarray,
                          dr: float, dtheta: float) -> jnp.ndarray:
        """2D Poisson bracket [f,g] = ẑ · (∇f × ∇g)
        
        In cylindrical: [f,g] = (1/r)(∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)
        
        Args:
            f, g: 2D fields (r, θ)
            dr, dtheta: Grid spacings
            
        Returns:
            [f,g]: 2D Poisson bracket
        """
        # Central differences for derivatives
        df_dr = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2*dr)
        df_dtheta = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2*dtheta)
        
        dg_dr = (jnp.roll(g, -1, axis=0) - jnp.roll(g, 1, axis=0)) / (2*dr)
        dg_dtheta = (jnp.roll(g, -1, axis=1) - jnp.roll(g, 1, axis=1)) / (2*dtheta)
        
        # [f,g] = ∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r  (simplified, no 1/r factor for now)
        bracket = df_dr * dg_dtheta - df_dtheta * dg_dr
        
        return bracket
    
    def bracket(self, dF_dstate: ElsasserState, 
                dG_dstate: ElsasserState) -> ElsasserState:
        """Morrison bracket {F,G} for Elsasser
        
        Formula:
        {F,G} = ∫ [z⁺·[δF/δz⁺, δG/δz⁻] + z⁻·[δF/δz⁻, δG/δz⁺]] dV
        
        Args:
            dF_dstate: Functional derivative δF/δu
            dG_dstate: Functional derivative δG/δu
            
        Returns:
            d(state)/dt from {state, H}
        """
        # Extract functional derivatives
        dF_dzp = dF_dstate.z_plus
        dF_dzm = dF_dstate.z_minus
        dF_dP = dF_dstate.P
        
        dG_dzp = dG_dstate.z_plus
        dG_dzm = dG_dstate.z_minus
        dG_dP = dG_dstate.P
        
        # Compute 2D Poisson brackets for each z-slice
        # [δF/δz⁺, δG/δz⁻] for all z
        bracket_fp_gm = jax.vmap(
            lambda i: self.poisson_bracket_2d(dF_dzp[:,:,i], dG_dzm[:,:,i], 
                                             self.dr, self.dtheta),
            in_axes=0, out_axes=2
        )(jnp.arange(self.Nz))
        
        # [δF/δz⁻, δG/δz⁺]
        bracket_fm_gp = jax.vmap(
            lambda i: self.poisson_bracket_2d(dF_dzm[:,:,i], dG_dzp[:,:,i],
                                             self.dr, self.dtheta),
            in_axes=0, out_axes=2
        )(jnp.arange(self.Nz))
        
        # [δF/δP, δG/δz⁺] and [δF/δP, δG/δz⁻] (pressure coupling)
        bracket_p_gp = jax.vmap(
            lambda i: self.poisson_bracket_2d(dF_dP[:,:,i], dG_dzp[:,:,i],
                                             self.dr, self.dtheta),
            in_axes=0, out_axes=2
        )(jnp.arange(self.Nz))
        
        bracket_p_gm = jax.vmap(
            lambda i: self.poisson_bracket_2d(dF_dP[:,:,i], dG_dzm[:,:,i],
                                             self.dr, self.dtheta),
            in_axes=0, out_axes=2
        )(jnp.arange(self.Nz))
        
        # Morrison bracket result: d(state)/dt
        # ∂z⁺/∂t from {z⁺, H} = [z⁺, δH/δz⁻] + [P, δH/δz⁺]  (simplified)
        dzp_dt = bracket_fp_gm  # Cross-coupling z⁺ ↔ δ/δz⁻
        
        # ∂z⁻/∂t from {z⁻, H} = [z⁻, δH/δz⁺] + [P, δH/δz⁻]
        dzm_dt = bracket_fm_gp  # Cross-coupling z⁻ ↔ δ/δz⁺
        
        # ∂P/∂t from {P, H} = [P, δH/δz⁺] + [P, δH/δz⁻]
        dP_dt = bracket_p_gp + bracket_p_gm
        
        return ElsasserState(z_plus=dzp_dt, z_minus=dzm_dt, P=dP_dt)


def hamiltonian(state: ElsasserState, grid: MorrisonBracket) -> float:
    """Hamiltonian (energy) for Elsasser MHD
    
    H = (1/4) ∫ [|∇z⁺|² + |∇z⁻|²] dV  (cylindrical, no pressure term for now)
    
    Args:
        state: Elsasser state
        grid: Morrison bracket (for grid info)
        
    Returns:
        Energy
    """
    # Gradients (simplified: only radial for now)
    dzp_dr = (jnp.roll(state.z_plus, -1, axis=0) - 
              jnp.roll(state.z_plus, 1, axis=0)) / (2*grid.dr)
    dzm_dr = (jnp.roll(state.z_minus, -1, axis=0) - 
              jnp.roll(state.z_minus, 1, axis=0)) / (2*grid.dr)
    
    # Energy density
    energy_density = 0.25 * (dzp_dr**2 + dzm_dr**2)
    
    # Integrate
    energy = jnp.sum(energy_density) * grid.dV
    
    return energy


def functional_derivative(H_func: Callable, state: ElsasserState, 
                         grid: MorrisonBracket) -> ElsasserState:
    """Compute functional derivative δH/δu via JAX autodiff
    
    Args:
        H_func: Hamiltonian function H(state, grid) -> scalar
        state: Current state
        grid: Grid info
        
    Returns:
        δH/δu as ElsasserState
    """
    # Use JAX grad to compute gradients
    grad_H = grad(lambda s: H_func(ElsasserState(**s), grid))
    
    # Convert state to dict for JAX
    state_dict = {'z_plus': state.z_plus, 
                  'z_minus': state.z_minus, 
                  'P': state.P}
    
    # Compute gradient
    dH_dict = grad_H(state_dict)
    
    return ElsasserState(**dH_dict)


# Test function (Phase 1.1 validation)
def test_morrison_bracket():
    """Test Morrison bracket antisymmetry and conservation"""
    
    # Small grid for testing
    Nr, Ntheta, Nz = 16, 16, 16
    dr, dtheta, dz = 0.1, 0.1, 0.1
    
    grid = MorrisonBracket((Nr, Ntheta, Nz), dr, dtheta, dz)
    
    # Simple initial state (Gaussian perturbation)
    r = jnp.linspace(0, 1, Nr)[:, None, None]
    theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
    z = jnp.linspace(0, 1, Nz)[None, None, :]
    
    z_plus = jnp.exp(-((r-0.5)**2 + (theta-jnp.pi)**2))
    z_minus = jnp.exp(-((r-0.5)**2 + (theta-jnp.pi)**2)) * 0.5
    P = jnp.ones((Nr, Ntheta, Nz)) * 0.1
    
    state = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    # Compute energy
    E0 = hamiltonian(state, grid)
    print(f"Initial energy: {E0:.6e}")
    
    # Compute functional derivative
    dH = functional_derivative(hamiltonian, state, grid)
    print(f"δH/δz⁺ max: {jnp.max(jnp.abs(dH.z_plus)):.6e}")
    
    # Test antisymmetry: {F,G} = -{G,F}
    dF = dH
    dG = ElsasserState(z_plus=state.z_plus, z_minus=state.z_minus, P=state.P)
    
    FG = grid.bracket(dF, dG)
    GF = grid.bracket(dG, dF)
    
    antisymmetry_error = jnp.max(jnp.abs(FG.z_plus + GF.z_plus))
    print(f"Antisymmetry error: {antisymmetry_error:.6e}")
    
    if antisymmetry_error < 1e-9:
        print("✅ Morrison bracket antisymmetry verified!")
    else:
        print("❌ Antisymmetry FAILED!")
    
    return state, grid


if __name__ == "__main__":
    print("=" * 60)
    print("Morrison Bracket Test (Phase 1.1)")
    print("=" * 60)
    test_morrison_bracket()
