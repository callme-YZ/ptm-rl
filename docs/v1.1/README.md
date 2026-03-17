# PTM-RL v1.1: Toroidal Geometry + Symplectic Integration

## Overview

v1.1 introduces structure-preserving physics to the PTM-RL framework:
- **Toroidal geometry**: Realistic tokamak coordinate system (R, φ, Z)
- **Symplectic time integration**: Energy-conserving numerical methods
- **RL control**: Physics-informed reinforcement learning

## Directory Structure

```
docs/v1.1/
├── design/
│   └── v1.1-toroidal-symplectic-design-v2.1.md  # Master design document (62KB)
├── notes/
│   ├── pyrokinetics-metric-tensor.md            # Toroidal metric tensor reference
│   ├── elsasser-2025-symplectic.md              # Variable-step symplectic methods
│   └── novelty-check.md                         # Scientific novelty verification
└── derivations/
    ├── toroidal-coordinates.md                  # Complete coordinate transformation
    └── symplectic-mhd-hamiltonian.md            # Hamiltonian formulation

```

## Key Documents

### Master Design Document
- **File**: `design/v1.1-toroidal-symplectic-design-v2.1.md` (62KB, 2328 lines)
- **Status**: ✅ Reviewed by 小A (9/10, Conditional Approval)
- **Contents**:
  - Part 1-4: Toroidal geometry + Symplectic integration (physics)
  - Part 5: RL Integration Interface (obs/action/reward/API)
  - Milestones: M1-M4 with acceptance criteria
  - Scientific novelty analysis

### Theory & References
- **Derivations**: Complete mathematical foundations
- **Notes**: Literature review and external references
- **Novelty check**: Competitive landscape analysis

## Implementation Phases

### Phase 1: Toroidal Geometry (Layer 1-2, 小P)
- Implement ToroidalGrid class
- Metric tensor computation
- Differential operators in toroidal coordinates
- Validation: convergence + coordinate transformation

### Phase 2: Symplectic Integration (Layer 2, 小P)
- Implement Störmer-Verlet or Elsässer integrator
- Replace RK4 time-stepping
- Validation: energy conservation <1e-8, symplectic residual <1e-12
- Long-time stability testing (>10⁴ steps)

### Phase 3: Integration + Testing (Layer 1-3, 小P → 小A)
- Combined toroidal+symplectic testing
- Update diagnostics (island_width_toroidal, etc.)
- **M3.1 Gate**: Decide fixed-dt vs adaptive-dt training strategy
- Performance benchmarking (target: 2-3× slowdown)
- Handoff to 小A for RL environment adaptation

### Phase 4: RL Training (Layer 3, 小A)
- RL environment wrapper implementation
- Policy training (100k steps)
- Validation: reward improvement, island width suppression
- Final acceptance: performance <3×, physics validated

## Status

**Current:** Design complete, awaiting implementation start  
**Next:** Create feature branch, begin Phase 1 (Toroidal Geometry)

## Version History

- **v2.1** (2026-03-17): Final design with RL integration interface
- **v2** (2026-03-17): Added theory derivations and novelty analysis
- **v1** (2026-03-17): Initial design specification

---

**Maintained by:** 小P (physics), 小A (RL), ∞ (coordination)  
**Project:** PTM-RL (Plasma Tearing Mode Control via Reinforcement Learning)  
**Repository:** https://github.com/callme-YZ/ptm-rl
