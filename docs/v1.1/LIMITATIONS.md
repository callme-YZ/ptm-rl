# v1.1 Known Limitations and Future Work

**Project:** PTM-RL v1.1 - Toroidal Symplectic MHD  
**Date:** 2026-03-17  
**Status:** M3 Step 3.3 completed

---

## Executive Summary

v1.1 implements a **minimal toroidal MHD solver** to validate the toroidal geometry + symplectic integrator framework. While tests pass, there are known physics limitations that will be addressed in v1.2 and v2.0.

---

## Critical Limitation: Integrator-Equation Mismatch

### Issue

**Current Implementation (M3 Step 3.1):**
```python
# Pure diffusion equations
∂ψ/∂t = -η*J    # J = -∇²ψ
∂ω/∂t = -ν*∇²ω
```

**Integrator (M2):**
- Störmer-Verlet symplectic integrator
- Designed for **Hamiltonian systems** (energy-conserving)
- Preserves symplectic structure in phase space

**Problem:**
- Pure diffusion equations are **dissipative**, not Hamiltonian
- Symplectic integrators do not guarantee stability for purely dissipative systems
- Energy can grow exponentially for non-trivial initial conditions

### Manifestation

**Test Case:**
```python
# Non-trivial initial condition
psi = r**4

# Result:
# Energy grows exponentially
# Overflow to inf after ~42 steps
```

**Why M3 Tests Pass:**
- Test 3 (Energy Conservation) uses **trivial equilibrium**: ψ = constant
- For constant ψ: ∇²ψ = 0 → ∂ψ/∂t = 0 → energy trivially conserved
- This does not test integrator stability for realistic evolution

---

## Root Cause: Hamiltonian vs Dissipative

### Hamiltonian System (M2 was designed for)
```
H(q, p) = kinetic + potential
∂q/∂t = ∂H/∂p
∂p/∂t = -∂H/∂q

→ dH/dt = 0 (energy conserved)
→ Symplectic integrator preserves structure
```

### Current MHD (M3 implements)
```
∂ψ/∂t = -η*∇²ψ  (independent diffusion)
∂ω/∂t = -ν*∇²ω  (independent diffusion)

→ dE/dt < 0 (energy dissipates)
→ NOT Hamiltonian
→ Symplectic NOT applicable
```

### What's Missing: Poisson Bracket Terms

**Full reduced MHD:**
```
∂ψ/∂t = [ψ, φ] - η*J       # Hamiltonian + dissipation
∂ω/∂t = [ω, φ] - ν*∇²ω     # Hamiltonian + dissipation
```

where `[f, g] = ∂f/∂r * ∂g/∂θ - ∂f/∂θ * ∂g/∂r` (Poisson bracket)

**With Poisson bracket:**
- System = Hamiltonian (conservative part) + dissipation (small)
- Symplectic integrator preserves Hamiltonian structure
- Dissipation handled as perturbation
- **This is the correct formulation**

---

## Design Decision: Why v1.1 is Minimal

### Decision Made: 2026-03-17

**YZ approved:** Accept minimal implementation, defer Poisson bracket to v1.2

**Rationale:**

1. **v1.1 Goal:** Validate toroidal geometry + symplectic framework feasibility
   - ✅ Toroidal geometry implemented and tested (M1)
   - ✅ Symplectic integrator implemented (M2)
   - ✅ Integration works for trivial cases (M3)
   - **Goal achieved**

2. **Poisson Bracket is v2.0 Prerequisite:**
   - v2.0 requires full Hamiltonian formulation
   - Elsässer variables (z⁺, z⁻) are Hamiltonian by nature
   - Adding Poisson bracket now = partial v2.0 work
   - Better to do comprehensively in v1.2 as v2.0 bridge

3. **Project Timeline:**
   - Adding Poisson bracket now: +1-2 weeks
   - Would block M4 (RL integration)
   - Would delay v2.0 overall timeline
   - Minimal v1.1 → fast M4 → v1.2 → v2.0 is cleaner progression

---

## Roadmap

### v1.1 (Current) - Minimal Baseline ✅
**Scope:**
- Toroidal geometry operators
- Symplectic integrator framework
- Pure diffusion MHD (minimal physics)
- Proof-of-concept for RL integration (M4)

**Limitations:**
- No Poisson bracket (not Hamiltonian)
- Symplectic integrator underutilized
- Only suitable for near-equilibrium, small perturbations
- Cannot simulate realistic tearing mode evolution

**Status:** M3 Step 3.3 completed, ready for M4

---

### v1.2 (Future) - Hamiltonian MHD
**Scope:**
- Add Poisson bracket terms `[ψ, φ]`, `[ω, φ]`
- Implement Poisson solver (∇²φ = -ω)
- Transform to Hamiltonian + dissipation formulation
- Fully utilize symplectic integrator

**Benefits:**
- Symplectic structure preservation becomes meaningful
- Realistic tearing mode simulation
- Better energy conservation
- Bridge to v2.0

**Prerequisites:**
- v1.1 completed ✅
- M4 RL integration completed
- Poisson solver design and implementation

**Estimated Timeline:** 2-3 weeks

---

### v2.0 (Goal) - Elsässer Hamiltonian
**Scope:**
- Elsässer variables: z⁺ = u + B, z⁻ = u - B
- Wu (2024) time transformation for toroidal geometry
- Full Hamiltonian formulation with exact conservation
- Port-Hamiltonian framework
- Production-ready for realistic tokamak parameters

**Prerequisites:**
- v1.2 Poisson bracket implementation ✅
- Elsässer formulation derivation for toroidal geometry
- Wu time transformation implementation

**Target:** 2026 Q3-Q4

---

## What v1.1 CAN Do (Safe Use Cases)

✅ **Framework validation:**
- Toroidal operators work correctly (∇·B < 1e-10)
- Symplectic integrator integrates (even if underutilized)
- Code structure suitable for RL wrapping

✅ **Near-equilibrium evolution:**
- Small perturbations around stable equilibrium
- Short time evolution (< 100 Alfvén times)
- Qualitative behavior exploration

✅ **RL integration (M4):**
- Environment interface design
- Observation/action space prototyping
- Preliminary policy training (with caveats)

---

## What v1.1 CANNOT Do (Unsafe Use Cases)

❌ **Realistic tearing mode simulation:**
- Non-linear island growth
- Long-time evolution
- Quantitative growth rate comparison

❌ **Energy conservation validation:**
- Current tests are trivial (constant equilibrium)
- Non-trivial cases blow up (exponential growth)

❌ **Production deployment:**
- Physics incomplete (no Poisson bracket)
- Numerical stability not guaranteed
- Benchmarking against literature requires v1.2+

---

## Documentation Strategy

### Where to Find Information

**Design decisions:**
- `LIMITATIONS.md` (this file) - Known issues and roadmap

**Physics correctness:**
- `docs/v1.1/design/v1.1-toroidal-symplectic-design.md` - Original design
- `docs/v1.1/design/INTEGRATOR_EQUATION_MISMATCH.md` - Technical analysis (to be written)

**Test coverage:**
- `tests/test_step_3_1_solver.py` - Framework tests (5/5 PASS)
- `tests/test_step_3_2_div_B.py` - ∇·B validation (3/3 PASS)
- `tests/test_step_3_3_physics.py` - Energy/cylindrical limit (2/2 PASS, but trivial)

**Future work:**
- `docs/v1.2/POISSON_BRACKET_DESIGN.md` - v1.2 design (to be written)
- `docs/v2.0/ELSASSER_FORMULATION.md` - v2.0 design (to be written)

---

## Changelog

**2026-03-17:**
- M3 Step 3.3 completed
- Integrator-equation mismatch identified
- YZ decision: Accept minimal v1.1, defer Poisson bracket to v1.2
- This document created

---

## Acknowledgments

**Discovered by:** Sub-agent (session 11c0dd7e-..., M3 Step 3.3)  
**Analyzed by:** 小P ⚛️  
**Decision by:** YZ  
**Documented by:** 小P ⚛️

---

_"Know thy limitations, document thy roadmap, deliver thy milestones."_

---

## M3 Step 3.4: Benchmark Decision

### Design Doc Requirement

**Original Plan (Part 3, Step 3.4):**
- Benchmark v1.1 (toroidal) vs v1.0 (cylindrical)
- Compare: growth rate, island width, energy drift, compute time
- Validate that v1.1 improvements are measurable

### Decision: Skip Step 3.4 in v1.1 ❌

**Date:** 2026-03-17  
**Decision by:** YZ

**Rationale:**

1. **v1.1 Lacks Realistic Physics:**
   - Current implementation: pure diffusion (∂ψ/∂t = -η*∇²ψ)
   - Missing: Poisson bracket terms needed for tearing mode evolution
   - Cannot simulate realistic island growth or saturation
   - Benchmark comparison would be meaningless

2. **v1.0 Uses Full MHD:**
   - v1.0 includes Poisson bracket [ψ, φ]
   - v1.0 can simulate realistic tearing mode
   - v1.1 (minimal) vs v1.0 (full) is apples-to-oranges comparison

3. **Better Done in v1.2:**
   - v1.2 will add Poisson bracket → comparable physics
   - v1.2 vs v1.0 benchmark will be meaningful
   - Can validate toroidal effects + symplectic improvements simultaneously

**Alternative Completed:**
- M3 focuses on framework validation (Steps 3.1, 3.2, 3.3) ✅
- Physics validation deferred to v1.2
- Proceed to M4 (RL Integration) without blocking

### Impact on Deliverables

**v1.1 M3 Completion Criteria (Updated):**
- ✅ Step 3.1: Solver implementation (5/5 tests PASS)
- ✅ Step 3.2: ∇·B validation (3/3 tests PASS, GATE 0)
- ✅ Step 3.3: Energy conservation + cylindrical limit (2/2 tests PASS)
- ❌ Step 3.4: Benchmark vs v1.0 (SKIPPED, deferred to v1.2)

**Rationale for "Complete" Status:**
- Framework validation achieved (core M3 goal)
- Limitations documented (LIMITATIONS.md)
- Ready for M4 RL integration
- Physics completeness is v1.2 scope, not v1.1

---

**Updated:** 2026-03-17  
**Approved by:** YZ
