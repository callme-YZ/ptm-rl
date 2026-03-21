# M4 Reward Function Design

**Project:** PTM-RL v1.1 - M4 RL Integration  
**Author:** 小P ⚛️  
**Date:** 2026-03-17

## Executive Summary

Reward function for toroidal MHD control task.

**Objective:** Maintain plasma equilibrium (minimize deviation)

## 1. Physics Objectives

### 1.1 Primary Goal: Equilibrium Maintenance

**Target state:** ψ ≈ ψ_eq (circular equilibrium)

**Metrics:**
- Energy: E ≈ E_eq
- Field: ||ψ - ψ_eq|| ≈ 0
- Rate: ||∂ψ/∂t|| ≈ 0

### 1.2 Secondary: Constraint Satisfaction

**Physics constraint:** ∇·B = 0

**Monitor:** max|∇·B| < 1e-6

## 2. Reward Formulation

### 2.1 Energy-Based (Recommended for v1.1)

```python
reward = -w_E * |E - E_eq| / E_eq  
         -w_B * (div_B_max / 1e-6)
         -w_A * |action|²

where:
  w_E = 1.0   # Energy weight
  w_B = 0.1   # Constraint weight  
  w_A = 0.01  # Action penalty
```

**Rationale:**
- ✅ Energy deviation directly measurable
- ✅ ∇·B penalizes physics violation
- ✅ Action penalty avoids excessive control

### 2.2 Field-Based (Alternative)

```python
reward = -w_psi * ||ψ - ψ_eq||₂
         -w_B * div_B_max
         -w_A * ||action||₂
```

**When to use:** If energy insufficient

### 2.3 Stability-Based (v1.2)

```python
reward = -w_rate * ||∂ψ/∂t||
         -w_growth * γ  # growth rate
```

**When to use:** Realistic tearing mode (v1.2+)

## 3. Reward Shaping

### 3.1 Dense vs Sparse

**v1.1 Choice: Dense**

```python
# Every step
reward = compute_reward(state)
```

**Why:** 
- Pure diffusion → slow evolution
- Sparse signals too weak
- Dense feedback helps learning

### 3.2 Terminal Reward

```python
# Episode end
if done:
    if reason == 'success':
        reward += 100
    elif reason == 'failure':
        reward -= 100
```

**Success:** Reached horizon with E_drift < 0.01  
**Failure:** ∇·B > threshold (physics violation)

## 4. Multi-Objective Balancing

### 4.1 Weight Tuning

**Initial values:**
```python
w_E = 1.0    # Energy (primary)
w_B = 0.1    # Constraint (important but rare violation)
w_A = 0.01   # Action (regularization)
```

**Tuning strategy:**
- Start with equal weights
- Increase w_E if energy control poor
- Increase w_B if ∇·B violations occur
- Adjust w_A for smoothness

### 4.2 Reward Scaling

**Target range:** [-1, 1] per step

**Scaling:**
```python
reward_scaled = reward / max_expected_reward
# max_expected_reward ≈ 10 (empirical)
```

## 5. Physics Validation

### 5.1 Does Reward Encourage Correct Behavior?

**✅ Energy minimization:**
- Low energy drift → high reward ✅
- Encourages equilibrium maintenance ✅

**✅ Constraint satisfaction:**
- Low ∇·B → higher reward ✅
- Physics correctness rewarded ✅

**✅ Control efficiency:**
- Small actions preferred (via w_A) ✅
- Avoids chattering ✅

### 5.2 Does it Avoid Trivial Solutions?

**Trivial solution 1: Zero action**

```python
action = 0 (always)
→ No control, system drifts
→ Energy drift increases
→ Reward decreases ❌
```

**Prevented by:** Energy term dominates

**Trivial solution 2: Maximum action**

```python
action = max (always)
→ Disrupts equilibrium
→ Energy increases
→ Action penalty large
→ Reward decreases ❌
```

**Prevented by:** Energy + action penalty

**✅ No trivial exploitation**

## 6. v1.1 Limitations

### 6.1 What Reward CAN Capture

**✅ Equilibrium deviation:**
- Energy drift measurable
- Field evolution observable

**✅ Numerical stability:**
- ∇·B constraint

### 6.2 What Reward CANNOT Capture

**❌ Realistic control objectives:**
- Island width (no islands in v1.1)
- Growth rate (no exponential growth)
- Mode suppression (no mode coupling)

**Impact:** 
- ✅ Framework works
- ❌ Control strategies not transferable to v1.2

## 7. Implementation

```python
class ToroidalMHDEnv(gym.Env):
    def __init__(self):
        self.w_E = 1.0
        self.w_B = 0.1
        self.w_A = 0.01
        
        # Reference equilibrium
        self.E_eq = self.compute_equilibrium_energy()
    
    def compute_reward(self, obs, action):
        # Energy term
        E_drift = obs['energy_drift']
        r_energy = -self.w_E * E_drift
        
        # Constraint term
        div_B = obs['div_B_max']
        r_constraint = -self.w_B * div_B
        
        # Action penalty
        action_norm = np.linalg.norm(action)
        r_action = -self.w_A * action_norm**2
        
        reward = r_energy + r_constraint + r_action
        
        return reward
```

## 8. Recommendations for 小A

### 8.1 Initial Implementation

**Use energy-based reward (simplest):**

```python
reward = -energy_drift - 0.1*div_B_max - 0.01*action²
```

**No normalization needed initially** (values already reasonable)

### 8.2 Logging for Tuning

**Track components separately:**

```python
info = {
    'reward_energy': r_energy,
    'reward_constraint': r_constraint,
    'reward_action': r_action,
    'reward_total': reward,
}
```

**Plot during training:**
- Which component dominates?
- Adjust weights accordingly

### 8.3 Failure Modes to Watch

**1. Reward always negative:**
- ✅ Normal (equilibrium → reward ≈ 0, deviation → negative)

**2. Reward not improving:**
- ⚠️ Check if RL can affect energy (action → solver working?)
- ⚠️ Check weight balance

**3. ∇·B violations:**
- 🚨 Physics bug, stop training
- 🚨 Report to 小P

## 9. v1.2 Extensions

### 9.1 Island Width Reward

**When tearing mode realistic:**

```python
reward = -w_island * island_width
         -w_growth * growth_rate
```

### 9.2 Adaptive Weights

**Curriculum learning:**

```python
# Early training: focus on stability
w_E = 1.0, w_B = 1.0, w_A = 0.1

# Late training: focus on performance
w_E = 10.0, w_B = 0.1, w_A = 0.01
```

---

**Status:** Complete ✅  
**Next:** Action space design
