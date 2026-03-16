#!/usr/bin/env python3
"""Test M3DC1 profile values"""

import sys
sys.path.insert(0, '..')

import numpy as np
from pytokeq.equilibrium.profiles.m3dc1_profile import M3DC1Profile

profile = M3DC1Profile()

psi_norm = np.linspace(0, 1, 11)

print("M3DC1 Profile Values:")
print("="*70)
header = f"{'psi_norm':>10} {'q':>12} {'pprime':>15} {'ffprime':>15} {'F':>12}"
print(header)
print("-"*70)

for pn in psi_norm:
    q = profile.q_profile(np.array([pn]))[0]
    pp = profile.pprime(np.array([pn]))[0]
    ffp = profile.ffprime(np.array([pn]))[0]
    F = profile.Fpol(np.array([pn]))[0]
    
    print(f"{pn:10.2f} {q:12.4f} {pp:15.6e} {ffp:15.6e} {F:12.4f}")

print("="*70)

# Check source term magnitude
print(f"\nGrad-Shafranov source term: -μ₀R²p' - FF'")
print(f"\nFor typical R ≈ 1.5 m:")

R = 1.5
MU0 = 4 * np.pi * 1e-7

for pn in [0.0, 0.5, 1.0]:
    pp = profile.pprime(np.array([pn]))[0]
    ffp = profile.ffprime(np.array([pn]))[0]
    
    source = -MU0 * R**2 * pp - ffp
    
    print(f"  psi_norm={pn:.1f}: source = {source:.6e}")

print(f"\nNote: If source ≈ 0, equilibrium will be very weak!")
