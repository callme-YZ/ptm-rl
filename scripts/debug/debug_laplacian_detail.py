import numpy as np
from src.pytokmhd.solver.mhd_equations import laplacian_cylindrical

Nr, Nz = 64, 128
r = np.linspace(0.1, 2.0, Nr)
z = np.linspace(-2.0, 2.0, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')
dr = r[1] - r[0]
dz = z[1] - z[0]

f_test = R**2
lap_f = laplacian_cylindrical(f_test, dr, dz, R)

print("Laplacian of r²:")
print(f"Expected: 4.0 everywhere")
print(f"\nActual values at different radii:")
for i in [0, Nr//4, Nr//2, 3*Nr//4, Nr-1]:
    print(f"  r={r[i]:.3f}: lap_f = {lap_f[i, Nz//2]:.6f} (mid-z)")

print(f"\nStatistics:")
print(f"  min = {np.min(lap_f):.6f}")
print(f"  max = {np.max(lap_f):.6f}")
print(f"  mean = {np.mean(lap_f):.6f}")
print(f"  std = {np.std(lap_f):.6f}")

# Check where it's wrong
wrong_mask = np.abs(lap_f - 4.0) > 0.1
n_wrong = np.sum(wrong_mask)
print(f"\nPoints with |lap_f - 4| > 0.1: {n_wrong}/{Nr*Nz}")

# Check boundaries specifically
print(f"\nBoundary values:")
print(f"  r=r[0]={r[0]:.3f}: lap_f[0,:] = {lap_f[0, 0]:.6f} .. {lap_f[0, -1]:.6f}")
print(f"  r=r[-1]={r[-1]:.3f}: lap_f[-1,:] = {lap_f[-1, 0]:.6f} .. {lap_f[-1, -1]:.6f}")
