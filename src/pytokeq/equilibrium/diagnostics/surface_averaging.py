#!/usr/bin/env python3
"""Flux Surface Averaging (FIXED)"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def find_flux_surface_contour(psi, R, Z, psi_target):
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    fig, ax = plt.subplots()
    cs = ax.contour(RR, ZZ, psi, levels=[psi_target])
    plt.close(fig)
    if len(cs.collections) == 0 or len(cs.collections[0].get_paths()) == 0:
        return np.array([]), np.array([])
    path = cs.collections[0].get_paths()[0]
    return path.vertices[:, 0], path.vertices[:, 1]

def surface_average(f, psi, R, Z, psi_target):
    R_surf, Z_surf = find_flux_surface_contour(psi, R, Z, psi_target)
    if len(R_surf) == 0:
        return 0.0
    interp = RegularGridInterpolator((R, Z), f.T, method='linear',
                                     bounds_error=False, fill_value=0)
    f_surf = interp(np.column_stack([R_surf, Z_surf]))
    # FIXED: Trapezoidal with midpoints
    R_c = np.append(R_surf, R_surf[0])
    Z_c = np.append(Z_surf, Z_surf[0])
    f_c = np.append(f_surf, f_surf[0])
    dl = np.sqrt(np.diff(R_c)**2 + np.diff(Z_c)**2)
    f_mid = 0.5 * (f_c[:-1] + f_c[1:])
    return np.sum(f_mid * dl) / (np.sum(dl) + 1e-12)

def compute_R_dpsi_dR_avg(psi, R, Z, psi_target):
    dpsi_dR = np.gradient(psi, R, axis=0)
    RR = np.meshgrid(R, Z, indexing='ij')[0]
    return surface_average(RR * dpsi_dR, psi, R, Z, psi_target)
