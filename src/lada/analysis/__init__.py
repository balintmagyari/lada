"""LaDa analysis module: Molecular dynamics calculation functions.

This module provides vectorized NumPy implementations of common MD analysis
calculations for LAMMPS simulation data, including radius of gyration, 
end-to-end distances, and dynamical properties (ACF, ISF, stress relaxation).
"""

from .calculations import (
    calculate_avg_rg_sq,
    calculate_avg_ree_sq,
    calculate_ree_vectors,
    calculate_segment_acf,
    calculate_rouse_mode_acf,
    calculate_isf,
    calc_stress_relaxation,
)

__all__ = [
    "calculate_avg_rg_sq",
    "calculate_avg_ree_sq",
    "calculate_ree_vectors",
    "calculate_segment_acf",
    "calculate_rouse_mode_acf",
    "calculate_isf",
    "calc_stress_relaxation",
]