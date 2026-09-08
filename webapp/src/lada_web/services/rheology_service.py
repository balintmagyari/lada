"""Stress relaxation modulus G(t) and its Prony-fitted dynamic moduli, via `lada.analysis`."""

from pathlib import Path
from typing import Literal

import pandas as pd
from lada.analysis import calc_dynamic_moduli_prony, calc_stress_relaxation
from lada.parsers import read_lammps_acf


def stress_relaxation(path: Path, volume: float, temperature: float) -> pd.DataFrame:
    df = read_lammps_acf(str(path))
    return calc_stress_relaxation(df, volume, temperature)


def dynamic_moduli(
    path: Path,
    volume: float,
    temperature: float,
    *,
    method: Literal["GK", "FSR", "both"],
    t_min: float,
    t_cutoff: float | None,
    n_modes: int,
    n_omega: int,
    omega_min: float | None,
    omega_max: float | None,
) -> pd.DataFrame:
    g_t = stress_relaxation(path, volume, temperature)
    return calc_dynamic_moduli_prony(
        g_t,
        method=method,
        t_min=t_min,
        t_cutoff=t_cutoff,
        n_modes=n_modes,
        n_omega=n_omega,
        omega_min=omega_min,
        omega_max=omega_max,
    )
