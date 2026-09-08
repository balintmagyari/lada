from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_SHEAR_COLS = ("ACF_Sxy", "ACF_Sxz", "ACF_Syz")
_NORMAL_COLS_CANONICAL = ("ACF_Nxy", "ACF_Nxz", "ACF_Nyz")


def _validate_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    """Raise KeyError listing any columns from *cols* absent in *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following expected columns are missing from the DataFrame: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calc_stress_relaxation(
    df: pd.DataFrame, volume: float, temperature: float, lag_col: str = "lag_time", kB: float = 1.0
) -> pd.DataFrame:
    """
    Calculate the stress relaxation modulus G(t) from a LAMMPS ACF DataFrame
    using the Green-Kubo (GK) and Full Stress Relaxation (FSR) methods.

    Assumes LJ units (k_B = 1), so the prefactor is simply V / T.

    Parameters
    ----------
    df : pd.DataFrame
        ACF DataFrame as returned by ``read_lammps_acf``.  Must contain
        columns for the shear ACFs (``ACF_Sxy``, ``ACF_Sxz``, ``ACF_Syz``)
        and the normal stress difference ACFs (``ACF_Nxy``, ``ACF_Nxz``,
        ``ACF_Nyz``). Column names are matched exactly — no typo tolerance.
    volume : float
        System volume in LJ units (length³).
    temperature : float
        System temperature in LJ units (energy).
    lag_col : str, optional
        Name of the lag-time column in *df* (default: ``'lag_time'``).
    kB : float, optional
        Boltzmann constant. Default is ``1.0`` for LJ reduced units. Set to
        the physical value (e.g. ``1.380649e-23``) when using SI units.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns:
          - ``lag_time``  : float - physical lag time (copied from *df*)
          - ``G_GK``      : float - Green-Kubo relaxation modulus
          - ``G_FSR``     : float - Full stress relaxation modulus

    Raises
    ------
    KeyError
        If any required ACF column is absent from *df*.
    ValueError
        If *volume* or *temperature* are non-positive, or if *lag_col* is
        not found in *df*.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if volume <= 0:
        raise ValueError(f"volume must be positive, got {volume}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if lag_col not in df.columns:
        raise ValueError(
            f"Lag-time column '{lag_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    _validate_columns(df, _SHEAR_COLS)
    _validate_columns(df, _NORMAL_COLS_CANONICAL)

    # ------------------------------------------------------------------
    # Prefactor  (V / k_B T),  k_B = 1 in LJ units
    # ------------------------------------------------------------------
    prefactor = volume / (temperature * kB)

    # ------------------------------------------------------------------
    # Green-Kubo
    # G_GK(t) = (V/T*kB) * (1/3) * ( ACF_Sxy + ACF_Sxz + ACF_Syz )
    # ------------------------------------------------------------------
    shear_mean = (df["ACF_Sxy"] + df["ACF_Sxz"] + df["ACF_Syz"]) / 3.0

    G_GK = prefactor * shear_mean

    # ------------------------------------------------------------------
    # Full Stress Relaxation
    # G_FSR(t) = (V/kB*T) * (1/6) * [   ACF_Sxy  + ACF_Sxz  + ACF_Syz
    #                               + (1/4)*ACF_Nxy
    #                               + (1/4)*ACF_Nxz
    #                               + (1/4)*ACF_Nyz ]
    # ------------------------------------------------------------------
    fsr_sum = (
        df["ACF_Sxy"]
        + df["ACF_Sxz"]
        + df["ACF_Syz"]
        + 0.25 * df["ACF_Nxy"]
        + 0.25 * df["ACF_Nxz"]
        + 0.25 * df["ACF_Nyz"]
    )

    G_FSR = prefactor * fsr_sum / 6.0

    # ------------------------------------------------------------------
    # Assemble output DataFrame
    # ------------------------------------------------------------------
    result = pd.DataFrame(
        {
            "lag_time": df[lag_col].values,
            "G_GK": G_GK.values,
            "G_FSR": G_FSR.values,
        }
    )

    return result


# 1. Overload for when return_fit is False (or omitted)
@overload
def calc_dynamic_moduli_prony(
    df: pd.DataFrame,
    method: Literal["GK", "FSR", "both"] = "GK",
    t_min: float = 1.0,
    t_cutoff: float | None = None,
    n_modes: int = 50,
    n_omega: int = 200,
    omega_min: float | None = None,
    omega_max: float | None = None,
    return_fit: bool = False,
) -> pd.DataFrame: ...


# 2. Overload for when return_fit is True
@overload
def calc_dynamic_moduli_prony(
    df: pd.DataFrame,
    method: Literal["GK", "FSR", "both"] = "GK",
    t_min: float = 1.0,
    t_cutoff: float | None = None,
    n_modes: int = 50,
    n_omega: int = 200,
    omega_min: float | None = None,
    omega_max: float | None = None,
    return_fit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def calc_dynamic_moduli_prony(
    df: pd.DataFrame,
    method: Literal["GK", "FSR", "both"] = "GK",
    t_min: float = 1.0,
    t_cutoff: float | None = None,
    n_modes: int = 50,
    n_omega: int = 200,
    omega_min: float | None = None,
    omega_max: float | None = None,
    return_fit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute G'(ω) and G''(ω) via a Prony-series fit of G(t) followed by an
    exact analytical Fourier-Laplace transform.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame as returned by :func:`calc_stress_relaxation`.  Must contain
        columns ``lag_time``, ``G_GK``, and ``G_FSR``.
    method : {"GK", "FSR", "both"}, default "GK"
        Which G(t) column(s) to fit and transform.
    t_min : float, default 1.0
        Lower bound of the fitting window.
    t_cutoff : float | None, default None
        Upper bound of the fitting window.
    n_modes : int, default 50
        Number of Maxwell modes.
    n_omega : int, default 200
        Number of angular frequency points in the output.
    omega_min : float | None, default None
        Optional manual lower bound for the frequency grid.
    omega_max : float | None, default None
        Optional manual upper bound for the frequency grid.
    return_fit : bool, default False
        If True, returns a tuple of (df_moduli, df_time), where df_time contains
        the raw G(t) data and the evaluated Prony fit for visual validation.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        If return_fit is False, returns df_moduli.
        If return_fit is True, returns (df_moduli, df_time).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    required = {"lag_time", "G_GK", "G_FSR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Expected output from calc_stress_relaxation."
        )
    if method not in ("GK", "FSR", "both"):
        raise ValueError(f"method must be 'GK', 'FSR', or 'both', got '{method}'.")
    if n_modes < 1:
        raise ValueError(f"n_modes must be a positive integer, got {n_modes}.")

    t_all = df["lag_time"].to_numpy(dtype=float)
    if len(t_all) < 2:
        raise ValueError("DataFrame must contain at least two time points.")
    if np.any(np.diff(t_all) <= 0):
        raise ValueError("lag_time values must be strictly increasing.")

    # ------------------------------------------------------------------
    # Time window [t_min, t_end]
    # ------------------------------------------------------------------
    t_end = t_all[-1] if t_cutoff is None else t_cutoff

    if t_cutoff is not None and t_cutoff > t_all[-1]:
        raise ValueError(
            f"t_cutoff ({t_cutoff}) exceeds the last lag_time ({t_all[-1]}). "
            f"Use None to use the full range."
        )
    if t_min >= t_end:
        raise ValueError(f"t_min ({t_min}) must be less than the upper time limit ({t_end}).")

    mask = (t_all >= t_min) & (t_all <= t_end)
    n_fit = int(mask.sum())
    if n_fit < 2:
        raise ValueError(f"The time window [{t_min}, {t_end}] leaves fewer than two data points.")
    if n_fit < n_modes:
        raise ValueError(
            f"The fitting window contains only {n_fit} data points but n_modes={n_modes}. "
            f"Reduce n_modes or widen the time window."
        )

    t_fit = t_all[mask]
    df_fit = df.iloc[mask]

    # ------------------------------------------------------------------
    # Prony mode time constants: log-spaced across the fit window
    # ------------------------------------------------------------------
    tau = np.logspace(np.log10(t_fit[0]), np.log10(t_fit[-1]), n_modes)

    # Design matrix: A[j, i] = exp(-t_fit[j] / tau[i])
    A = np.exp(-t_fit[:, None] / tau[None, :])  # (n_fit, n_modes)

    # ------------------------------------------------------------------
    # Output omega grid
    # If explicit bounds are provided, use them. Otherwise, default to
    # physics-based bounds derived from the simulation time limits.
    # ------------------------------------------------------------------
    if omega_min is None:
        omega_min = 2 * np.pi / (tau[-1] * 10.0)

    if omega_max is None:
        omega_max = np.pi / t_fit[0]

    assert omega_min is not None, "omega_min failed to initialize."
    assert omega_max is not None, "omega_max failed to initialize."

    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_omega)

    # ------------------------------------------------------------------
    # Inner function: NNLS fit + Analytical Calculation + Fit Evaluation
    # ------------------------------------------------------------------
    def _fit_and_transform(g_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Perform NNLS fit
        G_i, _ = nnls(A, g_fit)

        # Evaluate the time-domain fit directly using the design matrix A
        g_fit_eval = A @ G_i

        # Filter out inactive modes for computational efficiency
        active_mask = G_i > 0
        g_active = G_i[active_mask]
        tau_active = tau[active_mask]

        w = omega[:, None]
        t = tau_active[None, :]
        g_m = g_active[None, :]

        wt = w * t
        denominator = 1.0 + wt**2

        g_prime = np.sum((g_m * wt**2) / denominator, axis=1)
        g_dprime = np.sum((g_m * wt) / denominator, axis=1)

        return g_prime, g_dprime, g_fit_eval

    # ------------------------------------------------------------------
    # Assemble output DataFrames
    # ------------------------------------------------------------------
    out_freq: dict[str, np.ndarray] = {"omega": omega}
    out_time: dict[str, np.ndarray] = {"lag_time": t_fit}

    if method in ("GK", "both"):
        g_data = df_fit["G_GK"].to_numpy(dtype=float)
        gp, gdp, g_fit_eval = _fit_and_transform(g_data)

        suffix = "_GK" if method == "both" else ""
        out_freq[f"G_prime{suffix}"] = gp
        out_freq[f"G_dprime{suffix}"] = gdp

        out_time[f"G_data{suffix}"] = g_data
        out_time[f"G_fit{suffix}"] = g_fit_eval

    if method in ("FSR", "both"):
        g_data = df_fit["G_FSR"].to_numpy(dtype=float)
        gp, gdp, g_fit_eval = _fit_and_transform(g_data)

        suffix = "_FSR" if method == "both" else ""
        out_freq[f"G_prime{suffix}"] = gp
        out_freq[f"G_dprime{suffix}"] = gdp

        out_time[f"G_data{suffix}"] = g_data
        out_time[f"G_fit{suffix}"] = g_fit_eval

    df_moduli = pd.DataFrame(out_freq)

    if return_fit:
        df_time = pd.DataFrame(out_time)
        return df_moduli, df_time

    return df_moduli
