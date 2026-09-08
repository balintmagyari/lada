"""Tests for LAMMPS rheology calculations (stress relaxation, dynamic moduli)."""

import numpy as np
import pandas as pd
import pytest

from lada.analysis import calc_dynamic_moduli_prony, calc_stress_relaxation


def make_acf_df(n=5, use_typo=False, drop_col=None):
    """Build a minimal valid ACF DataFrame for stress relaxation tests."""
    normal_yz_col = "ACF_yz" if use_typo else "ACF_Nyz"
    df = pd.DataFrame(
        {
            "lag_time": np.linspace(0, 1, n),
            "ACF_Sxy": np.ones(n),
            "ACF_Sxz": np.ones(n),
            "ACF_Syz": np.ones(n),
            "ACF_Nxy": np.ones(n),
            "ACF_Nxz": np.ones(n),
            normal_yz_col: np.ones(n),
        }
    )
    if drop_col:
        df = df.drop(columns=drop_col)
    return df


class TestCalcStressRelaxation:
    """Tests for calc_stress_relaxation()."""

    def test_returns_dataframe_with_correct_columns(self):
        result = calc_stress_relaxation(make_acf_df(), volume=10.0, temperature=1.0)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["lag_time", "G_GK", "G_FSR"]

    def test_output_length_matches_input(self):
        df = make_acf_df(n=8)
        result = calc_stress_relaxation(df, volume=10.0, temperature=1.0)
        assert len(result) == 8

    def test_g_gk_formula(self):
        # With all ACF values = 1 and V=10, T=1: G_GK = (10/1) * (3/3) = 10
        result = calc_stress_relaxation(make_acf_df(), volume=10.0, temperature=1.0)
        assert result["G_GK"].values == pytest.approx(10.0)

    def test_g_fsr_formula(self):
        # With all ACF values = 1 and V=10, T=1:
        # G_FSR = (10/1) * (1/6) * (3 + (1/4)*3) = 10 * 3.75/6 = 6.25
        result = calc_stress_relaxation(make_acf_df(), volume=10.0, temperature=1.0)
        assert result["G_FSR"].values == pytest.approx(6.25)

    def test_lag_time_column_is_preserved(self):
        df = make_acf_df(n=5)
        result = calc_stress_relaxation(df, volume=5.0, temperature=2.0)
        np.testing.assert_array_equal(result["lag_time"].values, df["lag_time"].values)

    def test_missing_shear_column_raises_key_error(self):
        df = make_acf_df(drop_col="ACF_Sxy")
        with pytest.raises(KeyError):
            calc_stress_relaxation(df, volume=10.0, temperature=1.0)

    def test_missing_normal_column_raises_key_error(self):
        df = make_acf_df(drop_col="ACF_Nyz")
        with pytest.raises(KeyError):
            calc_stress_relaxation(df, volume=10.0, temperature=1.0)

    def test_typo_column_name_raises_key_error(self):
        # The old 'ACF_yz' typo is no longer accepted — must use 'ACF_Nyz'
        df = make_acf_df(use_typo=True)
        with pytest.raises(KeyError):
            calc_stress_relaxation(df, volume=10.0, temperature=1.0)

    def test_non_positive_volume_raises_value_error(self):
        with pytest.raises(ValueError):
            calc_stress_relaxation(make_acf_df(), volume=0.0, temperature=1.0)

    def test_non_positive_temperature_raises_value_error(self):
        with pytest.raises(ValueError):
            calc_stress_relaxation(make_acf_df(), volume=10.0, temperature=-1.0)

    def test_missing_lag_col_raises_value_error(self):
        df = make_acf_df(drop_col="lag_time")
        with pytest.raises(ValueError):
            calc_stress_relaxation(df, volume=10.0, temperature=1.0)


def make_maxwell_df(G0=1.0, tau=1.0, t_max=20.0, n=2000, uniform=True, t_start=0.0):
    """Maxwell model G(t) = G0*exp(-t/tau) as a calc_stress_relaxation-style DataFrame."""
    if uniform:
        t = np.linspace(t_start, t_max, n)
    else:
        t = np.sort(
            np.concatenate([[t_start], np.random.default_rng(42).uniform(t_start, t_max, n - 1)])
        )
    G = G0 * np.exp(-t / tau)
    return pd.DataFrame({"lag_time": t, "G_GK": G, "G_FSR": G * 0.8})


class TestCalcDynamicModuliProny:
    """Tests for calc_dynamic_moduli_prony()."""

    def test_columns_method_gk(self):
        df = make_maxwell_df(t_max=20.0, n=500)
        result = calc_dynamic_moduli_prony(df, method="GK")
        assert list(result.columns) == ["omega", "G_prime", "G_dprime"]

    def test_columns_method_fsr(self):
        df = make_maxwell_df(t_max=20.0, n=500)
        result = calc_dynamic_moduli_prony(df, method="FSR")
        assert list(result.columns) == ["omega", "G_prime", "G_dprime"]

    def test_columns_method_both(self):
        df = make_maxwell_df(t_max=20.0, n=500)
        result = calc_dynamic_moduli_prony(df, method="both")
        assert list(result.columns) == [
            "omega",
            "G_prime_GK",
            "G_dprime_GK",
            "G_prime_FSR",
            "G_dprime_FSR",
        ]

    def test_n_omega_controls_output_length(self):
        df = make_maxwell_df(t_max=20.0, n=500)
        for n in (50, 100, 300):
            result = calc_dynamic_moduli_prony(df, n_omega=n)
            assert len(result) == n

    def test_maxwell_model_accuracy(self):
        """Prony fit of G(t)=exp(-t) should recover G' and G'' within 5%."""
        # Maxwell: G'(ω)=(ωτ)²/(1+(ωτ)²), G''(ω)=ωτ/(1+(ωτ)²) with G0=τ=1
        df = make_maxwell_df(G0=1.0, tau=1.0, t_max=20.0, n=500)
        result = calc_dynamic_moduli_prony(df, method="GK", n_modes=30, n_omega=80, t_min=0.1)

        omega = result["omega"].to_numpy()
        g_prime_analytical = (omega**2) / (1 + omega**2)
        g_dprime_analytical = omega / (1 + omega**2)

        # Test in the mid-frequency range well within the omega grid
        mid = (omega > 0.3) & (omega < 5.0)
        np.testing.assert_allclose(
            result["G_prime"].to_numpy()[mid], g_prime_analytical[mid], rtol=0.05
        )
        np.testing.assert_allclose(
            result["G_dprime"].to_numpy()[mid], g_dprime_analytical[mid], rtol=0.05
        )

    def test_missing_column_raises_key_error(self):
        df = make_maxwell_df().drop(columns="G_FSR")
        with pytest.raises(KeyError):
            calc_dynamic_moduli_prony(df)

    def test_invalid_method_raises_value_error(self):
        with pytest.raises(ValueError, match="method must be"):
            calc_dynamic_moduli_prony(make_maxwell_df(), method="invalid")

    def test_t_min_above_t_end_raises_value_error(self):
        df = make_maxwell_df(t_max=20.0, n=100)
        with pytest.raises(ValueError, match="must be less than the upper time limit"):
            calc_dynamic_moduli_prony(df, t_min=25.0)

    def test_n_modes_exceeds_window_raises_value_error(self):
        df = make_maxwell_df(t_max=20.0, n=30)
        with pytest.raises(ValueError, match="n_modes"):
            calc_dynamic_moduli_prony(df, n_modes=200)

    def test_t_cutoff_beyond_data_raises_value_error(self):
        df = make_maxwell_df(t_max=20.0, n=100)
        with pytest.raises(ValueError, match="exceeds the last lag_time"):
            calc_dynamic_moduli_prony(df, t_cutoff=50.0)
