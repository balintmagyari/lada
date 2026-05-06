"""Tests for LAMMPS analysis calculations."""

import pytest
import numpy as np
import pandas as pd
from lada.analysis import (
    calculate_avg_rg_sq,
    calculate_avg_ree_sq,
    calculate_ree_vectors,
)


def create_synthetic_trajectory_df(num_atoms=40, num_timesteps=2, num_molecules=1):
    """Create a synthetic trajectory DataFrame for testing.

    Parameters
    ----------
    num_atoms : int
        Total number of atoms
    num_timesteps : int
        Number of timesteps
    num_molecules : int
        Number of molecules/chains

    Returns
    -------
    pd.DataFrame
        Synthetic trajectory data with columns:
        id, mol, type, xu, yu, zu, timestep
    """
    data = []

    atoms_per_mol = num_atoms // num_molecules

    for ts in range(num_timesteps):
        timestep = ts * 100
        for atom_id in range(1, num_atoms + 1):
            mol_id = (atom_id - 1) // atoms_per_mol + 1
            atom_type = (atom_id % 2) + 1

            # Generate unwrapped coordinates (with some variation per timestep)
            x = (atom_id % 10) + np.random.normal(0, 0.1) + ts * 0.01
            y = (atom_id % 5) + np.random.normal(0, 0.1) + ts * 0.02
            z = (atom_id % 3) + np.random.normal(0, 0.1) + ts * 0.015

            data.append({
                'id': atom_id,
                'mol': mol_id,
                'type': atom_type,
                'xu': x,
                'yu': y,
                'zu': z,
                'timestep': timestep,
            })

    return pd.DataFrame(data)


class TestCalculateAvgRgSq:
    """Tests for calculate_avg_rg_sq() function."""

    def test_rg_sq_with_dataframe(self):
        """Test Rg² calculation with single timestep DataFrame."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=1)

        rg_sq = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        assert rg_sq is not None
        assert isinstance(rg_sq, float)
        assert rg_sq > 0

    def test_rg_sq_with_numpy_array(self):
        """Test Rg² calculation with numpy array."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=1)
        arr = df.to_numpy()
        cols = df.columns.tolist()

        rg_sq = calculate_avg_rg_sq(
            arr,
            columns=cols,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        assert rg_sq is not None
        assert isinstance(rg_sq, float)
        assert rg_sq > 0

    def test_rg_sq_multiple_timesteps(self):
        """Test Rg² with multiple timesteps returns dict."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=3)

        result = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        # With multiple timesteps should return dict
        assert isinstance(result, dict)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result.values())
        assert all(v > 0 for v in result.values())

    def test_rg_sq_with_mass_weighting(self):
        """Test Rg² with mass column."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=1)
        df['mass'] = 1.0  # Add uniform mass

        rg_sq = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep',
            mass_col='mass'
        )

        assert rg_sq is not None
        assert isinstance(rg_sq, float)
        assert rg_sq > 0

    def test_rg_sq_missing_column_raises_error(self):
        """Test that missing required column raises ValueError."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=1)

        with pytest.raises(ValueError):
            calculate_avg_rg_sq(
                df,
                coord_cols=['x', 'y', 'z'],  # Wrong column names
                molecule_col='mol',
                timestep_col='timestep'
            )


class TestCalculateAvgReeSq:
    """Tests for calculate_avg_ree_sq() function."""

    def test_ree_sq_with_dataframe(self):
        """Test end-to-end distance² calculation."""
        df = create_synthetic_trajectory_df(
            num_atoms=40,
            num_timesteps=1,
            num_molecules=2
        )

        ree_sq = calculate_avg_ree_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        assert ree_sq is not None
        assert isinstance(ree_sq, float)
        assert ree_sq > 0

    def test_ree_sq_multiple_timesteps(self):
        """Test Ree² with multiple timesteps returns dict."""
        df = create_synthetic_trajectory_df(
            num_atoms=40,
            num_timesteps=3,
            num_molecules=2
        )

        result = calculate_avg_ree_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        assert isinstance(result, dict)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result.values())
        assert all(v > 0 for v in result.values())

    def test_ree_sq_values_positive(self):
        """Test that Ree² values are always positive."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=2)

        ree_sq = calculate_avg_ree_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        if isinstance(ree_sq, dict):
            assert all(v > 0 for v in ree_sq.values())
        else:
            assert ree_sq > 0


class TestCalculateReeVectors:
    """Tests for calculate_ree_vectors() function."""

    def test_ree_vectors_returns_array(self):
        """Test that ree vectors function returns proper DataFrame."""
        df = create_synthetic_trajectory_df(
            num_atoms=30,
            num_timesteps=1,
            num_molecules=1
        )

        vectors = calculate_ree_vectors(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol'
        )

        assert vectors is not None
        # Function returns a pandas DataFrame
        assert isinstance(vectors, pd.DataFrame)

    def test_ree_vectors_shape(self):
        """Test that ree vectors have correct shape."""
        df = create_synthetic_trajectory_df(
            num_atoms=30,
            num_timesteps=1,
            num_molecules=1
        )

        vectors = calculate_ree_vectors(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol'
        )

        # Should be DataFrame with dx, dy, dz columns
        assert isinstance(vectors, pd.DataFrame)
        assert 'dx' in vectors.columns
        assert 'dy' in vectors.columns
        assert 'dz' in vectors.columns

    def test_ree_vectors_multiple_molecules(self):
        """Test ree vectors with multiple molecules."""
        df = create_synthetic_trajectory_df(
            num_atoms=40,
            num_timesteps=1,
            num_molecules=2
        )

        vectors = calculate_ree_vectors(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol'
        )

        assert vectors is not None
        # Should have data for multiple molecules
        assert isinstance(vectors, pd.DataFrame)
        assert len(vectors) == 2  # One row per molecule


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_atom_calculation(self):
        """Test with minimum single atom."""
        df = create_synthetic_trajectory_df(num_atoms=1, num_timesteps=1)

        rg_sq = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        # Single atom should have Rg² = 0
        assert isinstance(rg_sq, float)

    def test_large_dataset(self):
        """Test with larger synthetic dataset."""
        df = create_synthetic_trajectory_df(
            num_atoms=1000,
            num_timesteps=5,
            num_molecules=25
        )

        rg_sq = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )

        assert isinstance(rg_sq, dict)
        assert len(rg_sq) == 5

    def test_numeric_stability(self):
        """Test with very small and very large coordinates."""
        df = create_synthetic_trajectory_df(num_atoms=20, num_timesteps=1)

        # Test with small coordinates
        df[['xu', 'yu', 'zu']] = df[['xu', 'yu', 'zu']] * 0.001
        rg_sq_small = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )
        assert isinstance(rg_sq_small, float)

        # Test with large coordinates
        df[['xu', 'yu', 'zu']] = df[['xu', 'yu', 'zu']] * 1000
        rg_sq_large = calculate_avg_rg_sq(
            df,
            coord_cols=['xu', 'yu', 'zu'],
            molecule_col='mol',
            timestep_col='timestep'
        )
        assert isinstance(rg_sq_large, float)



