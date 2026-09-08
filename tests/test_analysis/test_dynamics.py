"""Tests for LAMMPS dynamics calculations (segment ACF, Rouse mode ACF, ISF).

These functions consume a `.npz` trajectory file rather than a DataFrame, so
each test builds a small synthetic trajectory on disk via `np.savez`. Coverage
here is limited to smoke-testing the deprecated pre-1.2.0 aliases
(`calculate_segment_acf`, `calculate_rouse_mode_acf`, `calculate_isf`) against
their renamed `_from_trajectory` replacements — full correctness coverage of
the underlying physics is a separate, pre-existing gap.
"""

import numpy as np
import pytest

from lada.analysis import (
    calculate_isf,
    calculate_isf_from_trajectory,
    calculate_rouse_mode_acf,
    calculate_rouse_mode_acf_from_trajectory,
    calculate_segment_acf,
    calculate_segment_acf_from_trajectory,
)


@pytest.fixture
def trajectory_file(tmp_path):
    """A tiny synthetic trajectory: 6 frames, 4 atoms, drifting coordinates."""
    rng = np.random.default_rng(0)
    n_frames, n_atoms = 6, 4
    coords = np.cumsum(rng.normal(0, 0.1, size=(n_frames, n_atoms, 3)), axis=0)

    path = tmp_path / "trajectory.npz"
    np.savez(path, coords=coords)
    return str(path)


class TestCalculateSegmentAcfDeprecation:
    """calculate_segment_acf (deprecated) vs calculate_segment_acf_from_trajectory."""

    def test_old_name_warns(self, trajectory_file):
        segment_pairs = np.array([[0, 3], [1, 2]])
        with pytest.warns(DeprecationWarning, match="calculate_segment_acf_from_trajectory"):
            calculate_segment_acf(trajectory_file, segment_pairs, time_per_frame=0.5)

    def test_old_name_matches_new_name(self, trajectory_file):
        segment_pairs = np.array([[0, 3], [1, 2]])
        with pytest.warns(DeprecationWarning):
            old_result = calculate_segment_acf(trajectory_file, segment_pairs, time_per_frame=0.5)
        new_result = calculate_segment_acf_from_trajectory(
            trajectory_file, segment_pairs, time_per_frame=0.5
        )
        np.testing.assert_array_equal(old_result, new_result)


class TestCalculateRouseModeAcfDeprecation:
    """calculate_rouse_mode_acf (deprecated) vs calculate_rouse_mode_acf_from_trajectory."""

    def test_old_name_warns(self, trajectory_file):
        chain_indices = np.array([[0, 1, 2, 3]])
        with pytest.warns(DeprecationWarning, match="calculate_rouse_mode_acf_from_trajectory"):
            calculate_rouse_mode_acf(trajectory_file, chain_indices, p=1, time_per_frame=0.5)

    def test_old_name_matches_new_name(self, trajectory_file):
        chain_indices = np.array([[0, 1, 2, 3]])
        with pytest.warns(DeprecationWarning):
            old_result = calculate_rouse_mode_acf(
                trajectory_file, chain_indices, p=1, time_per_frame=0.5
            )
        new_result = calculate_rouse_mode_acf_from_trajectory(
            trajectory_file, chain_indices, p=1, time_per_frame=0.5
        )
        np.testing.assert_array_equal(old_result, new_result)


class TestCalculateIsfDeprecation:
    """calculate_isf (deprecated) vs calculate_isf_from_trajectory."""

    def test_old_name_warns(self, trajectory_file):
        with pytest.warns(DeprecationWarning, match="calculate_isf_from_trajectory"):
            calculate_isf(trajectory_file, time_per_frame=0.5, q_magnitude=7.0, n_vectors=8)

    def test_old_name_matches_new_name(self, trajectory_file):
        with pytest.warns(DeprecationWarning):
            old_result = calculate_isf(
                trajectory_file, time_per_frame=0.5, q_magnitude=7.0, n_vectors=8
            )
        new_result = calculate_isf_from_trajectory(
            trajectory_file, time_per_frame=0.5, q_magnitude=7.0, n_vectors=8
        )
        np.testing.assert_array_equal(old_result, new_result)
