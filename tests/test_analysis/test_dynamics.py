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


def assert_well_formed_progress(calls, total):
    """Shared shape checks for a recorded (current, total) call list."""
    assert calls, "progress_callback was never called"
    assert all(t == total for _, t in calls)
    currents = [c for c, _ in calls]
    increasing = all(b > a for a, b in zip(currents, currents[1:], strict=False))
    assert increasing, "not strictly increasing"
    assert currents[0] >= 1
    assert calls[-1] == (total, total)


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


class TestProgressCallback:
    """progress_callback: omitted still works, called (current, total) with current strictly
    increasing and 1-based, final call always (total, total)."""

    def test_omitting_the_callback_still_works(self, trajectory_file):
        segment_pairs = np.array([[0, 3], [1, 2]])
        result = calculate_segment_acf_from_trajectory(trajectory_file, segment_pairs, 0.5)
        assert result.shape == (6, 2)

    def test_segment_acf_reports_one_call_per_frame(self, trajectory_file):
        segment_pairs = np.array([[0, 3], [1, 2]])
        calls = []
        calculate_segment_acf_from_trajectory(
            trajectory_file, segment_pairs, 0.5, progress_callback=lambda c, t: calls.append((c, t))
        )
        assert_well_formed_progress(calls, total=6)  # n_frames from the fixture

    def test_rouse_mode_reports_one_call_per_frame(self, trajectory_file):
        chain_indices = np.array([[0, 1, 2, 3]])
        calls = []
        calculate_rouse_mode_acf_from_trajectory(
            trajectory_file,
            chain_indices,
            p=1,
            time_per_frame=0.5,
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        assert_well_formed_progress(calls, total=6)

    def test_isf_reports_one_call_per_q_vector_and_lag_not_just_per_q_vector(self, trajectory_file):
        """The inner (n_frames) loop must be instrumented, not just the outer n_vectors loop —
        otherwise progress would jump in 3 coarse steps instead of reflecting ISF's real cost."""
        n_vectors = 3
        n_frames = 6  # from the fixture
        calls = []
        calculate_isf_from_trajectory(
            trajectory_file,
            time_per_frame=0.5,
            q_magnitude=1.0,
            n_vectors=n_vectors,
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        total = n_vectors * n_frames
        assert_well_formed_progress(calls, total=total)
        assert len(calls) == total


class TestRouseModeRange:
    """The p range error used to say "strictly between 0 and N-1" for an inclusive range (D11)."""

    @pytest.mark.parametrize("p", [-1, 4])
    def test_out_of_range_p_names_the_inclusive_range(self, trajectory_file, p):
        chain_indices = np.array([[0, 1, 2, 3]])
        with pytest.raises(ValueError) as excinfo:
            calculate_rouse_mode_acf_from_trajectory(
                trajectory_file, chain_indices, p=p, time_per_frame=0.5
            )
        message = str(excinfo.value)
        assert f"p={p}" in message
        assert "from 0 to 3 inclusive" in message
        assert "strictly" not in message

    @pytest.mark.parametrize("p", [0, 3])
    def test_both_ends_of_the_range_are_accepted(self, trajectory_file, p):
        chain_indices = np.array([[0, 1, 2, 3]])
        result = calculate_rouse_mode_acf_from_trajectory(
            trajectory_file, chain_indices, p=p, time_per_frame=0.5
        )
        assert result.shape == (6, 2)
