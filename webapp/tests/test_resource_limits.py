"""Every bound is checked before the expensive step, never after it.

Two guardrails used to allocate first and refuse afterwards:

* the topology form built an `(n_chains, beads_per_chain)` array from two unbounded form
  integers, and only then compared the indices against the trajectory's atom count;
* the `.npz` inspector capped the frame count but not the atom count, so a few kilobytes of
  compressed archive could declare an array of hundreds of gigabytes.

The third case of the same shape — a request body the size middleware never measured — lives
in `test_body_limit.py`, next to the rest of the middleware's tests.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from numpy.lib import format as npformat

from lada_web.errors import TrajectoryTooLarge, TrajectoryTooLargeInMemory
from lada_web.services.dynamics_service import (
    inspect_trajectory,
    resolve_chain_indices,
    resolve_segment_pairs,
)

from .conftest import upload

SEGMENT_URL = "/api/dynamics/segment-acf"
MB = 1024 * 1024

# The fixture trajectory has 32 atoms, so any layout reaching past index 31 is out of range.
FIXTURE_ATOMS = 32


class Allocated(Exception):
    """Raised by the np.arange guard when an array is built before its bound is checked."""


@pytest.fixture
def no_oversized_arange(monkeypatch):
    """Make `np.arange(n)` raise instead of allocating when n is past the trajectory's end.

    A bound that is enforced arithmetically never reaches this; a bound enforced by
    inspecting the finished array does.
    """

    def install(limit: int) -> None:
        real = np.arange

        def guarded(*args, **kwargs):
            stop = args[0] if args else kwargs.get("stop", 0)
            if isinstance(stop, int) and stop > limit:
                raise Allocated(f"np.arange({stop}) ran before the bound was checked.")
            return real(*args, **kwargs)

        monkeypatch.setattr(np, "arange", guarded)

    return install


def declared_npz(path: Path, shape: tuple[int, int, int], dtype: str = "<f8") -> Path:
    """An .npz whose 'coords' header declares `shape` but carries none of the data.

    This is what a hostile (or merely truncated) `np.savez_compressed` archive looks like on
    the wire: a few hundred bytes that ask the server to materialise an enormous array.
    """
    header = io.BytesIO()
    npformat.write_array_header_1_0(
        header, {"descr": dtype, "fortran_order": False, "shape": shape}
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("coords.npy", header.getvalue())
    return path


# --- Topology: the block layout is bounded arithmetically ------------------


def test_uniform_chain_indices_are_bounded_before_they_are_built(no_oversized_arange):
    no_oversized_arange(FIXTURE_ATOMS)
    with pytest.raises(ValueError, match="out of range"):
        resolve_chain_indices(
            "uniform",
            n_chains=4000,
            beads_per_chain=4000,
            first_index=0,
            custom="",
            n_atoms=FIXTURE_ATOMS,
        )


def test_uniform_segment_pairs_are_bounded_before_they_are_built(no_oversized_arange):
    no_oversized_arange(FIXTURE_ATOMS)
    with pytest.raises(ValueError, match="out of range"):
        resolve_segment_pairs(
            "uniform",
            n_chains=4000,
            beads_per_chain=4000,
            first_index=0,
            custom="",
            n_atoms=FIXTURE_ATOMS,
        )


def test_uniform_segment_pairs_never_build_the_whole_block_matrix(no_oversized_arange):
    """Only the two end columns are needed, so nothing of size beads_per_chain is allocated."""
    no_oversized_arange(4)
    pairs = resolve_segment_pairs(
        "uniform", n_chains=4, beads_per_chain=8, first_index=0, custom="", n_atoms=32
    )
    assert pairs.tolist() == [[0, 7], [8, 15], [16, 23], [24, 31]]


def test_a_huge_uniform_topology_is_refused_without_allocating(
    client, data_dir, no_oversized_arange
):
    """4000 x 4000 int64 is 128 MB; the request must be refused before that is asked for."""
    no_oversized_arange(FIXTURE_ATOMS)
    form = {
        "mode": "uniform",
        "n_chains": "4000",
        "beads_per_chain": "4000",
        "first_index": "0",
        "time_per_frame": "1.0",
    }
    r = client.post(SEGMENT_URL, files=upload(data_dir / "sample_trajectory.npz"), data=form)
    assert r.status_code == 422, r.text
    assert "32 atoms" in r.json()["detail"]


# --- Trajectories: the decompressed size is bounded too --------------------


def test_inspect_trajectory_rejects_a_huge_declared_array(tmp_path, settings):
    path = declared_npz(tmp_path / "declared.npz", (100, 10_000_000, 3))
    assert path.stat().st_size < 4096, "the archive itself must be tiny"
    with pytest.raises(TrajectoryTooLargeInMemory) as excinfo:
        inspect_trajectory(
            path, settings.max_trajectory_frames, settings.max_trajectory_memory_bytes
        )
    assert "MB" in str(excinfo.value)


def test_the_frame_cap_still_wins_when_both_are_exceeded(tmp_path, settings):
    path = declared_npz(tmp_path / "long.npz", (10_000, 10_000_000, 3))
    with pytest.raises(TrajectoryTooLarge):
        inspect_trajectory(
            path, settings.max_trajectory_frames, settings.max_trajectory_memory_bytes
        )


def test_a_tiny_archive_declaring_a_huge_array_is_413(client, tmp_path, temp_root):
    path = declared_npz(tmp_path / "declared.npz", (100, 10_000_000, 3))
    form = {
        "mode": "uniform",
        "n_chains": "4",
        "beads_per_chain": "8",
        "first_index": "0",
        "time_per_frame": "1.0",
    }
    r = client.post(SEGMENT_URL, files=upload(path), data=form)
    assert r.status_code == 413, r.text
    assert "MB" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_the_memory_budget_counts_the_declared_dtype(tmp_path, settings):
    """float32 halves the footprint, so the same shape can pass where float64 fails."""
    shape = (100, 4_000, 3)
    wide = declared_npz(tmp_path / "f8.npz", shape, dtype="<f8")  # 9.6 MB
    narrow = declared_npz(tmp_path / "f4.npz", shape, dtype="<f4")  # 4.8 MB
    budget = 8 * MB
    with pytest.raises(TrajectoryTooLargeInMemory):
        inspect_trajectory(wide, settings.max_trajectory_frames, budget)
    assert inspect_trajectory(narrow, settings.max_trajectory_frames, budget) == (100, 4_000)


def test_a_trajectory_within_the_budget_still_runs(client, data_dir):
    form = {
        "mode": "uniform",
        "n_chains": "4",
        "beads_per_chain": "8",
        "first_index": "0",
        "time_per_frame": "1.0",
    }
    r = client.post(SEGMENT_URL, files=upload(data_dir / "sample_trajectory.npz"), data=form)
    assert r.status_code == 200, r.text
