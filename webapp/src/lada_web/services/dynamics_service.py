"""Resolve the chain topology, inspect an .npz trajectory, and run `lada.analysis`."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lada.analysis import (
    calculate_isf_from_trajectory,
    calculate_rouse_mode_acf_from_trajectory,
    calculate_segment_acf_from_trajectory,
)
from numpy.lib import format as npformat

from ..errors import TrajectoryTooLarge, TrajectoryTooLargeInMemory


def parse_custom_lines(text: str) -> list[list[int]]:
    """One chain per line: integers separated by commas and/or spaces. Blank and '#' lines skip."""
    chains: list[list[int]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = [t for t in line.replace(",", " ").split() if t]
        try:
            values = [int(t) for t in tokens]
        except ValueError:
            raise ValueError(
                f"Line {number}: expected whole numbers separated by commas or spaces, "
                f"got '{line}'."
            ) from None
        if len(values) < 2:
            raise ValueError(f"Line {number}: a chain needs at least two bead indices.")
        if min(values) < 0:
            raise ValueError(f"Line {number}: bead indices must be zero or positive.")
        chains.append(values)
    if not chains:
        raise ValueError("Custom topology is empty: enter one chain per line.")
    return chains


def _uniform_extent(
    n_chains: int | None, beads_per_chain: int | None, first_index: int, n_atoms: int
) -> tuple[int, int]:
    """Validate a uniform block layout with arithmetic only, allocating nothing.

    Both counts come straight from the form, so building the (n_chains, beads_per_chain)
    block matrix first and range-checking it afterwards lets 100000 x 100000 ask for 80 GB
    before anything can refuse it. The last index the layout would use is
    `first_index + n_chains * beads_per_chain - 1`, which needs no array at all.
    """
    if n_chains is None or beads_per_chain is None:
        raise ValueError(
            "Uniform topology needs both the number of chains and the beads per chain."
        )
    last = first_index + n_chains * beads_per_chain - 1
    if last >= n_atoms:
        raise ValueError(
            f"Bead index {last} is out of range: the trajectory has {n_atoms} atoms "
            f"(indices 0 to {n_atoms - 1})."
        )
    return n_chains, beads_per_chain


def _uniform_blocks(n_chains: int, beads_per_chain: int, first_index: int) -> np.ndarray:
    start = first_index + np.arange(n_chains)[:, None] * beads_per_chain
    return start + np.arange(beads_per_chain)[None, :]


def resolve_chain_indices(
    mode, *, n_chains, beads_per_chain, first_index, custom, n_atoms
) -> np.ndarray:
    if mode == "uniform":
        chains, beads = _uniform_extent(n_chains, beads_per_chain, first_index, n_atoms)
        return _uniform_blocks(chains, beads, first_index)
    chains_list = parse_custom_lines(custom)
    if len({len(c) for c in chains_list}) != 1:
        raise ValueError(
            "Every chain must list the same number of beads for a Rouse-mode analysis."
        )
    return np.array(chains_list, dtype=int)


def resolve_segment_pairs(
    mode, *, n_chains, beads_per_chain, first_index, custom, n_atoms
) -> np.ndarray:
    if mode == "uniform":
        chains, beads = _uniform_extent(n_chains, beads_per_chain, first_index, n_atoms)
        # Only each block's first and last bead are wanted, so build those two columns
        # directly instead of the whole (n_chains, beads_per_chain) matrix.
        starts = first_index + np.arange(chains) * beads
        return np.column_stack([starts, starts + beads - 1])
    return np.array([[c[0], c[-1]] for c in parse_custom_lines(custom)], dtype=int)


def inspect_trajectory(path: Path, max_frames: int, max_memory_bytes: int) -> tuple[int, int]:
    """Read the 'coords' array header without loading the data; enforce both caps.

    The header carries the shape and the dtype, so the decompressed footprint is known
    before `lada` materialises anything. Compression ratio is unbounded, so the upload size
    is no guide at all: a few megabytes of .npz can declare hundreds of gigabytes.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            if "coords.npy" not in archive.namelist():
                raise ValueError(
                    "The .npz file must contain an array named 'coords' with shape "
                    "(n_frames, n_atoms, 3)."
                )
            with archive.open("coords.npy") as fh:
                version = npformat.read_magic(fh)
                reader = (
                    npformat.read_array_header_1_0
                    if version == (1, 0)
                    else npformat.read_array_header_2_0
                )
                shape, _, dtype = reader(fh)
    except zipfile.BadZipFile:
        raise ValueError(
            "This is not a NumPy .npz archive. Save the trajectory with np.savez."
        ) from None
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"'coords' must have shape (n_frames, n_atoms, 3); this file has {shape}.")
    n_frames, n_atoms = int(shape[0]), int(shape[1])
    if n_frames > max_frames:
        raise TrajectoryTooLarge(n_frames, max_frames)
    needed = n_frames * n_atoms * 3 * dtype.itemsize
    if needed > max_memory_bytes:
        raise TrajectoryTooLargeInMemory(needed, max_memory_bytes)
    return n_frames, n_atoms


def check_indices(indices: np.ndarray, n_atoms: int) -> None:
    if indices.size and int(indices.max()) >= n_atoms:
        raise ValueError(
            f"Bead index {int(indices.max())} is out of range: the trajectory has {n_atoms} atoms "
            f"(indices 0 to {n_atoms - 1})."
        )


def _frame(result: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame({"lag_time": result[:, 0], name: result[:, 1]})


def segment_acf(path, pairs, time_per_frame, progress_callback=None) -> pd.DataFrame:
    return _frame(
        calculate_segment_acf_from_trajectory(
            str(path), pairs, time_per_frame, progress_callback=progress_callback
        ),
        "acf",
    )


def rouse_mode_acf(path, chains, p, time_per_frame, progress_callback=None) -> pd.DataFrame:
    return _frame(
        calculate_rouse_mode_acf_from_trajectory(
            str(path), chains, p, time_per_frame, progress_callback=progress_callback
        ),
        "acf",
    )


def isf(path, time_per_frame, q_magnitude, n_vectors, progress_callback=None) -> pd.DataFrame:
    return _frame(
        calculate_isf_from_trajectory(
            str(path), time_per_frame, q_magnitude, n_vectors, progress_callback=progress_callback
        ),
        "isf",
    )
