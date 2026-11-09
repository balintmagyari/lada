"""Resolve the chain topology, inspect an .npz trajectory, and run `lada.analysis`."""

import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lada.analysis import (
    ChainTopology,
    calculate_isf_from_trajectory,
    calculate_rouse_mode_acf_from_trajectory,
    calculate_segment_acf_from_trajectory,
    chains_from_data_file,
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
        # "0,0" is not a chain but a zero-length vector, whose autocorrelation is 0/0: without
        # this check every row of the result came back empty, with nothing saying why.
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ValueError(
                f"Line {number}: bead {repeated[0]} is listed more than once. Every bead of a "
                "chain must be a different atom; for head,tail the two ends must differ."
            )
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
    covered = n_chains * beads_per_chain
    last = first_index + covered - 1
    if last >= n_atoms:
        raise ValueError(
            f"Bead index {last} is out of range: the trajectory has {n_atoms} atoms "
            f"(indices 0 to {n_atoms - 1})."
        )
    # Covering only part of the file is refused rather than analysed quietly: a mistyped chain
    # count would otherwise return a plausible result computed from a fraction of the system,
    # with nothing on screen to say so. A file that genuinely holds non-chain particles is what
    # the custom topology mode is for.
    if first_index + covered != n_atoms:
        raise ValueError(
            f"This topology covers {covered} atoms starting at index {first_index}, but the "
            f"trajectory has {n_atoms}. Chains of equal length must account for every atom from "
            "the first bead onwards — check the number of chains, the beads per chain and the "
            "first index, or use the custom topology mode to list the chains by hand."
        )
    return n_chains, beads_per_chain


def _uniform_blocks(n_chains: int, beads_per_chain: int, first_index: int) -> np.ndarray:
    start = first_index + np.arange(n_chains)[:, None] * beads_per_chain
    return start + np.arange(beads_per_chain)[None, :]


NO_TOPOLOGY_FILE = (
    "Choose a LAMMPS data file to read the chain topology from, or switch to one of the manual "
    "topology modes."
)


def read_topology(path: Path) -> ChainTopology:
    """Chains as positions into a `coords` array, read from an uploaded data file."""
    return chains_from_data_file(str(path))


def _checked_topology(topology: ChainTopology | None, n_atoms: int) -> ChainTopology:
    """The uploaded topology, once it is known to describe this trajectory.

    A `.npz` holds no atom ids, so a chain's beads are found by ranking the data file's ids: the
    i-th smallest id is position i. That ranking only means anything if both files hold the same
    atoms, and nothing else in either file can be used to line them up, so a count that differs
    is refused rather than trimmed or padded.
    """
    if topology is None:
        raise ValueError(NO_TOPOLOGY_FILE)
    if topology.n_atoms != n_atoms:
        raise ValueError(
            f"The data file describes {topology.n_atoms} atoms but the trajectory holds "
            f"{n_atoms}. Chains are matched to the trajectory by atom-id order, so the two "
            "files have to describe the same system."
        )
    return topology


def topology_summary(topology: ChainTopology) -> str:
    """One line for the result's metadata: what was read, and from which section."""
    distinct = set(topology.lengths)
    count = len(topology.chains)
    chains = "1 chain" if count == 1 else f"{count} chains"
    beads = f"{distinct.pop()} beads" if len(distinct) == 1 else "varying lengths"
    source = "molecule ids" if topology.source == "molecule" else "the bond graph"
    return f"{chains} of {beads}, from {source}"


def topology_stats(topology: ChainTopology) -> dict:
    """Everything the preview shows, including the spread of chain lengths.

    A bare chain count hides the case that matters: molecule ids in a solvated system put every
    water and every counterion in its own "chain", and a count alone would look like success.
    """
    lengths = sorted(topology.lengths)
    distinct = set(lengths)
    common = Counter(lengths).most_common(3)
    return {
        "n_atoms": topology.n_atoms,
        "n_chains": len(lengths),
        "beads_per_chain": lengths[0] if len(distinct) == 1 else None,
        "shortest": lengths[0],
        "longest": lengths[-1],
        "median": int(np.median(lengths)),
        "lengths": [{"beads": beads, "chains": count} for beads, count in common],
        "source": topology.source,
        "summary": topology_summary(topology),
        "notes": topology.notes,
    }


def _example_chain(pair: list[int]) -> str:
    """A full bead list spelling out a head,tail pair, shortened when the chain is long."""
    head, tail = pair
    step = 1 if tail > head else -1
    beads = list(range(head, tail + step, step))
    if len(beads) > 6:
        return f"{beads[0]} {beads[1]} {beads[2]} ... {beads[-1]}"
    return " ".join(str(b) for b in beads)


def resolve_chain_indices(
    mode, *, n_chains, beads_per_chain, first_index, custom, n_atoms, topology=None
) -> np.ndarray:
    if mode == "uniform":
        chains, beads = _uniform_extent(n_chains, beads_per_chain, first_index, n_atoms)
        return _uniform_blocks(chains, beads, first_index)
    if mode == "data_file":
        return _checked_topology(topology, n_atoms).chain_indices()
    chains_list = parse_custom_lines(custom)
    # Rouse modes are sums over every bead of a chain, so a line naming only the two ends
    # ("0,7") would be analysed as a two-bead chain: p = 1 of that is the end-to-end vector, and
    # the page returned the segment ACF under a Rouse-mode title. Two indices are only a chain
    # here when they are neighbours, i.e. a genuine two-bead dumbbell.
    for number, chain in enumerate(chains_list, start=1):
        if len(chain) == 2 and abs(chain[1] - chain[0]) != 1:
            raise ValueError(
                f"Chain {number} is given as head,tail ({chain[0]},{chain[1]}), but Rouse modes "
                "need every bead of each chain, listed in backbone order "
                f"(for example {_example_chain(chain)}). Head,tail pairs only describe the "
                "end-to-end vector; list all beads, or use the equal-length chains mode."
            )
    if len({len(c) for c in chains_list}) != 1:
        raise ValueError(
            "Every chain must list the same number of beads for a Rouse-mode analysis."
        )
    return np.array(chains_list, dtype=int)


def resolve_segment_pairs(
    mode, *, n_chains, beads_per_chain, first_index, custom, n_atoms, topology=None
) -> np.ndarray:
    if mode == "uniform":
        chains, beads = _uniform_extent(n_chains, beads_per_chain, first_index, n_atoms)
        # Only each block's first and last bead are wanted, so build those two columns
        # directly instead of the whole (n_chains, beads_per_chain) matrix.
        starts = first_index + np.arange(chains) * beads
        return np.column_stack([starts, starts + beads - 1])
    if mode == "data_file":
        # Chains of unequal length are fine here, unlike for Rouse modes: an end-to-end vector
        # needs only the two ends, so a polydisperse melt works as it stands.
        return _checked_topology(topology, n_atoms).segment_pairs()
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
