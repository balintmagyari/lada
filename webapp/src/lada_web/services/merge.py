"""Combine several files' calculation results into one averaged (or regime-merged) DataFrame.

Kept local to the web app rather than promoted into core `lada`, matching how the dynamics
topology-resolution helpers are already scoped: this is presentation-layer convenience over the
library's single-file functions, not a new capability of the library itself.
"""

import numpy as np
import pandas as pd


def average_per_timestep(frames: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Average `value_col` across files that share the same `timestep` grid (repeat seeds).

    Raises if the files' timesteps don't match — repeats must come from the same dump schedule
    for an elementwise average to mean anything.
    """
    if len(frames) == 1:
        return frames[0]
    base = frames[0].sort_values("timestep").reset_index(drop=True)
    timesteps = base["timestep"].to_numpy()
    values = [base[value_col].to_numpy()]
    for frame in frames[1:]:
        ordered = frame.sort_values("timestep").reset_index(drop=True)
        if not np.array_equal(ordered["timestep"].to_numpy(), timesteps):
            raise ValueError(
                "Files have different timesteps; repeats must share the same dump schedule."
            )
        values.append(ordered[value_col].to_numpy())
    return pd.DataFrame(
        {"timestep": timesteps, value_col: np.mean(np.column_stack(values), axis=1)}
    )


def concat_with_repeat_label(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack files' rows with a 1-based `repeat` column instead of averaging.

    End-to-end vectors have no cross-file correspondence (a molecule id in one file is a
    different physical chain than the same id in another file), so there's nothing meaningful to
    average component-wise the way two Rg² scalars can be averaged. This keeps every file's rows,
    labelled by which file they came from, rather than silently averaging unrelated vectors.
    """
    if len(frames) == 1:
        return frames[0]
    return pd.concat(
        [frame.assign(repeat=i + 1) for i, frame in enumerate(frames)], ignore_index=True
    )


def _common_grid(frames: list[pd.DataFrame], grid_col: str, min_len: int, label: str) -> np.ndarray:
    """The shortest file's `grid_col`, after checking every other file agrees on that prefix."""
    base = frames[0][grid_col].to_numpy()[:min_len]
    for frame in frames[1:]:
        other = frame[grid_col].to_numpy()[:min_len]
        if not np.allclose(other, base, rtol=1e-9, atol=0):
            raise ValueError(
                f"Files have different {label}; repeats must use the same sampling settings "
                "(the rows must also be in the same order in every file)."
            )
    return base


def _with_discard_note(merged: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Record on the result how many rows truncate-to-shortest dropped, for `meta` to report."""
    dropped = discarded_rows(frames)
    if dropped:
        shortest = min(len(frame) for frame in frames)
        merged.attrs["discarded_rows"] = (
            f"{dropped} (files differ in length; averaged over the shortest file's {shortest} rows)"
        )
    return merged


def discarded_rows(frames: list[pd.DataFrame]) -> int:
    """Rows the truncate-to-shortest averaging drops, summed over the files that are longer.

    Averaging truncates to the shortest file so replicates of unequal length can still be
    combined. That is silent data loss, so routers report this count in `meta` whenever it is
    non-zero, rather than leaving a shorter run (a crash, a restart) to quietly shrink the result.
    """
    if len(frames) < 2:
        return 0
    lengths = [len(frame) for frame in frames]
    return sum(length - min(lengths) for length in lengths)


def average_rheology(frames: list[pd.DataFrame], value_cols: tuple[str, ...]) -> pd.DataFrame:
    """Truncate to the shortest file's lag-time grid, then average the value columns.

    Truncating before validating handles the realistic case of replicate runs with the same
    correlation settings but different total simulation length, while still catching genuinely
    incompatible grids.
    """
    if len(frames) == 1:
        return frames[0]
    min_len = min(len(frame) for frame in frames)
    lag_time = _common_grid(frames, "lag_time", min_len, "lag-time grids")
    out: dict[str, np.ndarray] = {"lag_time": lag_time}
    for col in value_cols:
        stacked = np.column_stack([frame[col].to_numpy()[:min_len] for frame in frames])
        out[col] = stacked.mean(axis=1)
    return _with_discard_note(pd.DataFrame(out), frames)


def average_dynamics(frames: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Truncate to the shortest file's frame count, then average the value column.

    lag_time is derived from time_per_frame, not read from the file, so once frame counts match,
    every file's lag-time grid is identical by construction — no grid-agreement check is needed
    the way `average_rheology` needs one for its file-provided lag_time.
    """
    if len(frames) == 1:
        return frames[0]
    min_len = min(len(frame) for frame in frames)
    lag_time = frames[0]["lag_time"].to_numpy()[:min_len]
    stacked = np.column_stack([frame[value_col].to_numpy()[:min_len] for frame in frames])
    merged = pd.DataFrame({"lag_time": lag_time, value_col: stacked.mean(axis=1)})
    return _with_discard_note(merged, frames)


def merge_regimes(
    x_arrays: list[np.ndarray], series: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stitch several regimes into one series, taking each lag time from the finest regime.

    Regimes are ordered by how far they reach, and each contributes only the lag times beyond the
    previous (finer) regime's end. This is the multi-tau correlator convention: where two regimes
    cover the same lag time, the one that sampled it more finely wins, and a coarser regime is
    used only past the point where the finer one stops.

    One rule serves both geometries. Nested regimes — the usual case, since a correlation
    function's lag axis is frame_index * time_per_frame and so always starts at zero — are each
    trimmed to the window they are best at. Regimes that happen to be adjacent keep all their
    points, because nothing finer reaches into them, which is what the previous non-overlap rule
    produced for that case.

    That previous rule rejected overlapping windows outright and required regimes to agree to
    1e-9 where they met. Real regimes are independent runs: they all start at lag time zero, and
    they agree at a shared lag time only to within run-to-run noise (~1%), so both halves of it
    refused ordinary data.
    """
    if not x_arrays:
        raise ValueError("At least one regime is required.")

    names = list(series[0])
    indexed = [(x, s) for x, s in zip(x_arrays, series, strict=True) if x.size]
    if not indexed:
        raise ValueError("Every regime is empty; there is nothing to merge.")
    # Finest first: under nested sampling the shortest reach is the finest grid.
    indexed.sort(key=lambda pair: float(pair[0].max()))

    x_parts: list[np.ndarray] = []
    col_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    cutoff = -np.inf
    for x, s in indexed:
        keep = x > cutoff
        if not keep.any():
            # Wholly covered by finer regimes: it carries no lag time they don't already hold.
            continue
        x_parts.append(x[keep])
        for name in names:
            col_parts[name].append(s[name][keep])
        cutoff = float(x.max())

    x_merged = np.concatenate(x_parts)
    order = np.argsort(x_merged, kind="stable")
    return x_merged[order], {
        name: np.concatenate(parts)[order] for name, parts in col_parts.items()
    }
