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
                f"Files have different {label}; repeats must use the same sampling settings."
            )
    return base


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
    return pd.DataFrame(out)


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
    return pd.DataFrame({"lag_time": lag_time, value_col: stacked.mean(axis=1)})


def merge_regimes(
    x_arrays: list[np.ndarray], series: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Concatenate several regimes' series, sorted by x, into one non-overlapping series.

    Regimes are checked for range overlap first — one regime's [min, max] window overlapping
    another's by more than a single shared boundary point is a genuine overlap, even if no
    individual sampled point happens to collide (e.g. a finely-sampled fast regime and a
    coarsely-sampled mid regime covering the same seconds at different lag times). A single
    shared boundary point (one regime's max equals another's min) is allowed, and is silently
    deduplicated if every series column also agrees there (a harmless resubmission of the same
    point twice) — but rejected if the value disagrees, since silently picking one of two
    disagreeing measurements at the same lag time would hide a real problem, not resolve one.
    """
    if not x_arrays:
        raise ValueError("At least one regime is required.")

    windows = sorted((float(x.min()), float(x.max())) for x in x_arrays if x.size)
    for (_, end), (start, _) in zip(windows, windows[1:], strict=False):
        if start < end:
            raise ValueError(
                f"Regimes overlap: one covers up to lag_time={end:g}, another starts at "
                f"lag_time={start:g}. Regimes must cover non-overlapping lag-time ranges."
            )

    x = np.concatenate(x_arrays)
    names = list(series[0])
    cols = {name: np.concatenate([s[name] for s in series]) for name in names}

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    cols_sorted = {name: col[order] for name, col in cols.items()}

    duplicate = np.zeros(len(x_sorted), dtype=bool)
    duplicate[1:] = x_sorted[1:] == x_sorted[:-1]
    for i in np.flatnonzero(duplicate):
        for name, col in cols_sorted.items():
            if not np.isclose(col[i], col[i - 1], rtol=1e-9):
                raise ValueError(
                    f"Regimes overlap at lag_time={x_sorted[i]:g}: '{name}' disagrees between "
                    "regimes. Regimes must cover non-overlapping lag-time ranges."
                )

    keep = ~duplicate
    return x_sorted[keep], {name: col[keep] for name, col in cols_sorted.items()}
