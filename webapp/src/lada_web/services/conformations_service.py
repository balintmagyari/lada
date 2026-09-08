"""Chain size and shape calculations: Rg^2, Ree^2 and end-to-end vectors from a dump file."""

from pathlib import Path

import pandas as pd
from lada.analysis import calculate_avg_ree_sq, calculate_avg_rg_sq, calculate_ree_vectors
from lada.parsers import read_dump


def _per_timestep(values: dict[float, float] | float, name: str) -> pd.DataFrame:
    """Turn a `calculate_avg_*` result into a `[timestep, name]` DataFrame.

    `read_dump` always adds a timestep column, so `values` is normally a dict keyed by
    timestep; the bare-float case is kept only for symmetry with the underlying `lada`
    functions, which also accept single-frame input without a timestep column.
    """
    if isinstance(values, dict):
        return pd.DataFrame({"timestep": list(values.keys()), name: list(values.values())})
    return pd.DataFrame({"timestep": [0], name: [values]})


def rg_sq(
    path: Path, *, coord_cols: list[str], molecule_col: str, mass_col: str | None
) -> pd.DataFrame:
    df = read_dump(str(path))
    values = calculate_avg_rg_sq(
        df, coord_cols=coord_cols, molecule_col=molecule_col, mass_col=mass_col
    )
    return _per_timestep(values, "rg_sq")


def ree_sq(
    path: Path, *, coord_cols: list[str], molecule_col: str, atom_id_col: str
) -> pd.DataFrame:
    df = read_dump(str(path))
    values = calculate_avg_ree_sq(
        df, coord_cols=coord_cols, molecule_col=molecule_col, atom_id_col=atom_id_col
    )
    return _per_timestep(values, "ree_sq")


def ree_vectors(
    path: Path, *, coord_cols: list[str], molecule_col: str, atom_id_col: str
) -> pd.DataFrame:
    df = read_dump(str(path))
    return calculate_ree_vectors(
        df, coord_cols=coord_cols, molecule_col=molecule_col, atom_id_col=atom_id_col
    )
