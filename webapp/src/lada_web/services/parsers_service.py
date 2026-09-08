"""Read LAMMPS output files with `lada.parsers` and return plain DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
from lada.parsers import dump_frames, read_data_file, read_dump, read_lammps_acf, read_lammps_log

NO_THERMO_TABLE = (
    "No thermodynamic table was found in this file. Upload a LAMMPS log that "
    "contains a thermo output block."
)


def log_table(path: Path) -> pd.DataFrame:
    thermo = read_lammps_log(str(path))
    # A file without a thermo block yields no columns and no rows; converting that to a
    # DataFrame raises a pandas shape error, so check before converting.
    if not thermo.columns or thermo.data.size == 0:
        raise ValueError(NO_THERMO_TABLE)
    # np.loadtxt drops to 1-D whenever either axis is 1 -- one thermo row (`run 0`) or one
    # thermo column (`thermo_style custom step`) -- and to_pandas() cannot label that. The
    # column count says which orientation it is, so reshape on it rather than guessing.
    data = thermo.data.reshape(-1, len(thermo.columns))
    return pd.DataFrame(data, columns=thermo.columns)


def dump_preview(path: Path) -> pd.DataFrame:
    """One row per frame; never concatenates the trajectory."""
    records = []
    for frame in dump_frames(str(path)):
        bounds = next((v for k, v in frame.metadata.items() if k.startswith("BOX BOUNDS")), None)
        sizes = (bounds[:, 1] - bounds[:, 0]) if bounds is not None else np.full(3, np.nan)
        records.append(
            {
                "timestep": frame.metadata["TIMESTEP"],
                "n_atoms": frame.data.shape[0],
                "box_x": float(sizes[0]),
                "box_y": float(sizes[1]),
                "box_z": float(sizes[2]),
                "columns": " ".join(frame.columns),
            }
        )
    if not records:
        raise ValueError(
            "No frames were found. Upload a LAMMPS dump file written by `dump ... custom`."
        )
    return pd.DataFrame(records)


def dump_table(path: Path) -> pd.DataFrame:
    return read_dump(str(path))


def data_table(path: Path, section: str) -> pd.DataFrame:
    return read_data_file(str(path)).to_pandas(section)


def acf_table(path: Path) -> pd.DataFrame:
    return read_lammps_acf(path)
