"""Read LAMMPS output files with `lada.parsers` and return plain DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
from lada.parsers import read_lammps_log

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
    # np.loadtxt collapses a single data line to 1-D, which to_pandas() cannot label, so
    # build the frame here instead. A log from `run 0` prints exactly one thermo row.
    return pd.DataFrame(np.atleast_2d(thermo.data), columns=thermo.columns)
