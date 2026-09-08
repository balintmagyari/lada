"""Read LAMMPS output files with `lada.parsers` and return plain DataFrames."""

from pathlib import Path

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
    # np.loadtxt drops to 1-D whenever either axis is 1 -- one thermo row (`run 0`) or one
    # thermo column (`thermo_style custom step`) -- and to_pandas() cannot label that. The
    # column count says which orientation it is, so reshape on it rather than guessing.
    data = thermo.data.reshape(-1, len(thermo.columns))
    return pd.DataFrame(data, columns=thermo.columns)
