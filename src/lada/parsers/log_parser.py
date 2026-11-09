"""Parser for LAMMPS log files.

This module extracts the thermodynamic table that appears between the
"Per MPI rank memory allocation" and "Loop time" markers and exposes it as a
small, convenient API.

Example:
    from LaDa.parsers.log_parser import read_lammps_log

    thermo = read_lammps_log("log.lammps")
    energy = thermo.get("E_pair")
    df = thermo.to_pandas()
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermoData:
    """Container for parsed LAMMPS thermodynamic output.

    A log with several ``run`` commands holds one thermo block per run, and each block can
    print a different set of columns (``thermo_style`` may change between runs). The blocks
    are aligned by column name: ``columns`` is the union of every block's header in order of
    first appearance, and a value a block did not print is ``NaN``. ``run_index`` says which
    block (0-based) each row came from, so the runs can be split apart again.

    Attributes:
        columns: List[str]
            The names of the table columns (e.g. ['Step', 'Temp', 'E_pair', ...]).
        data: np.ndarray
            A 2D float array with shape (n_steps, n_columns).
        run_index: np.ndarray | None
            A 1D int array with shape (n_steps,): the thermo block each row belongs to.
    """

    columns: list[str]
    data: np.ndarray
    run_index: np.ndarray | None = None

    def get(self, property_name: str) -> np.ndarray:
        """Return the values for a single thermodynamic property.

        Parameters
        ----------
        property_name:
            The column name to retrieve (case-sensitive).

        Returns
        -------
        np.ndarray
            A 1D array containing the requested property values.

        Raises
        ------
        ValueError
            If the requested property is not present in `columns`.
        """
        if property_name not in self.columns:
            raise ValueError(
                f"Property '{property_name}' not found in log. "
                f"Available columns are: {self.columns}"
            )

        index = self.columns.index(property_name)
        return self.data[:, index]

    def to_pandas(self):
        """Convert the parsed thermodynamic data into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns matching `self.columns` and rows matching the
            simulation timesteps.
        """
        import pandas as pd

        return pd.DataFrame(self.data, columns=self.columns)


def _numeric_row(stripped: str, n_columns: int) -> list[float] | None:
    """The values of a thermo row, or None if the line is not one.

    LAMMPS interleaves other output with the thermo rows -- ``WARNING:`` lines, SHAKE
    statistics, a ``Loop time`` variant, a truncated last line -- and none of it is data.
    Only a line of exactly ``n_columns`` numbers is.
    """
    parts = stripped.split()
    if len(parts) != n_columns:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return None


def read_lammps_log(filepath: str) -> ThermoData:
    """Parse a LAMMPS log file and return thermodynamic tabular output.

    The parser scans the file for each section starting with the line
    containing "Per MPI rank memory allocation" and ending just before the
    line containing "Loop time". Within that section, the first non-empty line
    is treated as the header (column names), and each following line holding
    one number per header column is a data row. Any other line inside the
    section (a ``WARNING:`` message, for example) is skipped.

    Each section is read against its own header. When the sections print
    different columns, the result is aligned by column name and padded with
    ``NaN`` (see :class:`ThermoData`).

    Args:
        filepath: Path to the LAMMPS log file.

    Returns:
        ThermoData: Container object holding the column names and numeric data.
    """
    columns: list[str] = []
    # One (header, rows) pair per thermo block, in file order.
    blocks: list[tuple[list[str], list[list[float]]]] = []

    in_table = False
    header: list[str] | None = None

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # 1. Check for the start marker
            if "Per MPI rank memory allocation" in stripped:
                in_table = True
                header = None  # The next non-empty line is this block's header
                continue

            # 2. Check for the end marker
            if "Loop time of" in stripped:
                in_table = False
                continue

            # 3. If we are currently inside the data block
            if in_table:
                if header is None:
                    # A warning printed between the memory line and the header is not it.
                    if stripped.startswith("WARNING"):
                        continue
                    header = stripped.split()
                    blocks.append((header, []))
                    for name in header:
                        if name not in columns:
                            columns.append(name)
                else:
                    row = _numeric_row(stripped, len(header))
                    if row is not None:
                        blocks[-1][1].append(row)

    n_rows = sum(len(rows) for _, rows in blocks)
    if n_rows == 0:
        return ThermoData(
            columns=columns, data=np.array([]), run_index=np.array([], dtype=int)
        )

    data_array = np.full((n_rows, len(columns)), np.nan)
    run_index = np.empty(n_rows, dtype=int)
    position = {name: i for i, name in enumerate(columns)}
    row_start = 0
    for block_number, (block_header, rows) in enumerate(blocks):
        if not rows:
            continue
        block = np.asarray(rows, dtype=float)
        targets = [position[name] for name in block_header]
        data_array[row_start : row_start + len(rows), targets] = block
        run_index[row_start : row_start + len(rows)] = block_number
        row_start += len(rows)

    return ThermoData(columns=columns, data=data_array, run_index=run_index)
