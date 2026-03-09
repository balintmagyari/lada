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
from typing import List
import numpy as np


@dataclass
class ThermoData:
    """Container for parsed LAMMPS thermodynamic output.

    Attributes:
        columns: List[str]
            The names of the table columns (e.g. ['Step', 'Temp', 'E_pair', ...]).
        data: np.ndarray
            A 2D float array with shape (n_steps, n_columns).
    """

    columns: List[str]
    data: np.ndarray

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


def read_lammps_log(filepath: str) -> ThermoData:
    """Parse a LAMMPS log file and return thermodynamic tabular output.

    The parser scans the file for the section starting with the line
    containing "Per MPI rank memory allocation" and ending just before the
    line containing "Loop time". Within that section, the first non-empty line
    is treated as the header (column names), and each following line is
    interpreted as floating-point data.

    Args:
        filepath: Path to the LAMMPS log file.

    Returns:
        ThermoData: Container object holding the column names and numeric data.
    """
    columns = []
    data_lines = []
    
    in_table = False
    header_found = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # 1. Check for the start marker
            if "Per MPI rank memory allocation" in stripped:
                in_table = True
                header_found = False  # Reset so we know to grab the header next
                continue
                
            # 2. Check for the end marker
            if "Loop time of" in stripped:
                in_table = False
                continue
                
            # 3. If we are currently inside the data block
            if in_table:
                if not header_found:
                    # The first line after the memory allocation is the header
                    columns = stripped.split()
                    header_found = True
                else:
                    # All subsequent lines are numerical data
                    data_lines.append(stripped)

    # Convert the collected text strings into a 2D NumPy array
    data_array = np.loadtxt(data_lines) if data_lines else np.array([])
    
    return ThermoData(columns=columns, data=data_array)
