import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional

# 1. Data structure to hold dump data
@dataclass
class DumpFrame:
    metadata: dict[str, list[str] | np.ndarray | int]  # Stores any ITEM block as a list of strings (or parsed numpy arrays)
    columns: list[str]                           # The names of the atom data columns
    data: np.ndarray                             # The numerical atom data

    # Internal cache for fast column lookups
    _column_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Create a simple mapping from column name to column index for fast lookups.
        self._column_index = {col: idx for idx, col in enumerate(self.columns)}

    def column_index(self, name: str) -> int:
        """Return the column index for a given column name."""
        return self._column_index[name]

    def get_column(self, name: str) -> np.ndarray:
        """Return the data values for a given column name."""
        return self.data[:, self.column_index(name)]

    def get_column_or(self, name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Return the column values if present, otherwise return a default."""
        idx = self._column_index.get(name)
        return self.data[:, idx] if idx is not None else default

    def to_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """Return the data block as a pandas DataFrame."""
        df = pd.DataFrame(self.data, columns=self.columns)
        return df.copy() if copy else df


def iter_dump_frames(filepath: str) -> Iterator[DumpFrame]:
    """Yield frames from a LAMMPS dump file.

    Each frame corresponds to a single timestep block in the dump file. The function
    is designed to cope with LAMMPS dump output where the order and presence of
    `ITEM:` blocks is not strictly fixed.

    Frames are yielded as `DumpFrame` objects with:
    - `metadata`: dict where each key is the `ITEM:` header (e.g., "TIMESTEP", "BOX BOUNDS pp pp pp") and
      each value is the list of following lines for that block (converted when possible).
    - `columns`: atom column names specified in the `ITEM: ATOMS ...` line.
    - `data`: a NumPy array built from the atom data block (via `np.loadtxt`).

    Metadata conversion rules:
    - "TIMESTEP" is converted to `int` when possible.
    - "NUMBER OF ..." entries are converted to `int` when possible.
    - "BOX BOUNDS ..." entries are parsed into floats and returned either as:
        * a NumPy array (regular 2D) when each line has the same number of values, or
        * a list of float lists when the box bounds contain mixed-length lines.

    Args:
        filepath: Path to a LAMMPS dump file.

    Yields:
        DumpFrame objects for each timestep found in the file.
    """

    def _convert_metadata(meta: dict[str, list[str] | np.ndarray | int]) -> dict[str, list[str] | np.ndarray | int]:
        """Normalize/convert common metadata values.

        This helper mutates `meta` in-place (on a copy) and returns it.

        Conversion rules:
        - `BOX BOUNDS...`: parse each line as floats. If every line has the same
          number of values, the result becomes a regular `np.ndarray`. Otherwise,
          the result remains a `list[list[float]]`.
        - `TIMESTEP`: convert to `int` (from single-string list).
        - `NUMBER OF...`: convert to `int` (from single-string list).

        Args:
            meta: The metadata dict to normalize.

        Returns:
            The normalized metadata dict.
        """
        for key, value in list(meta.items()):
            if key.startswith("BOX BOUNDS") and isinstance(value, list):
                try:
                    meta[key] = np.array([list(map(float, line.split())) for line in value])
                except Exception:
                    # If conversion fails, keep the original list of strings
                    pass

            elif key == "TIMESTEP" and isinstance(value, list) and value:
                try:
                    meta[key] = int(value[0])
                except Exception:
                    # If conversion fails, keep the original list of strings
                    pass

            elif key.startswith("NUMBER OF") and isinstance(value, list) and value:
                try:
                    meta[key] = int(value[0])
                except Exception:
                    # If conversion fails, keep the original list of strings
                    pass

        return meta
    
    metadata = {}
    columns = []
    bulk_data = []
    current_header = None
    
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue    # Continue if empty line
                
            # Check if new data block starts (identified by the "ITEM:" keyword)
            if stripped.startswith("ITEM:"):
                # Extract the name of the header (e.g., "TIMESTEP", "BOX BOUNDS pp pp pp")
                header_content = stripped[5:].strip()
                
                # If we hit a new TIMESTEP and we already have atom data, 
                # we know the previous frame is finished. Yield it!
                if header_content.startswith("TIMESTEP") and bulk_data:
                    frame_metadata = _convert_metadata(metadata.copy())
                    yield DumpFrame(
                        metadata=frame_metadata,
                        columns=columns.copy(),
                        data=np.loadtxt(bulk_data)
                    )
                    
                    # Reset the containers for the next frame
                    metadata = {}
                    columns = []
                    bulk_data = []
                
                # Special handling for the ATOMS block to extract column names
                if header_content.startswith("ATOMS"):
                    current_header = "ATOMS"
                    parts = header_content.split()
                    # Grab everything after "ATOMS" as the column names
                    columns = parts[1:] if len(parts) > 1 else []
                    continue
                # Special handling for the BONDS block to extract column names
                elif header_content.startswith("BONDS"):
                    current_header = "BONDS"
                    parts = header_content.split()
                    # Grab everything after "BONDS" as the column names
                    columns = parts[1:] if len(parts) > 1 else []
                    continue
                    
                # For any other generic header, set it as the active dictionary key
                current_header = header_content
                metadata[current_header] = []
                
            else:
                # We are reading the data lines belonging to the current header
                if current_header == "ATOMS" or current_header == "BONDS":
                    bulk_data.append(stripped)
                elif current_header is not None:
                    metadata[current_header].append(stripped)
                    
        # When the file ends, yield the very last frame
        if bulk_data:
            frame_metadata = _convert_metadata(metadata)
            yield DumpFrame(
                metadata=frame_metadata,
                columns=columns,
                data=np.loadtxt(bulk_data)
            )
