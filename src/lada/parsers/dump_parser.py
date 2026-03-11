import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional
import warnings
from tqdm import tqdm

@dataclass
class DumpFrame:
    """
    Container for parsed LAMMPS dump files.

    This class holds the metadata, column names and bulk data for a single timestep's
    dump data. It provides helper methods to easily access and convert data into Pandas
    dataframes.

    Attributes:
        metadata (Dict[str, list[str] | np.ndarray | int]): global parameters of the current timestep
        columns (List[str]): column names for the main data
        data (np.ndarray): main data
    """
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

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        """Return the data block as a pandas DataFrame."""
        df = pd.DataFrame(self.data, columns=self.columns)
        return df.copy() if copy else df


def dump_frames(filepath: str) -> Iterator[DumpFrame]:
    """Yield frames from a LAMMPS dump file.

    Each frame corresponds to a single timestep block in the dump file. The function
    is designed to cope with LAMMPS dump output where the order and presence of
    `ITEM:` blocks is not strictly fixed.

    However, the function relies on the assumption that each new block of code belonging
    to different timesteps starts with the line "ITEM: TIMESTEP". If this is not the case
    the function will not work properly!

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
    current_data = []
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
                if header_content.startswith("TIMESTEP") and current_data:
                    frame_metadata = _convert_metadata(metadata.copy())

                    header_parts = current_header.split() if current_header else []
                    columns = header_parts[1:] if len(header_parts) > 1 else []

                    yield DumpFrame(
                        metadata=frame_metadata,
                        columns=columns.copy(),
                        data=np.loadtxt(current_data)
                    )

                    current_header = 'TIMESTEP'

                    # Reset the containers for the next frame
                    metadata = {}
                    columns = []
                    current_data = []
                
                # Check if a current header exists and if there is data under that header
                elif current_header and current_data:
                    metadata[current_header] = current_data         # Add metadata using the current header and data
                    current_header = header_content                 # Change current_header to new value
                    current_data = []                               # Reset current data to empty list

                else:
                    current_header = header_content                 # Reset current_header to new if no data is present underneath it (or if current header has not been assigned yet)
                
            else:
                current_data.append(stripped)
                
        # When the file ends, yield the very last frame
        if current_data:
            frame_metadata = _convert_metadata(metadata.copy())

            header_parts = current_header.split() if current_header else []
            columns = header_parts[1:] if len(header_parts) > 1 else []

            yield DumpFrame(
                metadata=frame_metadata,
                columns=columns.copy(),
                data=np.loadtxt(current_data)
            )

def read_dump(filepath: str, timestep_col: str = 'timestep') -> pd.DataFrame:
    """Read a dump file and concatenate every timestep into a single DataFrame.

    Each row in the returned DataFrame corresponds to one atom from a single
    timestep. If the dump contains a `TIMESTEP` metadata block, it is added as a
    column (default: ``timestep``).

    Parameters
    ----------
    filepath : str
        Path to a LAMMPS dump file.
    timestep_col : str, default='timestep'
        If the dump file contains a 'TIMESTEP' metadata block, its data is added
        as a column with the column name defined by this parameter. 

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing data of all timesteps listed in the dump file.
        Timesteps to which data belong to are indicated by the value in the column
        defined by the parameter 'timestep_col'.
    """
    frames = list(dump_frames(filepath))

    if not frames:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for frame in tqdm(frames, desc="Reading dump file"):
        df = frame.to_pandas(copy=False)
        if "TIMESTEP" in frame.metadata:
            df = df.assign(**{timestep_col: frame.metadata["TIMESTEP"]}) # Use dictionary unpacking to feed keyword arguments to the 'assign' function
        df.insert(0, timestep_col, df.pop(timestep_col))    # Move TIMESTEP column as first column
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, copy=False)


# Old function that eventually need to be depreciated
def iter_dump_frames(filepath: str) -> Iterator[DumpFrame]:
    """Yield frames from a LAMMPS dump file.

    .. deprecated:: 1.1.0
       `iter_dump_frames` will be removed in lada 2.0.0. 
       Please use `dump_frames` function instead.

    Each frame corresponds to a single timestep block in the dump file. The function
    is designed to cope with LAMMPS dump output where the order and presence of
    `ITEM:` blocks is not strictly fixed.

    However, the function relies on the assumption that each new block of code belonging
    to different timesteps starts with the line "ITEM: TIMESTEP". If this is not the case
    the function will not work properly!

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

    warnings.warn(
        "Call to depreciated function 'iter_dump_frames'. "
        "This function will be removed in version 2.0.0. Use 'dump_frames instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

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
