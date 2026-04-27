import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal
import re
from pathlib import Path
import pandas as pd

ATOM_STYLE_COLUMNS = {
    "atomic": ["id", "type", "x", "y", "z", "nx", "ny", "nz"],
    "charge": ["id", "type", "q", "x", "y", "z", "nx", "ny", "nz"],
    "bond": ["id", "molecule", "type", "x", "y", "z", "nx", "ny", "nz"],
    "molecular": ["id", "molecule", "type", "x", "y", "z", "nx", "ny", "nz"],
    "full": ["id", "molecule", "type", "q", "x", "y", "z", "nx", "ny", "nz"],
}
ALLOWED_SECTIONS = ('Masses', 'Atoms', 'Bonds', 'Velocities', 'Angles', 'Dihedrals', 'Impropers', 
                    'Nonbond Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs')

@dataclass
class LammpsData:
    metadata: Dict[str, Any]
    sections: Dict[str, np.ndarray]

    def get(self, section_name: str) -> np.ndarray:
        if section_name not in self.sections:
            raise KeyError(f"Section '{section_name}' not found in data. Please check if requested section appears in the data file.")
        return self.sections[section_name]

    def to_pandas(self, 
                  section: Literal['Masses', 'Atoms', 'Bonds', 'Velocities', 'Angles', 'Dihedrals', 'Impropers', 
                                   'Nonbond Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs'] = 'Atoms'):
        # Force the code to crash if an invalid string slips through
        if section not in ALLOWED_SECTIONS:
            raise ValueError(
                f"Invalid section: '{section}'. "
                f"Must be one of: {ALLOWED_SECTIONS}"
            )

        import pandas as pd
        data_array = self.get(section)
        
        if section == 'Atoms':
            try:
                atom_style = self.metadata['atom style']
                columns = ATOM_STYLE_COLUMNS[atom_style]
                valid_columns = columns[:data_array.shape[1]]
                return pd.DataFrame(data_array, columns=valid_columns)
            except KeyError:
                warnings.warn("No atom style detected in file. Returning generic columns.")
                return pd.DataFrame(data_array, columns=[f"col_{i}" for i in range(data_array.shape[1])])
        
        elif section == 'Bonds':
            return pd.DataFrame(data_array, columns=['bond_id', 'bond_type', 'atom1_id', 'atom2_id'])
        
        elif section == 'Velocities':
            return pd.DataFrame(data_array, columns=['atom_id', 'vx', 'vy', 'vz'])
        
        elif section == 'Masses':
            return pd.DataFrame(data_array, columns=['atom_id', 'mass'])
        
        else:
            warnings.warn("No hardcoding done for this section's column names! Returning to generic columns.")
            return pd.DataFrame(data_array, columns=[f"col_{i}" for i in range(data_array.shape[1])])
        
def read_data_file(filepath: str) -> LammpsData:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    metadata = {}
    
    # 1. Extract metadata from the first line (e.g., timestep, units)
    first_line = lines[0].strip()
    metadata["description"] = first_line
    if "=" in first_line:
        # Splits segments like "timestep = 10000000" into dictionary keys
        for part in first_line.split(","):
            if "=" in part:
                key, val = part.split("=", 1)
                metadata[key.strip()] = val.strip()

    sections_raw: Dict[str, List[str]] = {}
    current_section = None
    detected_style = None

    for line in lines[1:]:
        raw_line = line.strip()
        if not raw_line:
            continue
            
        # 2. Identify Section Headers
        if raw_line[0].isalpha():
            parts = raw_line.split('#')
            current_section = parts[0].strip()
            sections_raw[current_section] = []

            # 3. Robustly detect the atom style from the Atoms comment
            if current_section == "Atoms" and len(parts) > 1:
                # Split the comment into individual words to avoid partial matches
                comment_words = parts[1].strip().lower().split()
                for style in ATOM_STYLE_COLUMNS:
                    if style in comment_words:
                        detected_style = style
                        metadata['atom style'] = detected_style
                        break
                        
        else:
            clean_line = raw_line.split('#')[0].strip()
            if not clean_line:
                continue
                
            if current_section is None:
                # Parse global metadata
                parts = clean_line.split()
                num_parts = []
                str_parts = []
                
                for p in parts:
                    try:
                        val = float(p)
                        if val.is_integer():
                            val = int(val)
                        num_parts.append(val)
                    except ValueError:
                        str_parts.append(p)
                        
                key = " ".join(str_parts)
                metadata[key] = num_parts[0] if len(num_parts) == 1 else num_parts
            else:
                sections_raw[current_section].append(clean_line)

    sections = {name: np.loadtxt(data) if data else np.array([]) 
                for name, data in sections_raw.items()}

    return LammpsData(
        metadata=metadata, 
        sections=sections, 
    )

def _find_last_timestep_block(lines: list[str]) -> tuple[int, int, int]:
    """
    Scan *lines* and return (timestep_value, block_start, block_end) for the
    *last* '# Timestep: N' block found.
 
    block_start is the index of the first *data* line (i.e. the line after
    the '# Timestep:' header).  block_end is one past the last data line.
 
    Raises ValueError if fewer than two timestep blocks are found (we always
    skip Timestep 0, so at least one production block must exist).
    """
    timestep_header_re = re.compile(r"^#\s*Timestep:\s*(\d+)", re.IGNORECASE)
 
    block_starts: list[tuple[int, int]] = []   # (timestep_value, line_index_of_header)
 
    for i, line in enumerate(lines):
        m = timestep_header_re.match(line.strip())
        if m:
            block_starts.append((int(m.group(1)), i))
 
    if len(block_starts) < 2:
        raise ValueError(
            f"Expected at least 2 timestep blocks (including the t=0 reference), "
            f"but found {len(block_starts)}."
        )
 
    # Take the last block
    last_ts_value, last_header_idx = block_starts[-1]
 
    # Data starts on the line after the header
    block_start = last_header_idx + 1
 
    # Data ends at the end of the file (there is no subsequent block)
    block_end = len(lines)
 
    return last_ts_value, block_start, block_end
 
def read_lammps_acf(filepath: str | Path, 
                    lag_col: str = "lag_time") -> pd.DataFrame:
    """
    Read a LAMMPS stress-ACF output file and return the last timestep block
    as a tidy Pandas DataFrame.
 
    Parameters
    ----------
    filepath : str or Path
        Path to the LAMMPS ACF text file.
    lag_col : str, optional
        Name to give the lag-time column (default: ``'lag_time'``).
 
    Returns
    -------
    pd.DataFrame
        Columns:
          - ``lag_col``         : float - physical lag time
          - one column per ACF  : float - autocorrelation values
          - ``timestep``        : int   - the LAMMPS timestep of this block
    """
    filepath = Path(filepath)
 
    if not filepath.is_file():
        raise FileNotFoundError(f"No such file: {filepath}")
 
    with filepath.open("r") as fh:
        lines = fh.readlines()
 
    # ------------------------------------------------------------------
    # 1.  Parse column names from line 1 (comma-separated)
    # ------------------------------------------------------------------
    acf_columns = [c.strip() for c in lines[0].strip().split(",") if c.strip()]
 
    # ------------------------------------------------------------------
    # 2.  Locate the last timestep block
    # ------------------------------------------------------------------
    timestep_value, block_start, block_end = _find_last_timestep_block(lines)
 
    # ------------------------------------------------------------------
    # 3.  Parse data rows (skip blank / comment lines inside the block)
    # ------------------------------------------------------------------
    records: list[list[float]] = []
 
    for line in lines[block_start:block_end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values = [float(v) for v in stripped.split()]
        except ValueError:
            # Silently skip any malformed lines
            continue
 
        if len(values) != len(acf_columns) + 1:
            # Unexpected column count — skip with a warning
            import warnings
            warnings.warn(
                f"Skipping line with unexpected column count "
                f"(got {len(values)}, expected {len(acf_columns) + 1}): {stripped!r}",
                stacklevel=2,
            )
            continue
 
        records.append(values)
 
    if not records:
        raise ValueError(
            f"No valid data rows found in the last timestep block "
            f"(Timestep: {timestep_value})."
        )
 
    # ------------------------------------------------------------------
    # 4.  Build DataFrame
    # ------------------------------------------------------------------
    all_columns = [lag_col] + acf_columns
    df = pd.DataFrame(records, columns=all_columns)
 
    # Ensure correct dtypes
    df[lag_col] = df[lag_col].astype(float)
    for col in acf_columns:
        df[col] = df[col].astype(float)
 
    # Attach the timestep as a scalar column
    df["timestep"] = timestep_value
 
    return df