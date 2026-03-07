import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import List, Iterator

# 1. Define a clean structure to hold a single frame's data
@dataclass
class DumpFrame:
    metadata: dict[str, List[str]]  # Stores any ITEM block as a list of strings
    columns: List[str]              # The names of the atom data columns
    data: np.ndarray                # The numerical atom data

# 2. Create the generator function
def iter_dump_frames(filepath: str) -> Iterator[DumpFrame]:
    """
    Dynamically parses a LAMMPS dump file with unpredictable headers.
    """
    
    metadata = {}
    columns = []
    atom_lines = []
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
                if header_content.startswith("TIMESTEP") and atom_lines:
                    yield DumpFrame(
                        metadata=metadata.copy(),
                        columns=columns.copy(),
                        data=np.loadtxt(atom_lines)
                    )
                    
                    # Reset the containers for the next frame
                    metadata = {}
                    columns = []
                    atom_lines = []
                
                # Special handling for the ATOMS block to extract column names
                if header_content.startswith("ATOMS"):
                    current_header = "ATOMS"
                    parts = header_content.split()
                    # Grab everything after "ATOMS" as the column names
                    columns = parts[1:] if len(parts) > 1 else []
                    continue
                    
                # For any other generic header, set it as the active dictionary key
                current_header = header_content
                metadata[current_header] = []
                
            else:
                # We are reading the data lines belonging to the current header
                if current_header == "ATOMS":
                    atom_lines.append(stripped)
                elif current_header is not None:
                    metadata[current_header].append(stripped)
                    
        # When the file ends, yield the very last frame
        if atom_lines:
            yield DumpFrame(
                metadata=metadata,
                columns=columns,
                data=np.loadtxt(atom_lines)
            )
