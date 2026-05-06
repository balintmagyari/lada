"""LaDa: LAMMPS Data Analysis and reading.

A lightweight Python package for parsing LAMMPS output files (dumps, logs, 
data files) and performing vectorized molecular dynamics analysis calculations.

Core modules:
- parsers: Extract data from LAMMPS output files
- analysis: Calculate molecular properties (Rg, end-to-end distance, ACF, ISF, etc.)
- modifiers: Manipulate LAMMPS topology files
"""

# Parsers
from .parsers import (
    iter_dump_frames,
    dump_frames,
    read_dump,
    read_lammps_log,
    read_data_file,
    read_lammps_acf,
)

# Analysis
from .analysis import (
    calculate_avg_rg_sq,
    calculate_avg_ree_sq,
    calculate_ree_vectors,
    calculate_segment_acf,
    calculate_rouse_mode_acf,
    calculate_isf,
    calc_stress_relaxation,
)

# Modifiers
from .modifiers import rewrite_end_beads

__all__ = [
    # Parsers
    "iter_dump_frames",
    "dump_frames",
    "read_dump",
    "read_lammps_log",
    "read_data_file",
    "read_lammps_acf",
    # Analysis
    "calculate_avg_rg_sq",
    "calculate_avg_ree_sq",
    "calculate_ree_vectors",
    "calculate_segment_acf",
    "calculate_rouse_mode_acf",
    "calculate_isf",
    "calc_stress_relaxation",
    # Modifiers
    "rewrite_end_beads",
]