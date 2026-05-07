"""LaDa: LAMMPS Data Analysis and reading.

A lightweight Python package for parsing LAMMPS output files (dumps, logs,
data files) and performing vectorized molecular dynamics analysis calculations.

Core modules:
- parsers: Extract data from LAMMPS output files
- analysis: Calculate molecular properties (Rg, end-to-end distance, ACF, ISF, etc.)
- modifiers: Manipulate LAMMPS topology files
- exporters: Write analysis results to files for external tools (e.g. pgfplots/LaTeX)
"""

# Parsers
# Analysis
from .analysis import (
    calc_stress_relaxation,
    calculate_avg_ree_sq,
    calculate_avg_rg_sq,
    calculate_isf,
    calculate_ree_vectors,
    calculate_rouse_mode_acf,
    calculate_segment_acf,
)

# Exporters
from .exporters import write_pgfplots_table

# Modifiers
from .modifiers import rewrite_end_beads
from .parsers import (
    dump_frames,
    iter_dump_frames,
    read_data_file,
    read_dump,
    read_lammps_acf,
    read_lammps_log,
)

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
    # Exporters
    "write_pgfplots_table",
]
