"""LaDa: Lightweight data access for LAMMPS dump files.

This package exposes a small API for streaming frames from LAMMPS dump files.
"""

from .parsers import iter_dump_frames
from .parsers import read_lammps_log
from .parsers import read_data_file

__all__ = ["iter_dump_frames", "read_lammps_log", "read_data_file"]
