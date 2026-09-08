"""Rewrite a LAMMPS data file's chain-end atom types with `lada.modifiers`."""

from pathlib import Path

import numpy as np
from lada.modifiers import rewrite_end_beads
from lada.parsers import read_data_file

# Column 2 of the Atoms section is the molecule id only for these atom styles (lada's schema).
MOLECULAR_STYLES = {"bond", "molecular", "full"}


def rewrite_end_beads_file(
    input_path: Path, output_path: Path, *, new_end_type: int, base_type: int
) -> Path:
    data = read_data_file(str(input_path))
    style = data.metadata.get("atom style")
    if style not in MOLECULAR_STYLES:
        raise ValueError(
            "The Atoms section must use a molecular-style atom style (bond, molecular or full) "
            f"so that column 2 is the molecule id; this file declares '{style or 'no style'}'. "
            "Add an 'Atoms # molecular' comment to the section header if the layout is right."
        )
    masses = data.sections.get("Masses")
    known_types = (
        set()
        if masses is None or masses.size == 0
        else set(np.atleast_2d(masses)[:, 0].astype(int))
    )
    if base_type not in known_types:
        raise ValueError(
            f"There is no base type {base_type} in the Masses section "
            f"(found {sorted(known_types)})."
        )
    rewrite_end_beads(str(input_path), str(output_path), new_end_type, base_type)
    return output_path
