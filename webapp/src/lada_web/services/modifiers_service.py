"""Rewrite a LAMMPS data file's chain-end atom types with `lada.modifiers`."""

from pathlib import Path

from lada.modifiers import rewrite_end_beads
from lada.parsers import read_data_file
from lada.parsers.data_parser import STYLES_WITH_MOLECULE


def rewrite_end_beads_file(
    input_path: Path, output_path: Path, *, new_end_type: int, base_type: int
) -> Path:
    data = read_data_file(str(input_path))
    style = data.metadata.get("atom style")
    if style not in STYLES_WITH_MOLECULE:
        raise ValueError(
            "The Atoms section must use an atom style whose column 2 is the molecule id "
            f"({', '.join(sorted(STYLES_WITH_MOLECULE))}); this file declares "
            f"'{style or 'no style'}'. Add an 'Atoms # molecular' comment to the section header "
            "if the layout is right."
        )
    masses = data.sections.get("Masses")
    known_types = set() if masses is None or masses.size == 0 else set(masses[:, 0].astype(int))
    if base_type not in known_types:
        raise ValueError(
            f"There is no base type {base_type} in the Masses section "
            f"(found {sorted(known_types)})."
        )
    rewrite_end_beads(str(input_path), str(output_path), new_end_type, base_type)
    return output_path
