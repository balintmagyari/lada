import re

# The "N atom types" header line. Matched whole so that the title (always line 1) or any other
# line that merely mentions "atom types" is never mistaken for it.
_ATOM_TYPES_RE = re.compile(r"^\s*(\d+)\s+atom\s+types\s*$")


def _atom_types_line(lines: list[str]) -> tuple[int, int]:
    """Return (line index, count) of the header's atom types line, skipping the title."""
    for index, line in enumerate(lines[1:], start=1):
        match = _ATOM_TYPES_RE.match(line.split("#", 1)[0])
        if match:
            return index, int(match.group(1))
    raise ValueError(
        "The data file has no 'N atom types' header line, so the atom type count is unknown."
    )


def rewrite_end_beads(
    input_file: str, output_file: str, new_end_type: int, base_type: int = 1
) -> None:
    """
    Read a LAMMPS data file, identify polymer end beads, and rewrite the topology.

    The function parses the `Atoms` section to group atoms by molecule ID and
    identifies the terminal beads as the minimum and maximum atom IDs within each
    molecule. Only real chains are touched: atoms with molecule ID 0 (no molecule)
    and the atom of a single-atom molecule keep their type. It rewrites the data
    file to the specified output path, replacing the atom type of these terminal
    beads with `new_end_type`.

    If `new_end_type` is one more than the header's atom types count, it is a new
    type: the count is raised by one and the mass and pair coefficients of
    `base_type` are cloned for it, so the new type has valid physical parameters
    in LAMMPS. If `new_end_type` is an existing type, the ends are retyped to it
    and the header, Masses and Pair Coeffs are left as they are.

    Parameters
    ----------
    input_file : str
        Path to the input LAMMPS data file.
    output_file : str
        Path where the modified LAMMPS data file will be saved.
    new_end_type : int
        The atom type integer to assign to the identified terminal beads: an
        existing type (1 to N) or the next new one (N + 1), where N is the
        header's atom types count.
    base_type : int, default=1
        The existing atom type integer in the input file whose mass and pair
        coefficients (if present) will be duplicated for a new `new_end_type`.

    Returns
    -------
    None
        The function writes the modified topology directly to disk and does not
        return any object.

    Raises
    ------
    ValueError
        If `new_end_type` is below 1 or would leave a gap in the type numbering
        (greater than N + 1), or if the file has no ``N atom types`` header line.
    """
    # Valid sections in a LAMMPS data file
    sections = [
        "Atoms",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Masses",
        "Velocities",
        "Pair Coeffs",
        "PairIJ Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
    ]

    with open(input_file) as f:
        lines = f.readlines()

    types_line, num_types = _atom_types_line(lines)
    if new_end_type < 1:
        raise ValueError(f"new_end_type must be a positive atom type, not {new_end_type}.")
    if new_end_type > num_types + 1:
        raise ValueError(
            f"new_end_type {new_end_type} would leave a gap in the atom type numbering: the "
            f"file has {num_types} atom types, so the end type must be an existing type "
            f"(1 to {num_types}) or the next new one ({num_types + 1})."
        )
    # An existing type already has its mass and coefficients; adding them again would
    # duplicate the entries.
    adds_new_type = new_end_type == num_types + 1

    mol_dict = {}
    base_mass = "1"
    base_pair_coeff = "1 1"

    # --- PASS 1: Map Topology & Extract Base Properties ---
    current_section = None
    # Line 1 is the title: free text that may well start with a section name.
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section headers
        if any(stripped.startswith(sec) for sec in sections):
            # Handle exact matches or matches followed by a space
            for sec in sections:
                if stripped == sec or stripped.startswith(sec + " "):
                    current_section = sec
                    break
            continue

        if current_section == "Atoms":
            parts = stripped.split()
            if len(parts) >= 3:
                atom_id = int(parts[0])
                mol_id = int(parts[1])
                if mol_id not in mol_dict:
                    mol_dict[mol_id] = []
                mol_dict[mol_id].append(atom_id)

        elif current_section == "Masses":
            parts = stripped.split()
            if len(parts) >= 2 and int(parts[0]) == base_type:
                base_mass = parts[1]

        elif current_section == "Pair Coeffs":
            parts = stripped.split(maxsplit=1)
            if len(parts) >= 2 and int(parts[0]) == base_type:
                base_pair_coeff = parts[1]

    # Isolate the terminal beads
    end_atoms = set()
    for mol_id, atoms in mol_dict.items():
        # Mol 0 is "no molecule" and a one-atom molecule is not a chain: neither has ends.
        if mol_id != 0 and len(atoms) >= 2:
            end_atoms.add(min(atoms))
            end_atoms.add(max(atoms))

    print(f"Mapped {len(end_atoms)} end beads. Base mass: {base_mass}")

    # --- PASS 2: Rewrite the Data File ---
    out_lines = []
    current_section = None
    # Nothing is appended for an existing type, so both count as already done.
    added_mass = not adds_new_type
    added_pair_coeff = not adds_new_type

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 0. The title line is copied as it is
        if i == 0:
            out_lines.append(line)
            i += 1
            continue

        # 1. Update the atom types header
        if i == types_line:
            out_lines.append(f"{new_end_type} atom types\n" if adds_new_type else line)
            i += 1
            continue

        # 2. Check for section transitions
        is_section_header = False
        for sec in sections:
            if stripped == sec or stripped.startswith(sec + " "):
                # Before moving to the new section, close out the previous one if needed
                if current_section == "Masses" and not added_mass:
                    # Remove trailing blank lines from the previous section
                    while out_lines and not out_lines[-1].strip():
                        out_lines.pop()
                    out_lines.append(f"{new_end_type} {base_mass}\n\n")
                    added_mass = True

                elif current_section == "Pair Coeffs" and not added_pair_coeff:
                    # Remove trailing blank lines from the previous section
                    while out_lines and not out_lines[-1].strip():
                        out_lines.pop()
                    out_lines.append(f"{new_end_type} {base_pair_coeff}\n\n")
                    added_pair_coeff = True

                current_section = sec
                is_section_header = True
                break

        if is_section_header:
            out_lines.append(line)
            i += 1
            continue

        # 3. Rewrite Atoms and swap types
        if current_section == "Atoms" and stripped:
            parts = line.split()
            atom_id = int(parts[0])

            if atom_id in end_atoms:
                parts[2] = str(new_end_type)
                # Restored standard LAMMPS spacing, removed the "New line: " text
                new_line = " ".join(parts) + "\n"
                out_lines.append(new_line)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

        i += 1

    # Edge case: If the file ended on Masses or Pair Coeffs
    if current_section == "Masses" and not added_mass:
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        out_lines.append(f"{new_end_type} {base_mass}\n")

    elif current_section == "Pair Coeffs" and not added_pair_coeff:
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        out_lines.append(f"{new_end_type} {base_pair_coeff}\n")

    # --- Write to Output ---
    with open(output_file, "w") as f_out:
        f_out.writelines(out_lines)

    print(f"Successfully wrote fully updated topology to {output_file}")
