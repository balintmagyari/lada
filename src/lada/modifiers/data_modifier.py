def rewrite_end_beads(input_file: str, output_file: str, new_end_type: int, base_type: int = 1) -> None:
    """
    Read a LAMMPS data file, identify polymer end beads, and rewrite the topology.

    The function parses the `Atoms` section to group atoms by molecule ID and 
    identifies the terminal beads as the minimum and maximum atom IDs within each 
    molecule. It rewrites the data file to the specified output path, replacing 
    the atom type of these terminal beads with `new_end_type`. It dynamically 
    updates the header's total atom types count and clones the mass and pair 
    coefficients from an existing `base_type` to ensure the new atom type has 
    valid physical parameters in LAMMPS.

    Parameters
    ----------
    input_file : str
        Path to the input LAMMPS data file.
    output_file : str
        Path where the modified LAMMPS data file will be saved.
    new_end_type : int
        The new atom type integer to assign to the identified terminal beads.
    base_type : int, default=1
        The existing atom type integer in the input file whose mass and pair 
        coefficients (if present) will be duplicated for the `new_end_type`.

    Returns
    -------
    None
        The function writes the modified topology directly to disk and does not 
        return any object.
    """
    # Valid sections in a LAMMPS data file
    sections = ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", 
                "Masses", "Velocities", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs"]
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    mol_dict = {}
    base_mass = "1"
    base_pair_coeff = "1 1"
    
    # --- PASS 1: Map Topology & Extract Base Properties ---
    current_section = None
    for line in lines:
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
        if len(atoms) >= 2:
            end_atoms.add(min(atoms))
            end_atoms.add(max(atoms))
        elif len(atoms) == 1:
            end_atoms.add(atoms[0])

    print(f"Mapped {len(end_atoms)} end beads. Base mass: {base_mass}")

    # --- PASS 2: Rewrite the Data File ---
    out_lines = []
    current_section = None
    types_updated = False
    added_mass = False
    added_pair_coeff = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 1. Update the atom types header
        if "atom types" in line and not types_updated:
            parts = line.split()
            num_types = int(parts[0])
            if new_end_type > num_types:
                out_lines.append(f"{new_end_type} atom types\n")
            else:
                out_lines.append(line)
            types_updated = True
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
    with open(output_file, 'w') as f_out:
        f_out.writelines(out_lines)
        
    print(f"Successfully wrote fully updated topology to {output_file}")