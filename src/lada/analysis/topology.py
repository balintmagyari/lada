"""Recover chain topology from a LAMMPS data file.

The trajectory functions in `lada.analysis.dynamics` take chains as arrays of 0-indexed
positions into a `.npz` `coords` array. That array carries coordinates and nothing else -- no
atom ids, no molecule ids -- so the positions have to come from somewhere, and until now that
meant counting chains by hand. A data file already states the topology exactly, so read it.

The link between the two files is the one convention `coords` is written under: position `i`
holds the atom with the `i`-th smallest atom id. Every index here is a rank in that ordering,
never an atom id and never a row number, which is why a data file describing a different number
of atoms cannot be used at all.

Chains are found in the order the user asked for: molecule ids first, then the bond graph, then
nothing -- a file with neither is refused rather than guessed at.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..parsers.data_parser import ATOM_STYLE_COLUMNS, STYLES_WITH_MOLECULE, read_data_file

# A "chain" of one bead has no end-to-end vector and no Rouse modes. Solvent molecules and free
# ions land here in their thousands, so they are dropped rather than refused.
MIN_BEADS = 2

RAGGED_CHAINS = "Every chain must list the same number of beads for a Rouse-mode analysis."


@dataclass
class ChainTopology:
    """Chains as 0-indexed positions into a trajectory's `coords` array.

    Attributes
    ----------
    chains : list[np.ndarray]
        One array per chain, in backbone order from one end to the other.
    n_atoms : int
        Atoms in the data file. A trajectory must declare the same number for the positions to
        mean anything.
    source : str
        ``'molecule'`` or ``'bonds'`` -- which section the chains were recovered from.
    notes : list[str]
        Anything the reader had to decide for itself: chains it dropped, orderings it could not
        confirm. Empty when the file answered every question.
    """

    chains: list[np.ndarray]
    n_atoms: int
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def lengths(self) -> list[int]:
        return [len(chain) for chain in self.chains]

    def segment_pairs(self) -> np.ndarray:
        """The head and tail of every chain, shape ``(n_chains, 2)``.

        Chains of unequal length are fine here: an end-to-end vector needs only the two ends.
        """
        return np.array([[chain[0], chain[-1]] for chain in self.chains], dtype=int)

    def chain_indices(self) -> np.ndarray:
        """Every bead of every chain, shape ``(n_chains, beads_per_chain)``.

        Raises
        ------
        ValueError
            If the chains are not all the same length. numpy's own message for a ragged array
            names an "inhomogeneous shape", which says nothing to someone looking at a melt.
        """
        if len(set(self.lengths)) != 1:
            raise ValueError(RAGGED_CHAINS)
        return np.array([np.asarray(chain) for chain in self.chains], dtype=int)


def _atoms_section(data) -> np.ndarray:
    atoms = data.sections.get("Atoms")
    if atoms is None or atoms.size == 0:
        raise ValueError("The data file has no Atoms section, so it describes no particles.")
    return atoms


def _atom_ids(data, atoms: np.ndarray) -> np.ndarray:
    ids = atoms[:, 0]
    if not np.all(np.equal(np.mod(ids, 1), 0)):
        raise ValueError("Atom ids must be whole numbers; this file has fractional ids.")
    ids = ids.astype(int)

    values, counts = np.unique(ids, return_counts=True)
    if values.size != ids.size:
        repeated = values[counts > 1][:3]
        raise ValueError(
            f"Atom id {', '.join(str(int(v)) for v in repeated)} appears more than once. "
            "Every atom needs a unique id, or there is no telling which coordinate is whose."
        )

    # A truncated file still parses: its header says 40000 and its Atoms section holds 39000.
    # Left alone that shifts every position after the cut, so the whole analysis is wrong.
    declared = data.metadata.get("atoms")
    if isinstance(declared, int) and declared != ids.size:
        raise ValueError(
            f"The header declares {declared} atoms but the Atoms section holds {ids.size} rows. "
            "The file looks truncated or edited by hand."
        )
    return ids


def _positions(ids: np.ndarray) -> dict[int, int]:
    """Map each atom id to its rank in ascending id order -- the `coords` convention."""
    order = np.argsort(ids, kind="stable")
    rank = np.empty(ids.size, dtype=int)
    rank[order] = np.arange(ids.size)
    return {int(atom): int(place) for atom, place in zip(ids, rank, strict=True)}


def _adjacency(data, known: set[int]) -> dict[int, set[int]]:
    """Undirected bond graph keyed by atom id; empty when the file has no usable Bonds."""
    bonds = data.sections.get("Bonds")
    if bonds is None or bonds.size == 0:
        return {}
    if bonds.shape[1] < 4:
        raise ValueError(
            "The Bonds section needs four columns (id, type, atom 1, atom 2); its rows hold "
            f"only {bonds.shape[1]}."
        )
    adjacency: dict[int, set[int]] = defaultdict(set)
    for raw_first, raw_second in bonds[:, 2:4].astype(int):
        first, second = int(raw_first), int(raw_second)
        if first not in known or second not in known:
            raise ValueError(
                f"A bond joins atoms {first} and {second}, but the Atoms section does not list "
                "both of them."
            )
        if first == second:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    return dict(adjacency)


def _components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Connected components of the bond graph, each sorted by atom id."""
    seen: set[int] = set()
    found = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            atom = stack.pop()
            group.append(atom)
            for neighbour in adjacency[atom]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        found.append(sorted(group))
    return found


def _walk(group: list[int], adjacency: dict[int, set[int]]) -> list[int] | None:
    """Order a group end to end along its bonds, or None when it is not a simple chain.

    A ring has no ends and a branch has three, so neither has a backbone order to find. Bonds
    to atoms outside the group are ignored, which is what lets one molecule be walked using the
    whole file's bond graph.
    """
    inside = set(group)
    degree = {atom: len(adjacency.get(atom, set()) & inside) for atom in group}
    if any(count > 2 for count in degree.values()):
        return None
    ends = [atom for atom, count in degree.items() if count == 1]
    if len(ends) != 2:
        return None

    ordered = [min(ends)]
    seen = {ordered[0]}
    while True:
        nexts = [a for a in adjacency.get(ordered[-1], set()) & inside if a not in seen]
        if not nexts:
            break
        ordered.append(nexts[0])
        seen.add(ordered[-1])
    # A chain plus a detached pair walks short of its own group.
    return ordered if len(ordered) == len(group) else None


def _not_linear(group: list[int], adjacency: dict[int, set[int]]) -> str:
    """Name what is wrong with a group the walk refused, in the file's own atom ids."""
    inside = set(group)
    for atom in group:
        count = len(adjacency.get(atom, set()) & inside)
        if count > 2:
            return f"atom {atom} has {count} bonds, so the chain branches there"
    if all(len(adjacency.get(a, set()) & inside) == 2 for a in group):
        return f"the chain through atom {min(group)} is a closed ring, so it has no ends"
    return f"the chain through atom {min(group)} is not one connected strand"


def _by_molecule(atoms: np.ndarray, style: str, ids: np.ndarray) -> list[list[int]]:
    column = ATOM_STYLE_COLUMNS[style].index("molecule")
    molecules = atoms[:, column].astype(int)
    groups: dict[int, list[int]] = defaultdict(list)
    for molecule, atom in zip(molecules, ids, strict=True):
        groups[int(molecule)].append(int(atom))
    return [sorted(group) for _, group in sorted(groups.items())]


def _drop_short(groups: list[list[int]], notes: list[str]) -> list[list[int]]:
    keep = [group for group in groups if len(group) >= MIN_BEADS]
    dropped = len(groups) - len(keep)
    if dropped:
        notes.append(
            f"{dropped} of {len(groups)} molecules hold fewer than {MIN_BEADS} beads and were "
            "left out (solvent, ions and free beads have no end-to-end vector)."
        )
    if not keep:
        raise ValueError(
            f"No molecule in this file has {MIN_BEADS} or more beads, so there are no chains "
            "to analyse."
        )
    return keep


def _order_by_molecule(
    groups: list[list[int]], adjacency: dict[int, set[int]], notes: list[str]
) -> list[list[int]]:
    """Order molecule-grouped chains along their bonds, falling back to atom-id order.

    Molecule ids say which atoms belong together and nothing about the order they sit in along
    the backbone. Where the file has bonds they settle it; where it has none, ascending atom id
    is the only guess left and the caller is told it was a guess.
    """
    if not adjacency:
        notes.append(
            "This file has no Bonds section, so the beads of each chain were taken in atom-id "
            "order. That is right for chains written one bead after another, and wrong otherwise."
        )
        return groups

    ordered, guessed = [], 0
    for group in groups:
        walked = _walk(group, adjacency)
        if walked is None:
            guessed += 1
            ordered.append(group)
        else:
            ordered.append(walked)
    if guessed:
        notes.append(
            f"{guessed} of {len(groups)} chains are not a single bonded strand (a ring, a "
            "branch, or beads bonded to another molecule), so their beads were taken in atom-id "
            "order instead."
        )
    return ordered


def _order_by_bonds(groups: list[list[int]], adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Order bond-grouped chains along their bonds; refuse anything that is not a strand.

    Without molecule ids the bond graph is the only statement of topology the file makes, so
    there is nothing to fall back to when it describes something that is not a chain.
    """
    ordered = []
    for group in groups:
        walked = _walk(group, adjacency)
        if walked is None:
            raise ValueError(
                "The chains in this file were read from its Bonds section, but "
                f"{_not_linear(group, adjacency)}. An end-to-end vector and a Rouse mode both "
                "need a chain with two ends. Add molecule ids to the Atoms section, or enter "
                "the chain layout by hand."
            )
        ordered.append(walked)
    return ordered


def chains_from_data_file(filepath: str) -> ChainTopology:
    """Read the chain topology of a LAMMPS data file.

    Parameters
    ----------
    filepath : str
        Path to a LAMMPS data file describing the same atoms as the trajectory to be analysed.

    Returns
    -------
    ChainTopology
        Chains as 0-indexed positions into a `coords` array, in backbone order.

    Raises
    ------
    ValueError
        If the file states neither molecule ids nor bonds, if its atom ids are unusable, or if
        a bonded molecule is not a simple linear chain.

    Notes
    -----
    Molecule ids decide which atoms form a chain; bonds, where the file has them, decide the
    order of the beads along it. Ordering matters more than it looks: a Rouse-mode amplitude
    weights bead ``n`` by ``cos(p*pi*(n - 1/2)/N)``, so a chain held in the wrong order returns a
    curve that is wrong but entirely plausible.

    Examples
    --------
    >>> topology = chains_from_data_file("system.data")  # doctest: +SKIP
    >>> calculate_rouse_mode_acf_from_trajectory(  # doctest: +SKIP
    ...     "traj.npz", topology.chain_indices(), p=1, time_per_frame=1.0
    ... )
    """
    data = read_data_file(filepath)
    atoms = _atoms_section(data)
    ids = _atom_ids(data, atoms)
    positions = _positions(ids)
    adjacency = _adjacency(data, set(positions))
    style = data.metadata.get("atom style")
    notes: list[str] = []

    if style in STYLES_WITH_MOLECULE:
        source = "molecule"
        ordered = _order_by_molecule(
            _drop_short(_by_molecule(atoms, style, ids), notes), adjacency, notes
        )
    elif adjacency:
        source = "bonds"
        components = _components(adjacency)
        loose = len(positions) - sum(len(group) for group in components)
        ordered = _order_by_bonds(_drop_short(components, notes), adjacency)
        if loose:
            notes.append(f"{loose} atoms take part in no bond and were left out.")
    else:
        raise ValueError(
            "This file cannot say which atoms form a chain: its Atoms section declares style "
            f"'{style or 'no style'}', which carries no molecule id, and it has no Bonds "
            "section either. Add an 'Atoms # molecular' comment if column 2 really is the "
            "molecule id, include the bonds, or enter the chain layout by hand."
        )

    chains = [np.array([positions[atom] for atom in group], dtype=int) for group in ordered]
    chains.sort(key=lambda chain: int(chain[0]))
    return ChainTopology(chains=chains, n_atoms=len(positions), source=source, notes=notes)
