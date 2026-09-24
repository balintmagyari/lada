"""Chains recovered from a LAMMPS data file.

The committed fixtures are sized to `tests/data/sample_trajectory.npz` (4 chains of 8 beads,
32 atoms) so the web app can assert that a data-file run and a hand-typed uniform run agree
number for number. Everything odd -- rings, branches, ragged chains, solvent -- is built in
tmp_path instead, because those files exist only to be refused.
"""

from pathlib import Path

import numpy as np
import pytest

from lada import chains_from_data_file
from lada.analysis.topology import RAGGED_CHAINS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MOLECULAR = DATA_DIR / "sample_chains_molecular.data"
BONDS_ONLY = DATA_DIR / "sample_chains_bonds_only.data"

# The layout tests/data/sample_trajectory.npz was generated with.
FIXTURE_PAIRS = [[0, 7], [8, 15], [16, 23], [24, 31]]


def build(path, *, style, atoms, bonds=(), declared_atoms=None):
    """Write a minimal LAMMPS data file from ready-made Atoms rows and (atom, atom) bonds."""
    bond_rows = "\n".join(f"{i} 1 {a} {b}" for i, (a, b) in enumerate(bonds, start=1))
    header = (
        f"topology test\n\n{declared_atoms if declared_atoms is not None else len(atoms)} atoms\n"
        f"{len(bonds)} bonds\n\n1 atom types\n1 bond types\n\n"
        "-20.0 20.0 xlo xhi\n-20.0 20.0 ylo yhi\n-20.0 20.0 zlo zhi\n\nMasses\n\n1 1.0\n\n"
    )
    section = f"Atoms # {style}\n\n" + "\n".join(atoms) + "\n"
    if bonds:
        section += "\nBonds\n\n" + bond_rows + "\n"
    path.write_text(header + section)
    return str(path)


def linear(n_chains, beads, style="molecular", first_id=1):
    """Rows for `n_chains` chains whose atom ids ascend along each backbone."""
    rows, bonds = [], []
    for c in range(n_chains):
        for k in range(beads):
            atom = first_id + c * beads + k
            rows.append(
                f"{atom} 1 {atom}.0 0.0 0.0"
                if style == "atomic"
                else f"{atom} {c + 1} 1 {atom}.0 0.0 0.0"
            )
            if k:
                bonds.append((atom - 1, atom))
    return rows, bonds


class TestMoleculeIds:
    def test_the_committed_fixture_matches_the_trajectory_layout(self):
        topology = chains_from_data_file(str(MOLECULAR))

        assert topology.source == "molecule"
        assert topology.n_atoms == 32
        assert topology.segment_pairs().tolist() == FIXTURE_PAIRS
        assert topology.notes == []

    def test_chain_indices_are_the_consecutive_blocks(self):
        indices = chains_from_data_file(str(MOLECULAR)).chain_indices()

        assert indices.shape == (4, 8)
        assert indices[0].tolist() == list(range(8))
        assert indices[3].tolist() == list(range(24, 32))

    def test_angle_style_carries_a_molecule_id_too(self, tmp_path):
        rows, bonds = linear(2, 4)
        path = build(tmp_path / "a.data", style="angle", atoms=rows, bonds=bonds)

        assert chains_from_data_file(path).segment_pairs().tolist() == [[0, 3], [4, 7]]

    def test_without_bonds_the_bead_order_is_a_stated_guess(self, tmp_path):
        rows, _ = linear(2, 4)
        path = build(tmp_path / "nb.data", style="molecular", atoms=rows)
        topology = chains_from_data_file(path)

        assert topology.segment_pairs().tolist() == [[0, 3], [4, 7]]
        assert any("atom-id order" in note for note in topology.notes)

    def test_bonds_order_the_beads_when_the_ids_do_not(self, tmp_path):
        """Molecule ids group; bonds order. A Rouse mode is wrong, not noisy, if this slips."""
        rows = [f"{i} 1 1 {i}.0 0.0 0.0" for i in (1, 2, 3, 4)]
        path = build(
            tmp_path / "s.data", style="molecular", atoms=rows, bonds=[(3, 1), (1, 4), (4, 2)]
        )
        topology = chains_from_data_file(path)

        # Backbone 3-1-4-2, walked from the lower-id end: 2,4,1,3 -> positions 1,3,0,2. Which
        # end it starts from is arbitrary and harmless (both ACFs are normalised, so a chain
        # read backwards gives the same curve); what matters is that an id sort would have
        # claimed ends 0 and 3, which are interior beads.
        assert topology.chains[0].tolist() == [1, 3, 0, 2]
        assert topology.segment_pairs().tolist() == [[1, 2]]

    def test_a_molecule_that_is_not_a_strand_falls_back_and_says_so(self, tmp_path):
        """A ring still has a molecule id to group by, so it is ordered by id, not refused."""
        rows = [f"{i} 1 1 {i}.0 0.0 0.0" for i in range(1, 5)]
        ring = [(1, 2), (2, 3), (3, 4), (4, 1)]
        path = build(tmp_path / "r.data", style="molecular", atoms=rows, bonds=ring)
        topology = chains_from_data_file(path)

        assert topology.chains[0].tolist() == [0, 1, 2, 3]
        assert any("not a single bonded strand" in note for note in topology.notes)


class TestBondGraph:
    def test_the_walk_beats_an_id_sort_on_the_committed_fixture(self):
        """Chain 4's ids deliberately run out of backbone order; only the walk gets it right."""
        topology = chains_from_data_file(str(BONDS_ONLY))

        assert topology.source == "bonds"
        assert topology.lengths == [8, 8, 8, 8]
        assert topology.chains[3].tolist() == [26, 24, 28, 25, 30, 27, 31, 29]

    def test_that_chains_ends_are_not_the_lowest_and_highest_index(self):
        pairs = chains_from_data_file(str(BONDS_ONLY)).segment_pairs().tolist()

        assert pairs[:3] == FIXTURE_PAIRS[:3]
        assert pairs[3] == [26, 29]
        assert pairs[3] != FIXTURE_PAIRS[3]

    def test_a_ring_is_refused_by_name(self, tmp_path):
        rows = [f"{i} 1 {i}.0 0.0 0.0" for i in range(1, 5)]
        path = build(
            tmp_path / "r.data", style="atomic", atoms=rows, bonds=[(1, 2), (2, 3), (3, 4), (4, 1)]
        )
        with pytest.raises(ValueError, match="closed ring"):
            chains_from_data_file(path)

    def test_a_branch_names_the_atom_it_branches_at(self, tmp_path):
        rows = [f"{i} 1 {i}.0 0.0 0.0" for i in range(1, 5)]
        path = build(
            tmp_path / "b.data", style="atomic", atoms=rows, bonds=[(1, 2), (2, 3), (2, 4)]
        )
        with pytest.raises(ValueError, match="atom 2 has 3 bonds"):
            chains_from_data_file(path)

    def test_unbonded_atoms_are_left_out_and_counted(self, tmp_path):
        rows, bonds = linear(2, 4, style="atomic")
        rows += [f"{i} 1 0.0 0.0 0.0" for i in (9, 10)]
        path = build(tmp_path / "loose.data", style="atomic", atoms=rows, bonds=bonds)
        topology = chains_from_data_file(path)

        assert topology.lengths == [4, 4]
        assert topology.n_atoms == 10
        assert any("no bond" in note for note in topology.notes)

    def test_a_bond_naming_an_unknown_atom_is_refused(self, tmp_path):
        rows = [f"{i} 1 {i}.0 0.0 0.0" for i in (1, 2)]
        path = build(tmp_path / "x.data", style="atomic", atoms=rows, bonds=[(1, 2), (2, 99)])
        with pytest.raises(ValueError, match="atoms 2 and 99"):
            chains_from_data_file(path)


class TestRefusals:
    def test_neither_molecule_ids_nor_bonds_names_both(self, tmp_path):
        rows = [f"{i} 1 {i}.0 0.0 0.0" for i in (1, 2, 3)]
        path = build(tmp_path / "n.data", style="atomic", atoms=rows)
        with pytest.raises(ValueError, match="no molecule id") as caught:
            chains_from_data_file(path)

        assert "no Bonds" in str(caught.value)
        assert "by hand" in str(caught.value)

    def test_a_repeated_atom_id_is_refused(self, tmp_path):
        rows = ["1 1 1 0.0 0.0 0.0", "2 1 1 1.0 0.0 0.0", "2 1 1 2.0 0.0 0.0"]
        path = build(tmp_path / "d.data", style="molecular", atoms=rows)
        with pytest.raises(ValueError, match="Atom id 2 appears more than once"):
            chains_from_data_file(path)

    def test_a_header_that_disagrees_with_the_rows_is_refused(self, tmp_path):
        rows, bonds = linear(1, 4)
        path = build(
            tmp_path / "t.data", style="molecular", atoms=rows, bonds=bonds, declared_atoms=40
        )
        with pytest.raises(ValueError, match="declares 40 atoms but the Atoms section holds 4"):
            chains_from_data_file(path)

    def test_a_file_of_nothing_but_solvent_is_refused(self, tmp_path):
        rows = [f"{i} {i} 1 0.0 0.0 0.0" for i in range(1, 4)]
        path = build(tmp_path / "w.data", style="molecular", atoms=rows)
        with pytest.raises(ValueError, match="no chains"):
            chains_from_data_file(path)


class TestMixedSystems:
    def test_short_molecules_are_dropped_and_counted(self, tmp_path):
        """A melt in water: the water must not become thousands of tiny "chains"."""
        rows, bonds = linear(2, 4)
        rows += [f"{9 + i} {3 + i} 1 0.0 0.0 0.0" for i in range(4)]
        path = build(tmp_path / "mix.data", style="molecular", atoms=rows, bonds=bonds)
        topology = chains_from_data_file(path)

        assert topology.lengths == [4, 4]
        assert topology.n_atoms == 12
        assert any("fewer than 2 beads" in note for note in topology.notes)

    def test_chains_may_cover_only_part_of_the_file(self, tmp_path):
        """Unlike the uniform form, a data file says which atoms are chains, so a subset is fine."""
        rows, bonds = linear(2, 4)
        rows += [f"{9 + i} {3 + i} 1 0.0 0.0 0.0" for i in range(4)]
        path = build(tmp_path / "mix.data", style="molecular", atoms=rows, bonds=bonds)
        topology = chains_from_data_file(path)

        covered = np.concatenate(topology.chains)
        assert covered.max() < topology.n_atoms
        assert len(covered) < topology.n_atoms

    def test_ragged_chains_give_pairs_but_not_a_rouse_array(self, tmp_path):
        rows = [f"{i} 1 1 {i}.0 0.0 0.0" for i in range(1, 5)]
        rows += [f"{i} 2 1 {i}.0 0.0 0.0" for i in range(5, 8)]
        bonds = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
        path = build(tmp_path / "rag.data", style="molecular", atoms=rows, bonds=bonds)
        topology = chains_from_data_file(path)

        assert topology.lengths == [4, 3]
        assert topology.segment_pairs().tolist() == [[0, 3], [4, 6]]
        with pytest.raises(ValueError, match=RAGGED_CHAINS):
            topology.chain_indices()

    def test_positions_are_ranks_not_atom_ids(self, tmp_path):
        """coords[i] is the i-th smallest atom id, so ids starting at 101 still index from 0."""
        rows, bonds = linear(2, 4, first_id=101)
        path = build(tmp_path / "off.data", style="molecular", atoms=rows, bonds=bonds)
        topology = chains_from_data_file(path)

        assert topology.segment_pairs().tolist() == [[0, 3], [4, 7]]

    def test_gaps_in_the_atom_ids_close_up(self, tmp_path):
        rows = ["7 1 1 0.0 0.0 0.0", "300 1 1 1.0 0.0 0.0", "42 1 1 2.0 0.0 0.0"]
        path = build(
            tmp_path / "gap.data", style="molecular", atoms=rows, bonds=[(7, 42), (42, 300)]
        )
        topology = chains_from_data_file(path)

        # Ranks: 7 -> 0, 42 -> 1, 300 -> 2. The walk runs 7-42-300.
        assert topology.chains[0].tolist() == [0, 1, 2]
