"""Tests for lada.modifiers.rewrite_end_beads against the molecular-style fixture."""

from pathlib import Path

import numpy as np
import pytest

from lada.modifiers import rewrite_end_beads
from lada.parsers import read_data_file

DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURE = DATA / "sample_molecular.data"
CHAIN_ENDS = {1, 4, 5, 8, 9, 12}


@pytest.fixture
def rewritten(tmp_path):
    out = tmp_path / "rewritten.data"
    rewrite_end_beads(str(FIXTURE), str(out), new_end_type=3, base_type=1)
    return read_data_file(str(out))


def test_chain_ends_get_the_new_type(rewritten):
    atoms = rewritten.to_pandas("Atoms")
    typed = atoms.loc[atoms["type"] == 3, "id"].astype(int)
    assert set(typed) == CHAIN_ENDS


def test_interior_beads_keep_their_type(rewritten):
    original = read_data_file(str(FIXTURE)).to_pandas("Atoms")
    atoms = rewritten.to_pandas("Atoms")
    interior = ~atoms["id"].isin(CHAIN_ENDS)
    assert (
        atoms.loc[interior, "type"].to_numpy() == original.loc[interior, "type"].to_numpy()
    ).all()


def test_atom_types_header_and_coefficients_are_extended(rewritten):
    assert rewritten.metadata["atom types"] == 3
    masses = rewritten.to_pandas("Masses")
    assert masses.loc[masses["type"] == 3, "mass"].item() == pytest.approx(1.0)
    assert [3.0, 1.0, 1.0] in rewritten.get("Pair Coeffs").tolist()


def test_bonds_are_untouched(rewritten):
    original = read_data_file(str(FIXTURE))
    assert np.array_equal(rewritten.get("Bonds"), original.get("Bonds"))


def _rewrite(tmp_path, source_text, **kwargs):
    src = tmp_path / "in.data"
    src.write_text(source_text)
    out = tmp_path / "out.data"
    rewrite_end_beads(str(src), str(out), **kwargs)
    return out


class TestExistingOrGappedEndType:
    """An existing type duplicated Masses/Pair Coeffs; a gap gave an invalid file (D5)."""

    def test_existing_type_is_reused_without_duplicating_coefficients(self, tmp_path):
        out = _rewrite(tmp_path, FIXTURE.read_text(), new_end_type=2, base_type=1)
        data = read_data_file(str(out))
        assert data.metadata["atom types"] == 2
        assert data.get("Masses").tolist() == [[1.0, 1.0], [2.0, 2.0]]
        assert data.get("Pair Coeffs").tolist() == [[1.0, 1.0, 1.0], [2.0, 0.5, 1.2]]
        atoms = data.to_pandas("Atoms")
        assert set(atoms.loc[atoms["id"].isin(CHAIN_ENDS), "type"]) == {2.0}

    def test_new_end_type_equal_to_base_type_changes_nothing_but_the_ends(self, tmp_path):
        out = _rewrite(tmp_path, FIXTURE.read_text(), new_end_type=1, base_type=1)
        data = read_data_file(str(out))
        assert data.metadata["atom types"] == 2
        assert len(data.get("Masses")) == 2
        assert len(data.get("Pair Coeffs")) == 2

    @pytest.mark.parametrize("new_end_type", [4, 10])
    def test_a_gap_in_the_type_numbering_is_refused(self, tmp_path, new_end_type):
        out = tmp_path / "out.data"
        with pytest.raises(ValueError, match=r"gap.*2 atom types.*\(1 to 2\).*\(3\)"):
            rewrite_end_beads(str(FIXTURE), str(out), new_end_type=new_end_type, base_type=1)
        assert not out.exists()

    @pytest.mark.parametrize("new_end_type", [0, -1])
    def test_a_non_positive_type_is_refused(self, tmp_path, new_end_type):
        with pytest.raises(ValueError, match="positive"):
            rewrite_end_beads(
                str(FIXTURE), str(tmp_path / "o.data"), new_end_type=new_end_type, base_type=1
            )


class TestAtomTypesHeaderMatch:
    """A title containing "atom types" was taken for the header line (D6)."""

    def test_title_mentioning_atom_types_is_left_alone(self, tmp_path):
        text = FIXTURE.read_text().splitlines(keepends=True)
        text[0] = "2 atom types melt\n"
        out = _rewrite(tmp_path, "".join(text), new_end_type=3, base_type=1)
        lines = out.read_text().splitlines()
        assert lines[0] == "2 atom types melt"
        assert "3 atom types" in lines
        assert "2 atom types" not in lines[1:]
        data = read_data_file(str(out))
        assert data.metadata["atom types"] == 3
        assert data.get("Masses")[:, 0].tolist() == [1.0, 2.0, 3.0]

    def test_title_starting_with_a_section_name_is_not_a_section(self, tmp_path):
        text = FIXTURE.read_text().splitlines(keepends=True)
        text[0] = "Atoms of three chains\n"
        out = _rewrite(tmp_path, "".join(text), new_end_type=3, base_type=1)
        assert out.read_text().splitlines()[0] == "Atoms of three chains"
        atoms = read_data_file(str(out)).to_pandas("Atoms")
        assert set(atoms.loc[atoms["type"] == 3, "id"].astype(int)) == CHAIN_ENDS

    def test_file_without_atom_types_line_is_refused(self, tmp_path):
        text = FIXTURE.read_text().replace("2 atom types\n", "")
        with pytest.raises(ValueError, match="atom types"):
            _rewrite(tmp_path, text, new_end_type=3, base_type=1)


class TestOnlyRealChainEndsAreRetyped:
    """Mol-0 atoms and the atom of a one-atom molecule were retyped as ends (D7)."""

    def test_mol_zero_and_single_atom_molecules_keep_their_type(self, tmp_path):
        text = FIXTURE.read_text().replace("12 atoms", "15 atoms")
        text = text.replace(
            "12 3 1 6.0 -2.0 0.0\n",
            "12 3 1 6.0 -2.0 0.0\n13 0 1 9.0 9.0 9.0\n14 0 2 8.0 8.0 8.0\n15 4 1 7.0 7.0 7.0\n",
        )
        out = _rewrite(tmp_path, text, new_end_type=3, base_type=1)
        atoms = read_data_file(str(out)).to_pandas("Atoms")
        assert set(atoms.loc[atoms["type"] == 3, "id"].astype(int)) == CHAIN_ENDS
        extra = atoms[atoms["id"] >= 13].set_index("id")["type"].to_dict()
        assert extra == {13: 1.0, 14: 2.0, 15: 1.0}
