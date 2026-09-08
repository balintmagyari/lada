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
    assert masses.loc[masses["atom_id"] == 3, "mass"].item() == pytest.approx(1.0)
    assert [3.0, 1.0, 1.0] in rewritten.get("Pair Coeffs").tolist()


def test_bonds_are_untouched(rewritten):
    original = read_data_file(str(FIXTURE))
    assert np.array_equal(rewritten.get("Bonds"), original.get("Bonds"))
