"""The read of a chain-topology data file shown before anything is calculated."""

from lada_web.main import TRAJECTORY_PREFIX

from .conftest import upload

URL = "/api/topology/preview"


def test_a_molecular_file_reports_its_chains(client, data_dir):
    r = client.post(URL, files=upload(data_dir / "sample_chains_molecular.data"))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_atoms"] == 32
    assert body["n_chains"] == 4
    assert body["beads_per_chain"] == 8
    assert body["source"] == "molecule"
    assert body["summary"] == "4 chains of 8 beads, from molecule ids"
    assert body["notes"] == []


def test_a_bonds_only_file_says_where_the_chains_came_from(client, data_dir):
    body = client.post(URL, files=upload(data_dir / "sample_chains_bonds_only.data")).json()

    assert body["source"] == "bonds"
    assert body["n_chains"] == 4


def test_unbonded_atoms_are_reported_rather_than_ignored(client, data_dir):
    """sample.data bonds only half its atoms; a preview that hid that would look like success."""
    body = client.post(URL, files=upload(data_dir / "sample.data")).json()

    assert body["n_atoms"] == 100
    assert body["n_chains"] == 1
    assert any("no bond" in note for note in body["notes"])


def test_the_length_spread_is_reported_for_a_polydisperse_file(client, tmp_path):
    rows = [f"{i} 1 1 {i}.0 0.0 0.0" for i in range(1, 5)]
    rows += [f"{i} 2 1 {i}.0 0.0 0.0" for i in range(5, 11)]
    path = tmp_path / "poly.data"
    path.write_text(
        "two chains of different length\n\n10 atoms\n\n1 atom types\n\n"
        "0.0 1.0 xlo xhi\n0.0 1.0 ylo yhi\n0.0 1.0 zlo zhi\n\n"
        "Atoms # molecular\n\n" + "\n".join(rows) + "\n"
    )
    body = client.post(URL, files=upload(path)).json()

    assert body["beads_per_chain"] is None
    assert body["shortest"] == 4
    assert body["longest"] == 6
    assert sorted(entry["beads"] for entry in body["lengths"]) == [4, 6]
    assert body["summary"] == "2 chains of varying lengths, from molecule ids"


def test_a_file_with_no_topology_at_all_is_422(client, tmp_path):
    path = tmp_path / "plain.data"
    path.write_text(
        "nothing to go on\n\n2 atoms\n\n1 atom types\n\n0.0 1.0 xlo xhi\n0.0 1.0 ylo yhi\n"
        "0.0 1.0 zlo zhi\n\nAtoms # atomic\n\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n"
    )
    r = client.post(URL, files=upload(path))

    assert r.status_code == 422
    assert "no molecule id" in r.json()["detail"]


def test_an_oversized_data_file_is_413(client, settings, temp_root):
    blob = b"x" * (int(settings.max_topology_bytes) + 1)
    r = client.post(URL, files={"file": ("huge.data", blob)})

    assert r.status_code == 413
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_the_preview_leaves_no_workspace_behind(client, data_dir, temp_root):
    r = client.post(URL, files=upload(data_dir / "sample_chains_molecular.data"))

    assert r.status_code == 200
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_the_preview_is_not_held_to_the_trajectory_body_ceiling():
    """It sits outside /api/dynamics/ so a text route does not inherit a multi-gigabyte ceiling."""
    assert not URL.startswith(TRAJECTORY_PREFIX)
