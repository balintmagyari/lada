"""The end-type checks of the core modifier reach the user as readable 422s (audit D5-D7)."""

from lada.parsers import read_data_file

from .conftest import upload

URL = "/api/modifiers/rewrite-end-beads"
END_ATOM_IDS = {1, 4, 5, 8, 9, 12}


def test_a_gap_in_the_type_numbering_is_422_naming_the_allowed_types(client, data_dir):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 5, "base_type": 1},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "gap" in detail
    assert "(1 to 2)" in detail and "(3)" in detail


def test_an_existing_type_is_reused_without_duplicate_coefficients(client, data_dir, tmp_path):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 2, "base_type": 1},
    )
    assert r.status_code == 200, r.text
    out = tmp_path / "out.data"
    out.write_bytes(r.content)
    data = read_data_file(str(out))
    assert data.metadata["atom types"] == 2
    assert data.get("Masses")[:, 0].tolist() == [1.0, 2.0]
    assert data.get("Pair Coeffs")[:, 0].tolist() == [1.0, 2.0]
    type_two = {int(row[0]) for row in data.get("Atoms") if int(row[2]) == 2}
    assert type_two == END_ATOM_IDS | {6}  # atom 6 is an interior bead already of type 2


def test_a_title_mentioning_atom_types_is_not_the_header(client, data_dir, tmp_path):
    src = tmp_path / "titled.data"
    lines = (data_dir / "sample_molecular.data").read_text().splitlines(keepends=True)
    lines[0] = "2 atom types melt\n"
    src.write_text("".join(lines))
    r = client.post(URL, files=upload(src), data={"new_end_type": 3, "base_type": 1})
    assert r.status_code == 200, r.text
    out = tmp_path / "out.data"
    out.write_bytes(r.content)
    assert out.read_text().splitlines()[0] == "2 atom types melt"
    assert read_data_file(str(out)).metadata["atom types"] == 3
