import numpy as np
from lada.parsers import read_data_file

from .conftest import upload

URL = "/api/modifiers/rewrite-end-beads"
END_ATOM_IDS = {1, 4, 5, 8, 9, 12}


def test_happy_path_downloads_rewritten_file(client, data_dir, tmp_path):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 3, "base_type": 1},
    )
    assert r.status_code == 200, r.text
    assert (
        r.headers["content-disposition"] == 'attachment; filename="sample_molecular_endbeads.data"'
    )
    assert r.headers["content-type"].startswith("text/plain")

    out_path = tmp_path / "out.data"
    out_path.write_bytes(r.content)
    data = read_data_file(str(out_path))
    assert data.metadata["atom types"] == 3
    atoms = np.atleast_2d(data.sections["Atoms"])
    for row in atoms:
        atom_id, atom_type = int(row[0]), int(row[2])
        assert atom_type == 3 if atom_id in END_ATOM_IDS else atom_type != 3


def test_atomic_style_file_is_422_mentioning_molecular(client, data_dir):
    r = client.post(
        URL,
        files=upload(data_dir / "sample.data"),
        data={"new_end_type": 3, "base_type": 1},
    )
    assert r.status_code == 422
    assert "molecular" in r.json()["detail"]


def test_unknown_base_type_is_422_mentioning_base_type(client, data_dir):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 3, "base_type": 9},
    )
    assert r.status_code == 422
    assert "base type 9" in r.json()["detail"]


def test_new_end_type_zero_is_422(client, data_dir):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 0, "base_type": 1},
    )
    assert r.status_code == 422


def test_browser_submit_of_bad_file_returns_html_error_page(client, data_dir):
    r = client.post(
        URL,
        files=upload(data_dir / "sample.data"),
        data={"new_end_type": 3, "base_type": 1},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("text/html")
    assert "molecular" in r.text


def test_oversized_upload_is_413_and_leaves_nothing_behind(client, temp_root):
    blob = b"x" * (1024 * 1024 + 1)
    r = client.post(
        URL,
        files={"file": ("big.data", blob)},
        data={"new_end_type": 3, "base_type": 1},
    )
    assert r.status_code == 413
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_page_has_plain_submit_button_and_no_htmx_on_form(client):
    r = client.get("/modifiers/rewrite-end-beads")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert f'formaction="{URL}"' in r.text
    assert "hx-post" not in r.text
    assert 'name="file"' in r.text
    assert 'name="new_end_type"' in r.text
    assert 'name="base_type"' in r.text


def test_happy_path_leaves_temp_root_empty(client, data_dir, temp_root):
    r = client.post(
        URL,
        files=upload(data_dir / "sample_molecular.data"),
        data={"new_end_type": 3, "base_type": 1},
    )
    assert r.status_code == 200
    assert not temp_root.exists() or list(temp_root.iterdir()) == []
