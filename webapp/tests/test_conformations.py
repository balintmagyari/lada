import io

import numpy as np
import pandas as pd
from lada.analysis import calculate_avg_ree_sq, calculate_avg_rg_sq
from lada.parsers import read_dump

from .conftest import upload

RG_URL = "/api/conformations/rg"
REE_URL = "/api/conformations/ree"
REE_VECTORS_URL = "/api/conformations/ree-vectors"


def _expected_per_timestep(path, func, **kwargs):
    """Run a `lada.analysis` function directly and return (timesteps, values) in order."""
    df = read_dump(str(path))
    values = func(df, **kwargs)
    timesteps = sorted(values.keys())
    return timesteps, [values[ts] for ts in timesteps]


def test_rg_json_matches_lada(client, data_dir):
    path = data_dir / "sample_dump.dump"
    r = client.post(RG_URL, files=upload(path))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["timestep", "rg_sq"]
    assert body["n_rows_total"] == 3
    assert body["download_stem"] == "sample_dump_rg_sq"

    expected_timesteps, expected_values = _expected_per_timestep(
        path, calculate_avg_rg_sq, coord_cols=["xu", "yu", "zu"], molecule_col="mol", mass_col=None
    )
    got_timesteps = [row[0] for row in body["rows"]]
    got_values = [row[1] for row in body["rows"]]
    assert np.allclose(got_timesteps, expected_timesteps)
    assert np.allclose(got_values, expected_values)


def test_rg_htmx_partial_has_plot_and_downloads(client, data_dir, htmx):
    r = client.post(RG_URL, files=upload(data_dir / "sample_dump.dump"), headers=htmx)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="result-plot"' in r.text
    assert 'formaction="/api/conformations/rg/download?fmt=csv"' in r.text
    assert 'formaction="/api/conformations/rg/download?fmt=pgfplots"' in r.text
    assert 'form="calc-form"' in r.text


def test_rg_missing_mass_column_is_422_with_library_message(client, data_dir):
    r = client.post(RG_URL, files=upload(data_dir / "sample_dump.dump"), data={"mass_col": "mass"})
    assert r.status_code == 422
    assert r.json()["detail"] == "Mass column 'mass' not found in data."


def test_rg_invalid_coord_cols_is_422(client, data_dir):
    r = client.post(
        RG_URL, files=upload(data_dir / "sample_dump.dump"), data={"coord_cols": "xu,yu"}
    )
    assert r.status_code == 422
    assert r.json()["detail"] == (
        "Coordinate columns must be three comma-separated names, for example xu,yu,zu."
    )


def test_ree_json_matches_lada(client, data_dir):
    path = data_dir / "sample_dump.dump"
    r = client.post(REE_URL, files=upload(path))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["timestep", "ree_sq"]
    assert body["n_rows_total"] == 3
    assert body["download_stem"] == "sample_dump_ree_sq"

    expected_timesteps, expected_values = _expected_per_timestep(
        path,
        calculate_avg_ree_sq,
        coord_cols=["xu", "yu", "zu"],
        molecule_col="mol",
        atom_id_col="id",
    )
    got_timesteps = [row[0] for row in body["rows"]]
    got_values = [row[1] for row in body["rows"]]
    assert np.allclose(got_timesteps, expected_timesteps)
    assert np.allclose(got_values, expected_values)


def test_ree_vectors_json_shape(client, data_dir):
    path = data_dir / "sample_dump.dump"
    r = client.post(REE_VECTORS_URL, files=upload(path))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["timestep", "mol", "dx", "dy", "dz"]
    assert body["download_stem"] == "sample_dump_ree_vectors"

    df = read_dump(str(path))
    n_molecules = df["mol"].nunique()
    assert body["n_rows_total"] == 3 * n_molecules


def test_rg_download_csv_matches_view(client, data_dir):
    view = client.post(RG_URL, files=upload(data_dir / "sample_dump.dump")).json()
    r = client.post(RG_URL + "/download?fmt=csv", files=upload(data_dir / "sample_dump.dump"))
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_dump_rg_sq.csv"'
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert list(df.columns) == view["columns"]
    # Exact, not allclose: a download must carry the same float64 the JSON view carries.
    assert np.array_equal(df.to_numpy(), np.array(view["rows"], dtype=float))


def test_rg_download_pgfplots_matches_view(client, data_dir):
    view = client.post(RG_URL, files=upload(data_dir / "sample_dump.dump")).json()
    r = client.post(RG_URL + "/download?fmt=pgfplots", files=upload(data_dir / "sample_dump.dump"))
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_dump_rg_sq.dat"'
    assert r.text.splitlines()[0] == " ".join(view["columns"])
    data = np.loadtxt(io.BytesIO(r.content), skiprows=1)
    assert np.array_equal(data, np.array(view["rows"], dtype=float))


def test_conformation_pages_render_with_calc_form(client):
    for path in ("/conformations/rg", "/conformations/ree", "/conformations/ree-vectors"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert 'id="calc-form"' in r.text
