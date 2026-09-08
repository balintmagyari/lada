import io

import numpy as np
import pandas as pd

from .conftest import upload

LOG_URL = "/api/parsers/log/table"
LOG_COLUMNS = ["Step", "Temp", "Press", "Vol", "Lx", "Ly", "Lz", "PotEng", "KinEng", "TotEng"]


def test_log_table_json(client, data_dir):
    r = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == LOG_COLUMNS
    assert body["n_rows_total"] == 5 and len(body["rows"]) == 5
    assert body["rows"][0][0] == 0 and body["rows"][-1][0] == 4000
    assert body["download_stem"] == "sample_log_thermo"


def test_log_table_htmx_partial(client, data_dir, htmx):
    r = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"), headers=htmx)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<table" in r.text
    assert 'formaction="/api/parsers/log/table/download?fmt=csv"' in r.text
    assert 'formaction="/api/parsers/log/table/download?fmt=pgfplots"' in r.text
    assert 'form="calc-form"' in r.text


def test_log_download_csv_matches_view(client, data_dir):
    view = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps")).json()
    r = client.post(LOG_URL + "/download?fmt=csv", files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_log_thermo.csv"'
    df = pd.read_csv(io.BytesIO(r.content))
    assert list(df.columns) == view["columns"]
    assert np.allclose(df.to_numpy(), np.array(view["rows"], dtype=float))


def test_log_download_pgfplots_matches_view(client, data_dir):
    view = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps")).json()
    r = client.post(
        LOG_URL + "/download?fmt=pgfplots", files=upload(data_dir / "sample_log.lammps")
    )
    assert r.headers["content-disposition"] == 'attachment; filename="sample_log_thermo.dat"'
    assert r.text.splitlines()[0] == " ".join(view["columns"])
    data = np.loadtxt(io.BytesIO(r.content), skiprows=1)
    assert np.allclose(data, np.array(view["rows"], dtype=float))


def test_log_download_rejects_unknown_format(client, data_dir):
    r = client.post(LOG_URL + "/download?fmt=xlsx", files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 422


def test_missing_file_is_422(client, htmx):
    assert client.post(LOG_URL).status_code == 422
    r = client.post(LOG_URL, headers=htmx)
    assert r.status_code == 422 and 'role="alert"' in r.text and "file" in r.text


def test_not_a_log_file_is_422_with_message(client, data_dir, htmx):
    r = client.post(LOG_URL, files=upload(data_dir / "sample.data"))
    assert r.status_code == 422 and r.json()["detail"]
    r = client.post(LOG_URL, files=upload(data_dir / "sample.data"), headers=htmx)
    assert r.status_code == 422 and 'role="alert"' in r.text


def test_oversized_upload_is_413_and_leaves_nothing_behind(client, temp_root):
    blob = b"x" * (1024 * 1024 + 1)
    r = client.post(LOG_URL, files={"file": ("big.lammps", blob)})
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []
