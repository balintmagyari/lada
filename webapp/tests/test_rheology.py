import io

import numpy as np
import pandas as pd
from lada.analysis import calc_dynamic_moduli_prony, calc_stress_relaxation
from lada.parsers import read_lammps_acf

from .conftest import upload

G_URL = "/api/rheology/stress-relaxation"
MODULI_URL = "/api/rheology/dynamic-moduli"


def _acf_df(data_dir):
    return read_lammps_acf(str(data_dir / "sample_stress_acf.txt"))


def test_stress_relaxation_json_matches_direct_call(client, data_dir):
    r = client.post(
        G_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "G_GK", "G_FSR"]
    assert body["n_rows_total"] == 100 and len(body["rows"]) == 20  # capped by max_table_rows
    assert body["download_stem"] == "sample_stress_acf_G_t"

    expected = calc_stress_relaxation(_acf_df(data_dir), 1000, 1)
    shown = np.array(body["rows"], dtype=float)
    assert np.allclose(shown, expected.to_numpy()[: len(body["rows"])])
    # The plot must carry every row, not just the shown/capped ones.
    assert body["plot"]["x"] == "lag_time"
    assert body["plot"]["y"] == ["G_GK", "G_FSR"]
    assert np.allclose(body["plot"]["series"]["G_GK"], expected["G_GK"].to_numpy())
    assert np.allclose(body["plot"]["series"]["G_FSR"], expected["G_FSR"].to_numpy())
    assert body["plot"]["x_log"] is False and body["plot"]["y_log"] is False


def test_stress_relaxation_non_positive_volume_is_422_validation_list(client, data_dir):
    r = client.post(
        G_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "0", "temperature": "1"},
    )
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_stress_relaxation_wrong_acf_schema_is_422_naming_missing_column(client, data_dir):
    r = client.post(
        G_URL,
        files=upload(data_dir / "sample_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 422
    assert "ACF_Sxy" in r.json()["detail"]


def test_dynamic_moduli_defaults(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["omega", "G_prime", "G_dprime"]
    assert body["n_rows_total"] == 200
    omega = body["plot"]["series"]["omega"]
    assert all(a < b for a, b in zip(omega, omega[1:], strict=False))
    assert body["plot"]["x"] == "omega"
    assert body["plot"]["y"] == ["G_prime", "G_dprime"]
    assert body["plot"]["x_log"] is True and body["plot"]["y_log"] is True
    assert body["download_stem"] == "sample_stress_acf_moduli"


def test_dynamic_moduli_matches_direct_call(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    body = r.json()
    g_t = calc_stress_relaxation(_acf_df(data_dir), 1000, 1)
    expected = calc_dynamic_moduli_prony(g_t)
    assert np.allclose(body["plot"]["series"]["omega"], expected["omega"].to_numpy())
    assert np.allclose(body["plot"]["series"]["G_prime"], expected["G_prime"].to_numpy())
    assert np.allclose(body["plot"]["series"]["G_dprime"], expected["G_dprime"].to_numpy())


def test_dynamic_moduli_method_both_yields_five_suffixed_columns(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1", "method": "both"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == [
        "omega",
        "G_prime_GK",
        "G_dprime_GK",
        "G_prime_FSR",
        "G_dprime_FSR",
    ]


def test_dynamic_moduli_n_modes_too_large_is_422_with_library_message(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1", "n_modes": "500"},
    )
    assert r.status_code == 422
    assert "Reduce n_modes" in r.json()["detail"]


def test_dynamic_moduli_empty_t_cutoff_is_accepted(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1", "t_cutoff": ""},
    )
    assert r.status_code == 200, r.text


def test_dynamic_moduli_pgfplots_download_matches_view(client, data_dir):
    view = client.post(
        MODULI_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    ).json()
    r = client.post(
        MODULI_URL + "/download?fmt=pgfplots",
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"] == 'attachment; filename="sample_stress_acf_moduli.dat"'
    assert r.text.splitlines()[0] == " ".join(view["columns"])
    data = np.loadtxt(io.BytesIO(r.content), skiprows=1)
    assert np.allclose(data[:, 0], view["plot"]["series"]["omega"])


def test_stress_relaxation_csv_download_matches_view(client, data_dir):
    view = client.post(
        G_URL,
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    ).json()
    r = client.post(
        G_URL + "/download?fmt=csv",
        files=upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_stress_acf_G_t.csv"'
    df = pd.read_csv(io.BytesIO(r.content))
    assert list(df.columns) == view["columns"]
    assert np.allclose(df["lag_time"].to_numpy(), view["plot"]["series"]["lag_time"])


def test_stress_relaxation_page_renders(client):
    r = client.get("/rheology/stress-relaxation")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'hx-post="/api/rheology/stress-relaxation"' in r.text


def test_dynamic_moduli_page_renders(client):
    r = client.get("/rheology/dynamic-moduli")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'hx-post="/api/rheology/dynamic-moduli"' in r.text
