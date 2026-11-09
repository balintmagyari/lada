import io

import numpy as np
import pandas as pd
from lada.analysis import calc_dynamic_moduli_prony, calc_stress_relaxation
from lada.parsers import read_lammps_acf

from .conftest import files_upload

G_URL = "/api/rheology/stress-relaxation"
MODULI_URL = "/api/rheology/dynamic-moduli"


def _acf_df(data_dir):
    return read_lammps_acf(str(data_dir / "sample_stress_acf.txt"))


def test_stress_relaxation_json_matches_direct_call(client, data_dir):
    r = client.post(
        G_URL,
        files=files_upload(data_dir / "sample_stress_acf.txt"),
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
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "0", "temperature": "1"},
    )
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_stress_relaxation_wrong_acf_schema_is_422_naming_missing_column(client, data_dir):
    r = client.post(
        G_URL,
        files=files_upload(data_dir / "sample_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 422
    assert "ACF_Sxy" in r.json()["detail"]


def test_stress_relaxation_multiple_files_averages_across_repeats(client, data_dir):
    a = data_dir / "sample_stress_acf.txt"
    b = data_dir / "sample_stress_acf_repeat2.txt"
    form = {"volume": "1000", "temperature": "1"}
    single_a = client.post(G_URL, files=files_upload(a), data=form).json()
    single_b = client.post(G_URL, files=files_upload(b), data=form).json()
    combined = client.post(G_URL, files=files_upload(a, b), data=form).json()
    assert combined["meta"]["n_files"] == 2
    assert combined["download_stem"] == "sample_stress_acf_avg2_G_t"
    expected_g_gk = (
        np.array(single_a["rows"])[: combined["n_rows_total"], 1]
        + np.array(single_b["rows"])[: combined["n_rows_total"], 1]
    ) / 2
    # rows are capped by max_table_rows; the plot series carry every averaged row.
    assert np.allclose(combined["plot"]["series"]["G_GK"][:20], expected_g_gk[:20])


def test_stress_relaxation_rejects_mismatched_lag_time_grids(client, data_dir):
    a = data_dir / "sample_stress_acf.txt"
    b = data_dir / "sample_acf.txt"
    form = {"volume": "1000", "temperature": "1"}
    r = client.post(G_URL, files=files_upload(a, b), data=form)
    # sample_acf.txt lacks the ACF_Sxy-style columns entirely, so this fails on schema first —
    # confirms per-file validation runs before any cross-file averaging is attempted.
    assert r.status_code == 422
    assert "ACF_Sxy" in r.json()["detail"]


def test_dynamic_moduli_defaults(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=files_upload(data_dir / "sample_stress_acf.txt"),
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
        files=files_upload(data_dir / "sample_stress_acf.txt"),
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
        files=files_upload(data_dir / "sample_stress_acf.txt"),
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
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1", "n_modes": "200"},
    )
    assert r.status_code == 422
    assert "Reduce n_modes" in r.json()["detail"]


def test_dynamic_moduli_empty_t_cutoff_is_accepted(client, data_dir):
    r = client.post(
        MODULI_URL,
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1", "t_cutoff": ""},
    )
    assert r.status_code == 200, r.text


def test_dynamic_moduli_pgfplots_download_matches_view(client, data_dir):
    view = client.post(
        MODULI_URL,
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    ).json()
    r = client.post(
        MODULI_URL + "/download?fmt=pgfplots",
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"] == 'attachment; filename="sample_stress_acf_moduli.dat"'
    assert r.text.splitlines()[0] == " ".join(view["columns"])
    data = np.loadtxt(io.BytesIO(r.content), skiprows=1)
    assert np.array_equal(data[:, 0], view["plot"]["series"]["omega"])


def test_stress_relaxation_csv_download_matches_view(client, data_dir):
    view = client.post(
        G_URL,
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    ).json()
    r = client.post(
        G_URL + "/download?fmt=csv",
        files=files_upload(data_dir / "sample_stress_acf.txt"),
        data={"volume": "1000", "temperature": "1"},
    )
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_stress_acf_G_t.csv"'
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert list(df.columns) == view["columns"]
    # Exact, not allclose: a download must carry the same float64 the JSON view carries.
    assert np.array_equal(df["lag_time"].to_numpy(), view["plot"]["series"]["lag_time"])
    assert np.array_equal(df["G_GK"].to_numpy(), view["plot"]["series"]["G_GK"])
    assert np.array_equal(df["G_FSR"].to_numpy(), view["plot"]["series"]["G_FSR"])


def test_stress_relaxation_page_renders(client):
    r = client.get("/rheology/stress-relaxation")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'data-start-url="/api/rheology/stress-relaxation/start"' in r.text


def test_dynamic_moduli_page_renders(client):
    r = client.get("/rheology/dynamic-moduli")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'data-start-url="/api/rheology/dynamic-moduli/start"' in r.text
