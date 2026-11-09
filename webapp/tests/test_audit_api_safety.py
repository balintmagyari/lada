"""Regression tests for the API-safety audit findings (E1, E2, E4, D8/E7, E8, D10).

Each test names the finding it pins down, so a failure points straight back at the report.
"""

import json

import pandas as pd
import pytest
from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.jobs import JobStore
from lada_web.main import create_app
from lada_web.services import merge
from lada_web.services.export_service import write_table

DYNAMICS_MERGE_URL = "/api/dynamics/merge-regimes"
RHEOLOGY_MERGE_URL = "/api/rheology/merge-regimes"
MODULI_FROM_REGIMES_URL = "/api/rheology/dynamic-moduli/from-regimes"
MIB = 1024 * 1024
TOO_LARGE = "The request body is larger than this server accepts."


def dynamics_body(stem: str = "merged", **overrides) -> dict:
    body = {
        "regimes": [
            {"label": "fast", "lag_time": [0.0, 1.0], "value": [1.0, 0.5]},
            {"label": "slow", "lag_time": [0.0, 10.0, 20.0], "value": [1.0, 0.2, 0.1]},
        ],
        "value_column": "acf",
        "title": "t",
        "y_label": "y",
        "download_stem": stem,
    }
    return body | overrides


def rheology_regimes(n: int = 40) -> list[dict]:
    lag = [0.1 * (i + 1) for i in range(n)]
    g = [float(2.0 ** (-t)) for t in lag]
    return [{"label": "r1", "lag_time": lag, "G_GK": g, "G_FSR": g}]


def disposition_name(response) -> str:
    return response.headers["content-disposition"].split("filename=", 1)[1].strip('"')


# --- E1: download_stem path traversal ------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "pgfplots"])
def test_e1_absolute_stem_writes_nothing_outside_the_workspace(client, tmp_path, fmt):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "pwned"
    r = client.post(
        f"{DYNAMICS_MERGE_URL}/download?fmt={fmt}", json=dynamics_body(str(target))
    )
    assert r.status_code == 200, r.text
    assert list(outside.iterdir()) == []
    name = disposition_name(r)
    assert "/" not in name and ".." not in name
    assert name.startswith("pwned")


@pytest.mark.parametrize(
    "url, body",
    [
        (f"{DYNAMICS_MERGE_URL}/download", dynamics_body("../../pwned_rel")),
        (
            f"{RHEOLOGY_MERGE_URL}/download",
            {"regimes": rheology_regimes(), "download_stem": "../../pwned_rel"},
        ),
        (
            f"{MODULI_FROM_REGIMES_URL}/download",
            {
                "regimes": rheology_regimes(),
                "t_min": 0.1,
                "n_modes": 5,
                "download_stem": "../../pwned_rel",
            },
        ),
    ],
)
def test_e1_relative_stem_cannot_climb_out_on_any_merge_download(
    client, temp_root, tmp_path, url, body
):
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    assert disposition_name(r) == "pwned_rel.csv"
    # Nothing landed anywhere above the per-request workspace.
    assert not list(tmp_path.rglob("pwned_rel*"))


def test_e1_view_route_echoes_the_sanitised_stem(client):
    r = client.post(DYNAMICS_MERGE_URL, json=dynamics_body("/etc/../../x y"))
    assert r.status_code == 200, r.text
    assert r.json()["download_stem"] == "x_y"


@pytest.mark.parametrize("stem", ["../escape", "/abs/escape", "sub/escape"])
def test_e1_write_table_refuses_a_path_outside_its_directory(tmp_path, stem):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError):
        write_table(pd.DataFrame({"a": [1]}), ws, stem, "csv")
    assert not list(tmp_path.rglob("escape*"))


# --- E2: JSON bodies get their own ceiling --------------------------------------------


def big_json(size: int) -> bytes:
    body = dynamics_body()
    body["title"] = "x" * size
    return json.dumps(body).encode()


def test_e2_six_megabyte_json_body_is_413(client):
    r = client.post(
        DYNAMICS_MERGE_URL, content=big_json(6 * MIB), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413
    assert r.json()["detail"] == TOO_LARGE


def test_e2_json_route_keeps_the_json_ceiling_whatever_the_content_type(client):
    r = client.post(
        f"{RHEOLOGY_MERGE_URL}/download",
        content=big_json(6 * MIB),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413


def test_e2_small_json_body_still_works(client):
    r = client.post(DYNAMICS_MERGE_URL, json=dynamics_body())
    assert r.status_code == 200, r.text


def test_e2_json_ceiling_is_configurable(temp_root):
    small = TestClient(
        create_app(Settings(max_json_mb=0.0001, temp_root=temp_root)),
        raise_server_exceptions=False,
    )
    r = small.post(DYNAMICS_MERGE_URL, json=dynamics_body())
    assert r.status_code == 413


def test_e2_multipart_uploads_keep_their_own_larger_limit(temp_root):
    roomy = TestClient(
        create_app(Settings(max_upload_mb=10, temp_root=temp_root)),
        raise_server_exceptions=False,
    )
    payload = b"not a log file\n" * (6 * MIB // 15)
    r = roomy.post("/api/parsers/log/table", files={"file": ("big.lammps", payload)})
    # Past the middleware (which would say TOO_LARGE) and refused by the parser instead.
    assert r.status_code != 413
    assert r.json().get("detail") != TOO_LARGE


# --- E4: Prony mode count and points per regime ---------------------------------------


def test_e4_n_modes_above_the_cap_is_422_on_the_json_route(client):
    body = {"regimes": rheology_regimes(), "t_min": 0.1, "n_modes": 201}
    r = client.post(MODULI_FROM_REGIMES_URL, json=body)
    assert r.status_code == 422
    assert "n_modes" in json.dumps(r.json())


def test_e4_n_modes_above_the_cap_is_422_on_the_form_route(client, data_dir):
    r = client.post(
        "/api/rheology/dynamic-moduli",
        files=[("files", ("acf.txt", (data_dir / "sample_stress_acf.txt").read_bytes()))],
        data={"volume": "1000", "temperature": "1", "n_modes": "100000"},
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == "n_modes"


def test_e4_too_many_points_in_one_regime_is_422(client):
    n = 100_001
    body = dynamics_body(regimes=[{"label": "huge", "lag_time": list(range(n)), "value": [0] * n}])
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 422


# --- D8/E7: the sweep must not delete a running job -----------------------------------


def test_d8_sweep_keeps_a_running_job_older_than_the_ttl():
    store = JobStore()
    job = store.create()
    store.update_progress(job.id, 0.5)
    job.created_at -= 10_000
    store.sweep(ttl_seconds=1)
    assert store.get(job.id) is job


def test_d8_sweep_drops_a_job_only_once_it_finished_longer_ago_than_the_ttl():
    store = JobStore()
    done, failed, fresh = store.create(), store.create(), store.create()
    store.set_result(done.id, 1)
    store.set_error(failed.id, "boom")
    store.set_result(fresh.id, 2)
    for job in (done, failed, fresh):
        job.created_at -= 10_000  # all created long ago
    done.finished_at -= 100
    failed.finished_at -= 100
    store.sweep(ttl_seconds=10)
    assert store.get(done.id) is None
    assert store.get(failed.id) is None
    assert store.get(fresh.id) is fresh


# --- E8: HEAD and 405 Allow -----------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/simulations", "/rheometer", "/dynamics/rouse-mode"])
def test_e8_pages_answer_head(client, path):
    r = client.head(path)
    assert r.status_code == 200
    assert r.content == b""


def test_e8_unknown_page_is_still_404_for_head(client):
    assert client.head("/no/such").status_code == 404


def test_e8_405_keeps_its_allow_header(client):
    r = client.get(DYNAMICS_MERGE_URL)
    assert r.status_code == 405
    assert r.headers["allow"] == "POST"


def test_e8_405_keeps_its_allow_header_on_the_html_error_page(client):
    r = client.put("/", headers={"accept": "text/html"})
    assert r.status_code == 405
    assert set(r.headers["allow"].replace(" ", "").split(",")) == {"GET", "HEAD"}


# --- D10: merge validation ------------------------------------------------------------


def test_d10_lag_value_length_mismatch_is_422_not_500(client):
    body = dynamics_body(
        regimes=[{"label": "bad", "lag_time": [0.0, 1.0, 2.0], "value": [1.0, 0.5]}]
    )
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 422
    assert "one value per lag time" in json.dumps(r.json())


def test_d10_rheology_length_mismatch_is_422_not_500(client):
    regime = {"label": "bad", "lag_time": [1.0, 2.0], "G_GK": [1.0, 0.5], "G_FSR": [1.0]}
    r = client.post(RHEOLOGY_MERGE_URL, json={"regimes": [regime]})
    assert r.status_code == 422
    assert "G_FSR" in json.dumps(r.json())


def test_d10_value_column_lag_time_is_refused(client):
    r = client.post(DYNAMICS_MERGE_URL, json=dynamics_body(value_column="lag_time"))
    assert r.status_code == 422
    assert "lag_time" in json.dumps(r.json())


def test_d10_duplicate_lag_times_are_refused(client):
    body = dynamics_body(
        regimes=[{"label": "dup", "lag_time": [0.0, 1.0, 1.0], "value": [1.0, 0.6, 0.4]}]
    )
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 422
    assert "repeats a lag time" in json.dumps(r.json())


def test_d10_merge_service_refuses_mismatch_and_duplicates_directly():
    import numpy as np

    with pytest.raises(ValueError, match="one value per lag time"):
        merge.merge_regimes([np.array([0.0, 1.0])], [{"v": np.array([1.0])}])
    with pytest.raises(ValueError, match="repeats a lag time"):
        merge.merge_regimes([np.array([0.0, 0.0])], [{"v": np.array([1.0, 2.0])}])
