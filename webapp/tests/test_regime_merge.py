"""Route-level tests for merging several already-computed regimes into one curve."""

import io

import numpy as np
import pandas as pd
from lada.analysis import calc_dynamic_moduli_prony

DYNAMICS_MERGE_URL = "/api/dynamics/merge-regimes"
RHEOLOGY_MERGE_URL = "/api/rheology/merge-regimes"
MODULI_FROM_REGIMES_URL = "/api/rheology/dynamic-moduli/from-regimes"


def dynamics_regime(label: str, lag_time: list[float], value: list[float]) -> dict:
    return {"label": label, "lag_time": lag_time, "value": value}


def rheology_regime(
    label: str, lag_time: list[float], g_gk: list[float], g_fsr: list[float]
) -> dict:
    return {"label": label, "lag_time": lag_time, "G_GK": g_gk, "G_FSR": g_fsr}


# --- Dynamics: /api/dynamics/merge-regimes ----------------------------------


def test_merge_regimes_combines_non_overlapping_dynamics_regimes(client):
    body = {
        "regimes": [
            dynamics_regime("fast", [0.0, 1.0], [1.0, 0.5]),
            dynamics_regime("slow", [10.0, 20.0], [0.2, 0.1]),
        ],
        "value_column": "acf",
        "title": "End-to-end vector autocorrelation",
        "y_label": "C(t)",
        "download_stem": "merged_segment_acf",
    }
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["columns"] == ["lag_time", "acf"]
    assert [row[0] for row in result["rows"]] == [0.0, 1.0, 10.0, 20.0]
    assert [row[1] for row in result["rows"]] == [1.0, 0.5, 0.2, 0.1]
    assert result["meta"]["regimes"] == 2
    assert result["download_stem"] == "merged_segment_acf"


def test_merge_regimes_single_regime_passes_through(client):
    body = {
        "regimes": [dynamics_regime("only", [0.0, 1.0], [1.0, 0.4])],
        "value_column": "isf",
        "title": "ISF",
        "y_label": "F(q,t)/F(q,0)",
        "download_stem": "merged_isf",
    }
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == [[0.0, 1.0], [1.0, 0.4]]


def test_merge_regimes_stitches_overlapping_dynamics_regimes_at_the_finer_end(client):
    # Regime "a" samples more finely and stops at 2.5; "b" takes over past that. Its 2.0 point
    # lies inside "a"'s range, so the finer sampling there wins and it is dropped.
    body = {
        "regimes": [
            dynamics_regime("a", [0.0, 1.0, 2.5], [1.0, 0.6, 0.3]),
            dynamics_regime("b", [2.0, 3.0], [0.35, 0.2]),
        ],
        "value_column": "acf",
        "title": "x",
        "y_label": "y",
        "download_stem": "x",
    }
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["plot"]["series"]["lag_time"] == [0.0, 1.0, 2.5, 3.0]
    assert r.json()["plot"]["series"]["acf"] == [1.0, 0.6, 0.3, 0.2]


def test_merge_regimes_stitches_real_nested_dynamics_regimes(client):
    # The geometry of real fast/mid/slow runs: each lag axis is frame * dt, so they all start
    # at zero and nest. This is exactly what the previous overlap rule refused outright.
    fast = [round(i * 0.4, 6) for i in range(6)]  # 0 -> 2.0
    mid = [i * 10.0 for i in range(4)]  # 0 -> 30.0
    body = {
        "regimes": [
            dynamics_regime("fast", fast, [1.0, 0.9, 0.82, 0.75, 0.7, 0.66]),
            dynamics_regime("mid", mid, [1.0, 0.4, 0.2, 0.1]),
        ],
        "value_column": "acf",
        "title": "x",
        "y_label": "y",
        "download_stem": "x",
    }
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    lag = r.json()["plot"]["series"]["lag_time"]
    assert lag == [*fast, 10.0, 20.0, 30.0]
    assert lag == sorted(set(lag))


def test_merge_regimes_rejects_empty_regime_list(client):
    body = {
        "regimes": [],
        "value_column": "acf",
        "title": "x",
        "y_label": "y",
        "download_stem": "x",
    }
    r = client.post(DYNAMICS_MERGE_URL, json=body)
    assert r.status_code == 422


def test_merge_regimes_download_round_trips(client):
    body = {
        "regimes": [
            dynamics_regime("fast", [0.0, 1.0], [1.0, 0.5]),
            dynamics_regime("slow", [10.0, 20.0], [0.2, 0.1]),
        ],
        "value_column": "acf",
        "title": "x",
        "y_label": "y",
        "download_stem": "merged_segment_acf",
    }
    view = client.post(DYNAMICS_MERGE_URL, json=body).json()
    r = client.post(f"{DYNAMICS_MERGE_URL}/download?fmt=csv", json=body)
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="merged_segment_acf.csv"'
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert np.array_equal(df.to_numpy(), np.array(view["rows"], dtype=float))


# --- Rheology: /api/rheology/merge-regimes -----------------------------------


def test_rheology_merge_regimes_combines_non_overlapping_gt(client):
    body = {
        "regimes": [
            rheology_regime("fast", [0.0, 1.0], [10.0, 6.0], [7.0, 4.0]),
            rheology_regime("slow", [10.0, 20.0], [1.0, 0.5], [0.7, 0.3]),
        ],
        "download_stem": "merged_G_t",
    }
    r = client.post(RHEOLOGY_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["columns"] == ["lag_time", "G_GK", "G_FSR"]
    assert [row[0] for row in result["rows"]] == [0.0, 1.0, 10.0, 20.0]
    assert result["meta"]["regimes"] == 2


def test_rheology_merge_regimes_keeps_the_finer_regimes_value_where_they_meet(client):
    # Independent runs never agree exactly at a shared lag time; requiring that they do (the
    # old rule) rejected ordinary data. The finer regime's value wins instead.
    body = {
        "regimes": [
            rheology_regime("a", [0.0, 1.0], [10.0, 6.0], [7.0, 4.0]),
            rheology_regime("b", [1.0, 2.0], [5.0, 3.0], [3.0, 2.0]),
        ],
        "download_stem": "x",
    }
    r = client.post(RHEOLOGY_MERGE_URL, json=body)
    assert r.status_code == 200, r.text
    series = r.json()["plot"]["series"]
    assert series["lag_time"] == [0.0, 1.0, 2.0]
    assert series["G_GK"] == [10.0, 6.0, 3.0]
    assert series["G_FSR"] == [7.0, 4.0, 2.0]


# --- Rheology: /api/rheology/dynamic-moduli/from-regimes ---------------------


def test_dynamic_moduli_from_regimes_merges_gt_then_fits_once(client):
    """The endpoint must match: merge G(t) by hand, then call calc_dynamic_moduli_prony once."""
    fast = rheology_regime("fast", list(np.linspace(0.0, 5.0, 100)), None, None)
    slow = rheology_regime("slow", list(np.linspace(6.0, 20.0, 100)), None, None)

    def maxwell(t: np.ndarray) -> np.ndarray:
        return 10.0 * np.exp(-np.array(t) / 3.0)

    fast["G_GK"] = fast["G_FSR"] = list(maxwell(np.array(fast["lag_time"])) * 1.0)
    slow["G_GK"] = slow["G_FSR"] = list(maxwell(np.array(slow["lag_time"])) * 1.0)

    body = {
        "regimes": [fast, slow],
        "method": "GK",
        "t_min": 0.5,
        "n_modes": 20,
        "n_omega": 40,
        "download_stem": "merged_moduli",
    }
    r = client.post(MODULI_FROM_REGIMES_URL, json=body)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["columns"] == ["omega", "G_prime", "G_dprime"]

    merged_gt = pd.DataFrame(
        {
            "lag_time": fast["lag_time"] + slow["lag_time"],
            "G_GK": fast["G_GK"] + slow["G_GK"],
            "G_FSR": fast["G_FSR"] + slow["G_FSR"],
        }
    ).sort_values("lag_time")
    expected = calc_dynamic_moduli_prony(merged_gt, method="GK", t_min=0.5, n_modes=20, n_omega=40)
    assert np.allclose(result["plot"]["series"]["omega"], expected["omega"].to_numpy())
    assert np.allclose(result["plot"]["series"]["G_prime"], expected["G_prime"].to_numpy())


def test_dynamic_moduli_from_regimes_stitches_overlapping_regimes(client):
    body = {
        "regimes": [
            rheology_regime("a", [0.0, 1.0, 2.5], [10.0, 6.0, 3.0], [7.0, 4.0, 2.0]),
            rheology_regime("b", [2.0, 3.0], [3.5, 2.0], [2.2, 1.0]),
        ],
        "download_stem": "x",
        "t_min": 0.5,
        "n_modes": 2,
        "n_omega": 8,
    }
    r = client.post(MODULI_FROM_REGIMES_URL, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["omega", "G_prime", "G_dprime"]


def test_regime_merge_pages_still_have_no_page_route(client):
    """These are pure API endpoints; no HTML page is registered for them."""
    assert client.get("/dynamics/merge-regimes").status_code == 404
