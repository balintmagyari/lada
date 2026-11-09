"""Regression tests for the master-curve audit (findings C1–C8 and E5).

The same synthetic material as test_rheometry_tts.py: a generalised Maxwell model whose sweep
at T is its sweep at the reference with ω multiplied by a WLF shift factor.
"""

import json

import numpy as np
import pytest

from lada_web.rheometry.tts import TTSRequest, best_shift, shift_factors

C1, C2, T_REF = 8.86, 101.6, 150.0
TAUS = np.logspace(-3, 3, 13)
MODES = 1e5 * TAUS**-0.25


def true_log_at(t, reference=T_REF):
    def wlf(x):
        return -C1 * (x - T_REF) / (C2 + x - T_REF)

    return wlf(t) - wlf(reference)


def maxwell(omega, taus=TAUS, modes=MODES):
    wt = np.outer(omega, taus)
    return (modes * wt**2 / (1 + wt**2)).sum(1), (modes * wt / (1 + wt**2)).sum(1)


def sweeps(temperatures, omega=None, noise=0.01, seed=1, taus=TAUS, modes=MODES):
    omega = np.logspace(-1, 2, 16) if omega is None else omega
    rng = np.random.default_rng(seed)
    curves = []
    for n, t in enumerate(temperatures):
        g1, g2 = maxwell(omega * 10 ** true_log_at(t), taus, modes)
        g1 = g1 * (1 + noise * rng.standard_normal(omega.size))
        g2 = g2 * (1 + noise * rng.standard_normal(omega.size))
        curves.append(
            {
                "id": f"c{n}",
                "label": f"{t:g} °C",
                "temperature": float(t),
                "omega": omega.tolist(),
                "storage": g1.tolist(),
                "loss": g2.tolist(),
            }
        )
    return curves


def by_id(result):
    return {s.id: s for s in result.shifts}


# --- C1: a best shift pinned where the overlap runs out is not a shift ------------------------


@pytest.mark.parametrize("t2", [200.0, 205.0])
def test_c1_a_shift_pinned_to_the_overlap_boundary_is_not_found(t2):
    # Overlap of 0.08 and -0.11 decades: the search used to stop at the edge of what its
    # minimum overlap allowed (log aT = -2.4, against a true -2.9 or -3.1) and report it.
    result = shift_factors(TTSRequest(curves=sweeps((150.0, t2)), reference=150))
    shift = by_id(result)["c1"]
    assert shift.auto_log_at is None and not shift.fitted
    assert result.notes


@pytest.mark.parametrize("t2", [190.0, 193.0, 196.0])
def test_c1_a_small_but_real_overlap_is_found_by_relaxing_the_minimum(t2):
    # Pinned at the stricter minimum overlap, found with the looser one instead of -2.4.
    result = shift_factors(TTSRequest(curves=sweeps((150.0, t2)), reference=150))
    shift = by_id(result)["c1"]
    assert shift.auto_log_at == pytest.approx(true_log_at(t2), abs=0.03)


def test_c1_best_shift_says_why_when_pinned():
    curves = sweeps((150.0, 205.0), noise=0.0)
    from lada_web.rheometry.tts import _log_series

    fixed = _log_series(curves[0]["omega"], curves[0]["storage"])
    moving = _log_series(curves[1]["omega"], curves[1]["storage"])
    shift, _, problem = best_shift([(moving, fixed)])
    assert shift is None and problem == "edge"


# --- C2: one failed sweep must not move the next, nor the fits --------------------------------


def garbled():
    curves = sweeps((150.0, 160.0, 170.0, 180.0, 190.0))
    rng = np.random.default_rng(5)
    for k in ("storage", "loss"):
        curves[1][k] = (np.array(curves[1][k]) * np.exp(rng.normal(0, 1.5, 16))).tolist()
    return curves


def test_c2_a_failed_sweep_does_not_cascade_onto_the_next():
    result = shift_factors(TTSRequest(curves=garbled(), reference=150))
    shifts = by_id(result)
    assert shifts["c1"].auto_log_at is None and not shifts["c1"].fitted
    for i, t in enumerate((150.0, 160.0, 170.0, 180.0, 190.0)):
        if i == 1:
            continue
        error = abs(10 ** (shifts[f"c{i}"].log_at - true_log_at(t)) - 1)
        assert error < 0.02, (t, shifts[f"c{i}"].log_at)
    assert any("160 °C" in note and "left out of the fits" in note for note in result.notes)


def test_c2_the_fits_leave_out_shifts_that_were_not_found():
    result = shift_factors(TTSRequest(curves=garbled(), reference=150))
    assert result.wlf is not None
    assert result.wlf.c1 == pytest.approx(C1, rel=0.05)
    assert result.wlf.c2 == pytest.approx(C2, rel=0.05)


def test_c2_a_shift_set_by_hand_on_a_failed_sweep_is_fitted_and_chained_from():
    true = true_log_at(160.0)
    result = shift_factors(TTSRequest(curves=garbled(), reference=150, manual={"c1": true}))
    shifts = by_id(result)
    assert shifts["c1"].manual and shifts["c1"].fitted and shifts["c1"].log_at == true
    assert shifts["c2"].log_at == pytest.approx(true_log_at(170.0), abs=0.01)


# --- C3: a flat plateau leaves the shift undetermined -----------------------------------------

PLATEAU_TAUS = np.array([1e-6, 1e-5, 1e-4, 3e3, 1e4])
PLATEAU_MODES = np.array([1e6, 3e5, 1e5, 5e4, 5e4])


def test_c3_a_flat_storage_plateau_is_reported_as_undetermined():
    curves = sweeps(
        (150.0, 160.0, 170.0, 180.0, 190.0), taus=PLATEAU_TAUS, modes=PLATEAU_MODES, seed=1
    )
    result = shift_factors(TTSRequest(curves=curves, reference=150, use="storage"))
    others = [s for s in result.shifts if s.temperature != 150.0]
    assert all(s.auto_log_at is None and not s.fitted for s in others)
    assert any("too flat" in note for note in result.notes)
    assert result.wlf is None


def test_c3_the_same_sweeps_superpose_on_the_loss_modulus():
    curves = sweeps(
        (150.0, 160.0, 170.0, 180.0, 190.0), taus=PLATEAU_TAUS, modes=PLATEAU_MODES, seed=1
    )
    result = shift_factors(TTSRequest(curves=curves, reference=150, use="both"))
    for s in result.shifts:
        assert s.log_at == pytest.approx(true_log_at(s.temperature), abs=0.02)
    assert not result.notes


# --- C4: many close sweeps do not drift ---------------------------------------------------------


def test_c4_sixty_close_sweeps_do_not_accumulate_bias():
    temps = np.linspace(150, 190, 60)
    curves = sweeps(temps, omega=np.logspace(-1, 2, 50), noise=0.02, seed=0)
    result = shift_factors(TTSRequest(curves=curves, reference=150))
    worst = max(abs(s.log_at - true_log_at(s.temperature)) for s in result.shifts)
    assert worst < 0.05  # chaining neighbour to neighbour drifted to about 0.24


# --- C5: WLF needs three distinct temperatures with real shifts besides the reference ---------


def test_c5_two_temperatures_besides_the_reference_do_not_make_a_wlf_fit():
    result = shift_factors(TTSRequest(curves=sweeps((150.0, 160.0, 170.0)), reference=150))
    assert result.wlf is None and "three or more" in result.wlf_note
    assert result.arrhenius is not None


def test_c5_duplicate_temperatures_count_once():
    result = shift_factors(TTSRequest(curves=sweeps((150.0, 170.0, 170.0)), reference=150))
    assert result.wlf is None and "three or more" in result.wlf_note
    result = shift_factors(TTSRequest(curves=sweeps((150.0, 160.0, 170.0, 170.0)), reference=150))
    assert result.wlf is None


def test_c5_three_temperatures_besides_the_reference_do():
    curves = sweeps((150.0, 160.0, 170.0, 180.0))
    result = shift_factors(TTSRequest(curves=curves, reference=150))
    assert result.wlf is not None


# --- C6: a second sweep at the reference temperature is a sweep like any other --------------


def test_c6_only_the_reference_sweep_itself_is_fixed():
    curves = sweeps((150.0, 150.0, 160.0, 170.0))
    result = shift_factors(TTSRequest(curves=curves, reference=150, manual={"c1": 0.3}))
    assert result.reference_id == "c0"
    shifts = by_id(result)
    assert shifts["c0"].log_at == 0.0 and not shifts["c0"].manual
    assert shifts["c1"].log_at == 0.3 and shifts["c1"].manual


# --- C8 and E5: what the route refuses ----------------------------------------------------------


def post(client, body):
    return client.post(
        "/api/rheometry/tts",
        content=json.dumps(body),  # json.dumps writes NaN, as a browser bug might
        headers={"Content-Type": "application/json"},
    )


def test_c8_a_curve_with_too_many_points_is_refused(client):
    curves = sweeps((150.0, 160.0))
    curves[1]["omega"] = np.logspace(-1, 2, 5001).tolist()
    assert post(client, {"curves": curves}).status_code == 422
    curves = sweeps((150.0, 160.0))
    curves[1]["loss"] = [1.0] * 5001
    assert post(client, {"curves": curves}).status_code == 422


@pytest.mark.parametrize("temperature", [-273.15, -300.0, float("nan"), float("inf")])
def test_e5_temperatures_at_or_below_absolute_zero_or_not_finite_are_refused(client, temperature):
    curves = sweeps((150.0, 160.0))
    curves[0]["temperature"] = temperature
    assert post(client, {"curves": curves, "reference": temperature}).status_code == 422


def test_e5_a_reference_or_manual_shift_that_is_not_finite_is_refused(client):
    curves = sweeps((150.0, 160.0))
    assert post(client, {"curves": curves, "reference": float("nan")}).status_code == 422
    assert post(client, {"curves": curves, "manual": {"c1": float("nan")}}).status_code == 422
    assert post(client, {"curves": curves, "manual": {"c1": float("inf")}}).status_code == 422


def test_e5_a_reference_at_absolute_zero_takes_the_nearest_sweep(client):
    r = post(client, {"curves": sweeps((150.0, 160.0)), "reference": -273.15})
    assert r.status_code == 200
    assert r.json()["reference"] == 150.0
