"""Time-temperature superposition, checked against a known answer.

A generalised Maxwell material is "measured" at five temperatures: its sweep at T is its
sweep at the reference temperature with ω multiplied by a WLF shift factor, plus 1 % noise.
Superposition has to find those shift factors and the WLF constants again.
"""

import math

import numpy as np
import pytest

from lada_web.rheometry.tts import TTSRequest, shift_factors

C1, C2, T_REF = 8.86, 101.6, 150.0
TEMPERATURES = (150.0, 160.0, 170.0, 180.0, 190.0)
TAUS = np.logspace(-3, 3, 13)
MODES = 1e5 * TAUS**-0.25
OMEGA = np.logspace(-1, 2, 16)


def true_log_at(temperature, reference=T_REF):
    def wlf(t):
        return -C1 * (t - T_REF) / (C2 + t - T_REF)

    return wlf(temperature) - wlf(reference)


def maxwell(omega):
    wt = np.outer(omega, TAUS)
    return (MODES * wt**2 / (1 + wt**2)).sum(1), (MODES * wt / (1 + wt**2)).sum(1)


def sweeps(seed=11, noise=0.01, temperatures=TEMPERATURES, vertical=False):
    rng = np.random.default_rng(seed)
    curves = []
    for t in temperatures:
        g1, g2 = maxwell(OMEGA * 10 ** true_log_at(t))
        if vertical:  # what a real material does: moduli scale with absolute temperature
            factor = (t + 273.15) / (T_REF + 273.15)
            g1, g2 = g1 * factor, g2 * factor
        g1 = g1 * (1 + noise * rng.standard_normal(OMEGA.size))
        g2 = g2 * (1 + noise * rng.standard_normal(OMEGA.size))
        curves.append(
            {
                "id": f"T{t:g}",
                "label": f"{t:g} °C",
                "temperature": t,
                "omega": OMEGA.tolist(),
                "storage": g1.tolist(),
                "loss": g2.tolist(),
            }
        )
    return curves


def by_temperature(result):
    return {s.temperature: s for s in result.shifts}


@pytest.mark.parametrize("seed", [11, 23])
def test_shift_factors_are_recovered_within_two_percent(seed):
    result = shift_factors(TTSRequest(curves=sweeps(seed), reference=T_REF))
    assert result.reference == T_REF
    for t, shift in by_temperature(result).items():
        error = abs(10 ** (shift.log_at - true_log_at(t)) - 1)
        assert error < 0.02, (t, shift.log_at, true_log_at(t))


@pytest.mark.parametrize("seed", [11, 23])
def test_the_wlf_constants_are_recovered_within_five_percent(seed):
    result = shift_factors(TTSRequest(curves=sweeps(seed), reference=T_REF))
    assert result.wlf is not None
    assert result.wlf.c1 == pytest.approx(C1, rel=0.05)
    assert result.wlf.c2 == pytest.approx(C2, rel=0.05)


def test_noise_free_sweeps_shift_almost_exactly():
    result = shift_factors(TTSRequest(curves=sweeps(noise=0.0), reference=T_REF))
    for t, shift in by_temperature(result).items():
        assert shift.log_at == pytest.approx(true_log_at(t), abs=2e-3)


def test_any_measured_temperature_can_be_the_reference():
    result = shift_factors(TTSRequest(curves=sweeps(noise=0.0), reference=170))
    assert result.reference == 170
    for t, shift in by_temperature(result).items():
        assert shift.log_at == pytest.approx(true_log_at(t, reference=170), abs=3e-3)
    assert by_temperature(result)[170].log_at == 0.0


def test_a_reference_between_sweeps_takes_the_nearest_and_says_so():
    result = shift_factors(TTSRequest(curves=sweeps(), reference=163))
    assert result.reference == 160
    assert any("nearest one, 160 °C" in note for note in result.notes)


def test_without_a_reference_the_middle_sweep_is_used():
    assert shift_factors(TTSRequest(curves=sweeps())).reference == 170


def test_the_order_the_sweeps_arrive_in_does_not_matter():
    curves = sweeps()
    forward = shift_factors(TTSRequest(curves=curves, reference=T_REF))
    backward = shift_factors(TTSRequest(curves=curves[::-1], reference=T_REF))
    assert [s.log_at for s in forward.shifts] == pytest.approx([s.log_at for s in backward.shifts])


def test_one_modulus_alone_also_superposes():
    for use in ("storage", "loss"):
        result = shift_factors(TTSRequest(curves=sweeps(noise=0.0), reference=T_REF, use=use))
        for t, shift in by_temperature(result).items():
            assert shift.log_at == pytest.approx(true_log_at(t), abs=5e-3), use


def test_a_shift_set_by_hand_is_applied_and_the_fit_follows_it():
    curves = sweeps()
    auto = shift_factors(TTSRequest(curves=curves, reference=T_REF))
    manual = shift_factors(TTSRequest(curves=curves, reference=T_REF, manual={"T190": -3.0}))
    hand = by_temperature(manual)[190]
    assert hand.log_at == -3.0 and hand.manual
    assert hand.auto_log_at == pytest.approx(by_temperature(auto)[190].log_at)
    assert manual.wlf.c1 != pytest.approx(auto.wlf.c1, rel=1e-3)


def test_the_reference_can_never_be_shifted_by_hand():
    result = shift_factors(TTSRequest(curves=sweeps(), reference=T_REF, manual={"T150": 1.0}))
    assert by_temperature(result)[150].log_at == 0.0
    assert not by_temperature(result)[150].manual
    assert any("never shifted" in note for note in result.notes)


def test_the_vertical_correction_restores_the_shift_of_temperature_scaled_moduli():
    curves = sweeps(noise=0.0, vertical=True)
    result = shift_factors(TTSRequest(curves=curves, reference=T_REF, vertical=True))
    for t, shift in by_temperature(result).items():
        assert shift.log_bt == pytest.approx(math.log10((t + 273.15) / (T_REF + 273.15)))
        assert shift.log_at == pytest.approx(true_log_at(t), abs=3e-3)


def test_a_sweep_that_cannot_overlap_is_placed_beside_its_neighbour_and_flagged():
    curves = sweeps(noise=0.0, temperatures=(150.0, 160.0))
    far = dict(curves[1], id="far", label="far away", temperature=300.0)
    far["omega"] = (OMEGA * 1e12).tolist()
    far["storage"] = [1e-9] * OMEGA.size  # nothing like the others
    far["loss"] = [1e-9] * OMEGA.size
    result = shift_factors(TTSRequest(curves=[*curves, far], reference=T_REF))
    shift = by_temperature(result)[300]
    assert shift.auto_log_at is None
    assert any("far away does not overlap" in note for note in result.notes)


def test_too_few_temperatures_for_a_fit_say_so():
    two = shift_factors(TTSRequest(curves=sweeps(temperatures=(150.0, 160.0)), reference=150))
    assert two.wlf is None and "three or more" in two.wlf_note
    assert two.arrhenius is not None
    one = shift_factors(TTSRequest(curves=sweeps(temperatures=(150.0,))))
    assert one.arrhenius is None and "two or more" in one.arrhenius_note


def test_the_arrhenius_energy_matches_its_definition():
    result = shift_factors(TTSRequest(curves=sweeps(noise=0.0), reference=T_REF))
    shifts = result.shifts
    x = np.array([1 / (s.temperature + 273.15) - 1 / (T_REF + 273.15) for s in shifts])
    y = np.array([s.log_at * math.log(10) for s in shifts])
    expected = np.dot(x, y) / np.dot(x, x) * 8.314462618 / 1000
    assert result.arrhenius.ea_kj_per_mol == pytest.approx(expected)


# --- Route and page ---------------------------------------------------------------------------


def test_the_route_returns_shifts_and_fits(client):
    r = client.post("/api/rheometry/tts", json={"curves": sweeps(), "reference": 150})
    assert r.status_code == 200
    body = r.json()
    assert [s["temperature"] for s in body["shifts"]] == list(TEMPERATURES)
    assert body["wlf"]["c1"] == pytest.approx(C1, rel=0.05)


def test_the_route_refuses_an_empty_request(client):
    assert client.post("/api/rheometry/tts", json={"curves": []}).status_code == 422


def test_the_master_curve_page_carries_the_shift_factor_panel(client):
    r = client.get("/rheometer/master-curve")
    assert r.status_code == 200
    assert 'data-preset="master_curve"' in r.text
    for part in (
        'id="tts-panel"',
        'id="tts-reference"',
        'id="tts-rows"',
        'id="tts-plot"',
        'data-tts-export="csv"',
    ):
        assert part in r.text
    # The panel belongs to the master curve only.
    assert 'id="tts-panel"' not in client.get("/rheometer/frequency-sweep").text
