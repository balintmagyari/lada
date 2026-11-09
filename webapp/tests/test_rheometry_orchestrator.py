"""TA Orchestrator data files (.RSDnn), checked against Orchestrator's own text exports.

Each fixture pair is one real test saved twice by Orchestrator: the binary data file and its
File > Export text table. The binary file holds only raw signals, so every column the reader
rebuilds must agree with the export Orchestrator computed from the same signals.
"""

import math
from pathlib import Path

import pytest

from lada_web.rheometry import orchestrator
from lada_web.rheometry.readers import ACCEPTED, read_file

HERE = Path(__file__).parent / "data" / "rheometry" / "orchestrator"
PAIRS = {
    "FS": ("TA_Orchestrator_FS.RSD02", "frequency_sweep", 120.0, 28),
    "SS": ("TA_Orchestrator_SS.RSD05", "amplitude_sweep", 40.0, 21),
    "TS": ("TA_Orchestrator_TS.RSD06", "time_sweep", 100.0, 181),
}


def load(name):
    return read_file(HERE / name, name)


@pytest.mark.parametrize("test", PAIRS)
def test_every_column_agrees_with_orchestrators_own_export(test):
    binary, *_ = PAIRS[test]
    rsd = load(binary).blocks[0]
    txt = load(f"TA_Orchestrator_{test}.txt").blocks[0]
    reference = {c.label: c for c in txt.columns}
    compared = 0
    for column in rsd.columns:
        if column.label not in reference:
            continue
        ref = reference[column.label]
        assert column.quantity == ref.quantity, column.label
        assert column.unit == ref.unit, column.label
        for got, want in zip(column.values, ref.values, strict=True):
            # The data file stores float32 and the export rounds to six digits.
            assert got == pytest.approx(want, rel=1e-4, abs=1e-9), column.label
        compared += 1
    # G′, G″, tan δ, torque, strain, stress (not in TS), frequency, temperature, the derived ones.
    assert compared >= 12


@pytest.mark.parametrize("test", PAIRS)
def test_the_test_type_temperature_and_size_are_found(test):
    binary, test_type, temperature, points = PAIRS[test]
    dataset = load(binary)
    assert dataset.format == "orchestrator"
    assert dataset.format_label == "TA Orchestrator data file"
    (block,) = dataset.blocks
    assert block.test_type == test_type
    assert block.temperature == temperature
    assert block.n_points == points


def test_the_metadata_of_the_test_is_read():
    meta = load("TA_Orchestrator_SS.RSD05").metadata
    assert meta["Test"] == "StrSwp_PA4%+2PB4%_L1_40degC_50rads_PP8_xducer2"
    assert meta["Operator"] == "Balint"
    assert meta["Started"] == "2026-11-02 05:28:52 UTC"
    assert meta["Orchestrator version"] == "V6.5.8"
    assert meta["Plate diameter"] == "8 mm"
    assert meta["Gap"] == "0.4126 mm"
    # This test ran on the high-range transducer; the others on the low one.
    assert meta["Transducer range"] == "2106.6 g·cm"
    assert load("TA_Orchestrator_FS.RSD02").metadata["Transducer range"] == "210.66 g·cm"


def test_the_time_sweeps_elapsed_time_is_exact():
    rsd = {c.label: c for c in load("TA_Orchestrator_TS.RSD06").blocks[0].columns}
    txt = {c.label: c for c in load("TA_Orchestrator_TS.txt").blocks[0].columns}
    assert rsd["time"].values == txt["time"].values


def test_time_is_added_where_orchestrator_shows_only_the_clock():
    columns = {c.label: c for c in load("TA_Orchestrator_FS.RSD02").blocks[0].columns}
    assert "ClockTime" not in columns
    times = columns["time"].values
    assert times == sorted(times) and len(times) == 28


def test_torque_in_gram_centimetres_is_converted_to_newton_metres():
    for name in ("TA_Orchestrator_FS.RSD02", "TA_Orchestrator_FS.txt"):
        torque = next(c for c in load(name).blocks[0].columns if c.label == "Torque")
        assert torque.quantity == "torque" and torque.unit == "N·m"
        assert torque.values[0] == pytest.approx(5.66374 * 9.80665e-5, rel=1e-5)


def test_every_import_says_how_the_moduli_were_computed():
    assert any("parallel-plate" in w for w in load("TA_Orchestrator_TS.RSD06").warnings)


def test_the_picker_and_the_server_accept_every_test_number():
    assert ".rsd02" in ACCEPTED and ".rsd99" in ACCEPTED and ".rsd00" in ACCEPTED


def test_a_truncated_file_is_refused_with_directions(tmp_path):
    raw = (HERE / "TA_Orchestrator_FS.RSD02").read_bytes()
    path = tmp_path / "cut.RSD02"
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError, match="Orchestrator file could not be read.*Export"):
        read_file(path, path.name)


def test_a_file_named_rsd_that_is_not_one_is_refused(tmp_path):
    path = tmp_path / "notes.RSD03"
    path.write_bytes(b"this is plain text\n1 2\n3 4\n")
    with pytest.raises(ValueError, match="named like a TA Orchestrator data file"):
        read_file(path, path.name)


def test_the_notes_lose_their_rtf_formatting():
    rtf = (
        r"{\rtf1\ansi{\fonttbl{\f0\fswiss System;}}\viewkind4\pard\b\f0\fs20 "
        r"Sample A\par second line\par }"
    )
    assert orchestrator._plain_notes(rtf) == "Sample A\nsecond line"


def test_the_route_reads_all_three_files_in_one_upload(client):
    files = [
        ("files", (name, (HERE / name).read_bytes(), "application/octet-stream"))
        for name, *_ in PAIRS.values()
    ]
    r = client.post("/api/rheometry/parse", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] == []
    assert [d["blocks"][0]["test_type"] for d in body["datasets"]] == [
        "frequency_sweep",
        "amplitude_sweep",
        "time_sweep",
    ]
    g1 = next(c for c in body["datasets"][0]["blocks"][0]["columns"] if c["label"] == "G'")
    assert all(isinstance(v, float) and math.isfinite(v) for v in g1["values"])


def test_my_data_offers_orchestrator_files_and_the_time_sweep_page_exists(client):
    page = client.get("/rheometer/my-data").text
    assert ".rsd02" in page and "TA Orchestrator" in page
    r = client.get("/rheometer/time-sweep")
    assert r.status_code == 200 and 'data-preset="time_sweep"' in r.text


@pytest.mark.parametrize(("lo", "hi"), [(0.2974, 0.2982), (0.29, 0.31), (2.0, 3.0), (10.0, 50.0)])
def test_a_log_axis_that_spans_a_sliver_of_a_decade_still_gets_labels(lo, hi):
    """A time sweep's tan δ barely moves: a log axis that narrow must still be labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lada_web.rheometry.figure import AxisSpec, _style_axis

    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [lo, hi])
        _style_axis(ax, AxisSpec(title="tan δ", log=True), "y", grid=False)
        fig.canvas.draw()
        shown = [t for t in ax.get_yticks() if lo * 0.999 <= t <= hi * 1.001]
        labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert len(shown) >= 2 and len(labels) >= 2, (shown, labels)
    finally:
        plt.close(fig)
