"""Regression tests for the rheometer-import audit (findings A1-A24, E6, E9).

Each test is named after the finding it pins down. The inputs are the auditor's: small files
that a real instrument or spreadsheet could write, which the importer used to misread.
"""

import datetime
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from lada_web.rheometry import quantities as q
from lada_web.rheometry import tabular
from lada_web.rheometry.derive import derive
from lada_web.rheometry.detect import detect_test_type, metadata_temperature
from lada_web.rheometry.model import Block, Column
from lada_web.rheometry.readers import Limits, read_file

MYDATA_JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lada_web"
    / "static"
    / "js"
    / "rheometry"
    / "mydata.js"
)

STORE_JS = MYDATA_JS.with_name("store.js")
STUDIO_JS = MYDATA_JS.with_name("studio.js")


def read_text(tmp_path, name, text, limits=None):
    path = tmp_path / name
    path.write_bytes(text.encode() if isinstance(text, str) else text)
    return read_file(path, name, limits)


def col(block, quantity):
    return next(c for c in block.columns if c.quantity == quantity and not c.derived)


def by_label(block, label):
    return next(c for c in block.columns if c.label == label)


# --- Units ------------------------------------------------------------------------------------


def test_a1_unit_case_tells_milli_from_mega(tmp_path):
    assert q.parse_unit("mPa", "stress").factor == 1e-3
    assert q.parse_unit("MPa", "stress").factor == 1e6
    assert q.parse_unit("mPa s", "viscosity").symbol == "mPa·s"
    assert q.parse_unit("MPa·s", "viscosity").factor == 1e6
    # Case-insensitive only where nothing else could be meant.
    assert q.parse_unit("PA", "stress").symbol == "Pa"
    assert q.parse_unit("pa.s", "viscosity").symbol == "Pa·s"
    assert q.parse_unit("CP", "viscosity").symbol == "mPa·s"
    assert q.parse_unit("MHz", "frequency") is None
    assert q.parse_unit("1/mPa", "compliance") is None
    ds = read_text(tmp_path, "mpa.csv", "Shear stress (mPa),Strain (%)\n1,1\n2,2\n3,3\n")
    assert col(ds.blocks[0], "stress").values == pytest.approx([1e-3, 2e-3, 3e-3])


def test_a2_an_unknown_unit_in_parentheses_is_kept_and_warned_about(tmp_path):
    ds = read_text(tmp_path, "p.csv", "Time (s),Torque (nN.m),Stress (psi)\n1,5,1\n2,6,2\n")
    (block,) = ds.blocks
    for label, unit in (("Torque", "nN.m"), ("Stress", "psi")):
        c = by_label(block, label)
        assert c.quantity is None and c.source_unit == unit
        assert any(f"“{label}” is given in “{unit}”" in w for w in ds.warnings)
    assert by_label(block, "Stress").values == [1.0, 2.0]
    # A bracket that is part of the name is still not a unit.
    assert q.split_header("Tan(delta)") == ("Tan(delta)", None)
    assert q.split_header("Compliance J(t) (1/Pa)") == ("Compliance J(t)", "1/Pa")
    assert q.split_header("Strain (mm/mm)") == ("Strain", "mm/mm")


def test_a3_tab_files_with_decimal_commas_split_on_tabs(tmp_path):
    text = (
        "Angular frequency [rad/s]\tStorage modulus [Pa]\tLoss modulus [Pa]\n"
        "0,1\t10,5\t20,25\n1,0\t100,5\t150,75\n10,0\t1000,5\t900,25\n"
    )
    assert tabular.sniff_delimiter(text) == "\t"
    (block,) = read_text(tmp_path, "de.txt", text).blocks
    assert col(block, "angular_frequency").values == [0.1, 1.0, 10.0]
    assert col(block, "storage_modulus").values == [10.5, 100.5, 1000.5]
    assert block.test_type == "frequency_sweep"


def _rederive(client, block, key, quantity, unit=None):
    r = client.post(
        "/api/rheometry/rederive",
        json={"block": block, "overrides": [{"key": key, "quantity": quantity, "unit": unit}]},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _parsed(client, name, text):
    r = client.post("/api/rheometry/parse", files=[("files", (name, text.encode()))])
    return r.json()["datasets"][0]["blocks"][0]


def _values(block, key):
    return next(c for c in block["columns"] if c["key"] == key)


def test_a4_a_unit_less_strain_survives_rederive_and_can_be_made_a_fraction(client):
    block = _parsed(client, "s.csv", "Strain,Storage modulus\n1,100\n10,90\n100,50\n")
    assert _values(block, "strain")["values"] == [1.0, 10.0, 100.0]
    same = _rederive(client, block, "strain", "strain")
    assert _values(same, "strain")["values"] == [1.0, 10.0, 100.0]
    # Through "not a quantity" and back: still the numbers the file wrote.
    plain = _rederive(client, block, "strain", None)
    back = _rederive(client, plain, "strain", "strain")
    assert _values(back, "strain")["values"] == [1.0, 10.0, 100.0]
    # "No unit" (a fraction) can be applied, and stays applied on the next correction.
    fraction = _rederive(client, block, "strain", "strain", "")
    assert _values(fraction, "strain")["values"] == [100.0, 1000.0, 10000.0]
    assert _values(fraction, "strain")["source_unit"] == "1"
    again = _rederive(client, fraction, "strain", "strain")
    assert _values(again, "strain")["values"] == [100.0, 1000.0, 10000.0]


def test_a5_semicolon_csv_with_decimal_commas_is_read(tmp_path):
    text = (
        "Angular frequency [rad/s];Storage modulus [Pa];Loss modulus [Pa]\n"
        "0,1;10,5;20,25\n1,0;100,5;150,75\n10,0;1000,5;900,25\n"
    )
    assert tabular.sniff_delimiter(text) == ";"
    (block,) = read_text(tmp_path, "semi.csv", text).blocks
    assert col(block, "loss_modulus").values == [20.25, 150.75, 900.25]


def run_mydata(expression: str):
    """Evaluate `expression` against mydata.js's test hooks in Node, with the real registry."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    harness = """
const fs = require('fs');
const noop = () => {};
global.document = { readyState: 'loading', addEventListener: noop, getElementById: () => null };
global.window = {};
eval(fs.readFileSync(%s, 'utf8'));
eval(fs.readFileSync(%s, 'utf8'));
const hooks = global.window.ladaMyDataHooks;
const reg = %s;
hooks.setRegistry({ quantities: reg });
const units = (k) => reg.find((x) => x.key === k).units;
console.log(JSON.stringify(%s));
"""
    registry = json.dumps(q.registry_json())
    script = harness % (json.dumps(str(STORE_JS)), json.dumps(str(MYDATA_JS)), registry, expression)
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    ("quantity", "written", "expected"),
    [
        ("viscosity", "mPa.s", "mPa·s"),
        ("viscosity", "cP", "mPa·s"),
        ("viscosity", "MPa·s", "MPa·s"),
        ("torque", "µN.m", "µN·m"),
        ("strain", "", "%"),  # no unit given: the % the server assumed, not "no unit"
        ("strain", "1", ""),
        ("temperature", "degC", "°C"),
        ("shear_rate", "s-1", "1/s"),
        ("storage_modulus", "mPa", "mPa"),
        ("storage_modulus", "MPa", "MPa"),
        ("storage_modulus", "psi", None),
    ],
)
def test_a6_the_column_editor_preselects_the_files_unit(quantity, written, expected):
    expr = "(function(){ var u = hooks.matchUnit(units(%s), %s); return u ? u.symbol : null; })()"
    got = run_mydata(expr % (json.dumps(quantity), json.dumps(written)))
    assert got == expected
    # The server reads the same spelling as the same unit.
    if expected is not None:
        server = (
            q.parse_unit(written, quantity) if written else q.QUANTITY_BY_KEY[quantity].canonical
        )
        assert server.symbol == expected


def test_a7_whitespace_tables_keep_header_names_with_spaces(tmp_path):
    text = "Shear rate (1/s)  Viscosity (Pa.s)\n1   10\n10  5\n100 2\n"
    (block,) = read_text(tmp_path, "ws.txt", text).blocks
    assert col(block, "shear_rate").values == [1.0, 10.0, 100.0]
    assert col(block, "viscosity").values == [10.0, 5.0, 2.0]
    assert block.test_type == "flow_curve"


@pytest.mark.parametrize(
    ("cell", "comma", "value"),
    [
        ("12,500", False, 12500.0),
        ("1,234.5", False, 1234.5),
        ("1,234,567", False, 1234567.0),
        ("1.234,5", False, 1234.5),
        ("1.234,5", True, 1234.5),
        ("1,234.5", True, 1234.5),
        ("0,125", False, 0.125),
        ("1,5", False, 1.5),
    ],
)
def test_a8_thousands_separators_are_read(cell, comma, value):
    assert tabular.parse_number(cell, comma) == value


def test_a8_a_file_with_thousands_separators_keeps_every_row(tmp_path):
    text = "Shear rate (1/s)\tViscosity (Pa.s)\n1\t1,234.5\n10\t1,100.2\n100\t1,000\n1000\t999.5\n"
    (block,) = read_text(tmp_path, "t.txt", text).blocks
    assert col(block, "viscosity").values == [1234.5, 1100.2, 1000.0, 999.5]
    quoted = 'Shear rate (1/s),Viscosity (Pa.s)\n1,"1,234.5"\n10,"1,100.2"\n100,"1,000"\n'
    (block,) = read_text(tmp_path, "q.csv", quoted).blocks
    assert col(block, "viscosity").values == [1234.5, 1100.2, 1000.0]
    thou = "Shear rate (1/s)\tViscosity (Pa.s)\n1\t12,500\n10\t3.25\n100\t1.5\n"
    (block,) = read_text(tmp_path, "thou.txt", thou).blocks
    assert col(block, "viscosity").values == [12500.0, 3.25, 1.5]


def test_a9_cr_line_endings_are_read(tmp_path):
    (block,) = read_text(
        tmp_path, "cr.csv", "Shear rate (1/s),Viscosity (Pa.s)\r1,10\r10,5\r"
    ).blocks
    assert col(block, "viscosity").values == [10.0, 5.0]


def test_a9_a_file_the_csv_module_refuses_is_reported_on_its_own(client):
    huge = 'Shear rate (1/s),Viscosity (Pa.s)\n"' + "x" * 200_000 + '",1\n1,10\n10,5\n'
    good = "Shear rate (1/s),Viscosity (Pa.s)\n1,10\n10,5\n"
    r = client.post(
        "/api/rheometry/parse",
        files=[("files", ("huge.csv", huge.encode())), ("files", ("good.csv", good.encode()))],
    )
    assert r.status_code == 200
    body = r.json()
    assert [d["filename"] for d in body["datasets"]] == ["good.csv"]
    assert [e["filename"] for e in body["errors"]] == ["huge.csv"]
    assert "could not be split into cells" in body["errors"][0]["message"]


def test_a10_a_units_row_with_mm_and_rpm_is_a_units_row(tmp_path):
    text = (
        "Time,Speed,Gap,Axial force,Viscosity,Shear rate\ns,rpm,mm,N,Pa·s,1/s\n"
        "1,10,1,0.1,5,1\n2,20,1,0.1,4,10\n3,30,1,0.1,3,100\n"
    )
    (block,) = read_text(tmp_path, "u.csv", text).blocks
    assert by_label(block, "Gap").source_unit == "mm"
    assert by_label(block, "Speed").source_unit == "rpm"
    assert col(block, "viscosity").values == [5.0, 4.0, 3.0]
    assert block.test_type == "flow_curve"


@pytest.mark.parametrize(
    "header",
    ["Normal stress (Pa)", "Elastic stress (Pa)", "Viscous stress (Pa)", "First normal stress"],
)
def test_a11_other_stresses_are_not_shear_stress(header):
    assert q.recognise(header) is None


def test_a11_a_symbol_beside_the_name_is_still_recognised():
    assert q.recognise("G' Storage modulus") == "storage_modulus"
    assert q.recognise("Shear stress τ (Pa)") == "stress"


def test_a12_frequency_in_rad_per_s_is_angular_frequency(tmp_path):
    text = "Frequency (rad/s),G' (Pa),G'' (Pa)\n0.1,1,2\n1,10,20\n10,100,200\n"
    (block,) = read_text(tmp_path, "f.csv", text).blocks
    assert col(block, "angular_frequency").values == [0.1, 1.0, 10.0]
    assert block.test_type == "frequency_sweep"


def _block(**cols):
    n = len(next(iter(cols.values())))
    return Block(
        id="b",
        name="b",
        n_points=n,
        columns=[Column(key=k, label=k, quantity=k, values=v) for k, v in cols.items()],
    )


def test_a13_derived_phase_angle_is_in_the_right_quadrant():
    block = derive(_block(storage_modulus=[-1.0, 1.0], loss_modulus=[1.0, 1.0]))
    phase = next(c for c in block.columns if c.quantity == "phase_angle").values
    assert phase == pytest.approx([135.0, 45.0])


def test_a14_one_blank_line_does_not_split_a_table(tmp_path):
    text = "Shear rate (1/s),Viscosity (Pa.s)\n1,10\n10,5\n\n100,2\n1000,1\n"
    ds = read_text(tmp_path, "b.csv", text)
    (block,) = ds.blocks
    assert col(block, "shear_rate").values == [1.0, 10.0, 100.0, 1000.0]
    assert block.n_points == 4


AP_TEXT = """Project:\tP1
Test:\tFreq sweeps
Result:\tSample A

Interval and data points:\t1\t3
Meas. Pts.\tAngular Frequency\tStorage Modulus\tLoss Modulus\tTemperature
\t[rad/s]\t[Pa]\t[Pa]\t[°C]
1\t1\t10\t20\t25.0
2\t10\t100\t150\t25.0
3\t100\t1000\t900\t25.0

Interval and data points:\t2\t3
Meas. Pts.\tAngular Frequency\tStorage Modulus\tLoss Modulus\tTemperature
\t[rad/s]\t[Pa]\t[Pa]\t[°C]
1\t1\t5\t20\t50.0
2\t10\t50\t150\t50.0
3\t100\t500\t900\t50.0
"""


def test_a15_rheocompass_intervals_keep_their_numbers(tmp_path):
    ds = read_text(tmp_path, "ap.txt", AP_TEXT)
    assert [b.name for b in ds.blocks] == ["Sample A, Interval 1", "Sample A, Interval 2"]
    older = "\n".join(line for line in AP_TEXT.splitlines() if not line.startswith("Interval"))
    ds = read_text(tmp_path, "ap2.txt", older)
    assert [b.name for b in ds.blocks] == ["Sample A, Interval 1", "Sample A, Interval 2"]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"Temperature": "25 C"}, 25.0),
        ({"Temperature": "298.15 K"}, 25.0),
        ({"Test temperature": "77 °F"}, 25.0),
        ({"Temperature": "25,5 °C"}, 25.5),
        ({"Temperature range": "-20 to 200 °C"}, None),
        ({"Temperature": "25 Pa"}, None),
    ],
)
def test_a16_metadata_temperature_respects_unit_and_ranges(metadata, expected):
    assert metadata_temperature(metadata) == expected


def test_a16_a_kelvin_metadata_temperature_is_converted_on_import(tmp_path):
    text = (
        "Temperature: 298.15 K\nAngular frequency (rad/s),G' (Pa),G'' (Pa)\n1,10,20\n10,100,150\n"
    )
    assert read_text(tmp_path, "k.csv", text).blocks[0].temperature == 25.0


def test_a17_a_temperature_ramp_under_auto_strain_is_a_temperature_sweep():
    block = _block(
        storage_modulus=[1e6, 1e5, 1e3, 10],
        strain=[0.01, 0.1, 1, 10],
        angular_frequency=[6.28] * 4,
        temperature=[20, 60, 100, 140],
    )
    assert detect_test_type(derive(block)) == "temperature_sweep"
    # An amplitude sweep at one temperature is still an amplitude sweep.
    sweep = _block(storage_modulus=[1, 2, 3], strain=[0.1, 1, 10], temperature=[25, 25.1, 25])
    assert detect_test_type(derive(sweep)) == "amplitude_sweep"


def test_a18_a_malformed_project_entry_is_never_accepted(client):
    good = client.post(
        "/api/rheometry/parse",
        files=[("files", ("g.csv", b"Shear rate (1/s),Viscosity (Pa.s)\n1,10\n10,5\n"))],
    ).json()["datasets"][0]
    bad_entries = [
        {"id": "x", "blocks": [{}]},
        {"id": "x", "filename": "a", "format_label": "b", "blocks": [{"id": "b", "name": "n"}]},
        dict(good, blocks=[dict(good["blocks"][0], n_points=99)]),
        dict(good, blocks=[dict(good["blocks"][0], columns=[{"key": "k", "values": "no"}])]),
        dict(good, warnings="not a list"),
        None,
        "text",
    ]
    each = "function (d) { return hooks.validDataset(d); }"
    expr = f"[hooks.validDataset({json.dumps(good)})].concat({json.dumps(bad_entries)}.map({each}))"
    assert run_mydata(expr) == [True] + [False] * len(bad_entries)


def test_e9_validation_errors_read_as_sentences_in_the_page():
    detail = [{"loc": ["body", "block"], "msg": "Value error, Column “a” holds 2 values"}]
    got = run_mydata(f"hooks.detailText({json.dumps(detail)})")
    assert got == "Column “a” holds 2 values"
    assert run_mydata('hooks.detailText("Plain")') == "Plain"


def test_a19_block_temperature_follows_a_corrected_column(client):
    text = "Temperature (°C),Time (s),G' (Pa)\n25,1,10\n25,2,20\n25,3,30\n"
    block = _parsed(client, "t.csv", text)
    assert block["temperature"] == 25.0
    updated = _rederive(client, block, "temperature", None)
    assert updated["temperature"] is None


def test_a20_a_second_table_in_a_step_does_not_inherit_the_first(tmp_path):
    text = (
        "Filename\tx.tri\n\n[step]\nTwo tables\n"
        "Angular frequency\tStorage modulus\tLoss modulus\nrad/s\tPa\tPa\n1\t1\t2\n10\t10\t20\n\n"
        "Angular frequency\tStorage modulus\tLoss modulus\nrad/s\tkPa\tkPa\n1\t1\t2\n10\t10\t20\n"
    )
    ds = read_text(tmp_path, "trios.txt", text)
    assert [b.metadata for b in ds.blocks] == [{}, {}]
    assert col(ds.blocks[1], "storage_modulus").values == [1000.0, 10000.0]


def test_a21_dot_thousands_in_a_decimal_comma_file(tmp_path):
    assert tabular.parse_number("1.234", True) == 1234.0
    text = "Scherrate [1/s];Viskosität [Pa·s]\n1;1.234,5\n10;1.234\n100;12,5\n1000;3,25\n"
    (block,) = read_text(tmp_path, "de.csv", text).blocks
    assert col(block, "viscosity").values == [1234.5, 1234.0, 12.5, 3.25]


def test_a22_time_cells_are_read_in_seconds(tmp_path):
    import openpyxl

    assert tabular.parse_number("00:01:30") == 90.0
    assert tabular.parse_number("1:00:00,5", True) == 3600.5
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Time", "G'", "G''"])
    for seconds in (10, 20, 30):
        sheet.append([datetime.time(0, 0, seconds), 1, 2])
    sheet.append([datetime.timedelta(seconds=40), 1, 2])
    path = tmp_path / "t.xlsx"
    book.save(path)
    (block,) = read_file(path, "t.xlsx").blocks
    assert col(block, "time").values == [10.0, 20.0, 30.0, 40.0]


def test_a23_dynamic_viscosity_is_eta_prime_not_steady_shear_viscosity():
    assert q.recognise("Dynamic viscosity (Pa.s)") == "dynamic_viscosity"
    assert q.QUANTITY_BY_KEY["dynamic_viscosity"].symbol == "η′"
    assert q.recognise("Viscosity") == "viscosity"


def test_a24_a_table_without_a_header_is_noted(tmp_path):
    ds = read_text(tmp_path, "n.csv", "1,2,3\n0.1,10,100\n1,20,200\n")
    assert any("no header row" in w for w in ds.warnings)


def test_e6_the_row_cap_stops_splitting_early():
    # Past the cap, a field the csv module would refuse: stopping early never reaches it.
    text = "a,b\n" + "1,2\n" * 50 + '"' + "x" * 200_000 + '",1\n'
    with pytest.raises(ValueError, match="data points"):
        tabular.split_rows(text, max_rows=10)
    with pytest.raises(ValueError, match="could not be split"):
        tabular.split_rows(text)


def test_e6_a_file_over_the_cap_is_refused_with_the_usual_message(tmp_path):
    text = "Shear rate (1/s),Viscosity (Pa.s)\n" + "1,2\n" * 30
    with pytest.raises(ValueError, match="more than the 10 data points"):
        read_text(tmp_path, "big.csv", text, Limits(max_rows=10))


def test_e9_rederive_refuses_a_block_with_an_unknown_quantity_or_short_column(client):
    block = _parsed(client, "f.csv", "Shear rate (1/s),Viscosity (Pa.s)\n1,10\n10,5\n")
    bogus = json.loads(json.dumps(block))
    bogus["columns"][0]["quantity"] = "bogus"
    r = client.post("/api/rheometry/rederive", json={"block": bogus, "overrides": []})
    assert r.status_code == 422
    assert "“bogus”, which is not a quantity this app knows" in r.text
    short = json.loads(json.dumps(block))
    short["columns"][1]["values"] = [1.0]
    r = client.post(
        "/api/rheometry/rederive",
        json={"block": short, "overrides": [{"key": "shear_rate", "quantity": None}]},
    )
    assert r.status_code == 422
    assert "holds 1 values but the block has 2 points" in r.text
    assert "zip()" not in r.text


def test_a13_with_positive_moduli_the_phase_angle_is_atan_of_tan_delta():
    block = derive(_block(storage_modulus=[2.0], loss_modulus=[1.0]))
    tan = next(c for c in block.columns if c.quantity == "tan_delta").values
    assert tan == [pytest.approx(0.5)]
    assert math.isclose(
        next(c for c in block.columns if c.quantity == "phase_angle").values[0],
        math.degrees(math.atan(0.5)),
    )


# --- Follow-up: the graph pages skip a damaged stored dataset, with the same check -------------


def _good_dataset():
    return {
        "id": "d1",
        "filename": "a.csv",
        "format_label": "Delimited text",
        "warnings": [],
        "blocks": [
            {
                "id": "d1-b1",
                "name": "Table 1",
                "n_points": 2,
                "temperature": None,
                "columns": [
                    {"key": "c0", "label": "w", "quantity": "angular_frequency", "values": [1, 2]}
                ],
            }
        ],
    }


def run_store(expression: str):
    """Evaluate `expression` against the real store.js in Node (no IndexedDB there)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = (
        "const fs = require('fs'); global.window = {};"
        f"eval(fs.readFileSync({json.dumps(str(STORE_JS))}, 'utf8'));"
        "const store = global.window.ladaRheoStore;"
        f"console.log(JSON.stringify({expression}));"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_followup_the_store_checks_datasets_for_every_page():
    good = _good_dataset()
    short = _good_dataset()
    short["blocks"][0]["columns"][0]["values"] = [1]  # n_points says 2
    no_blocks = {k: v for k, v in good.items() if k != "blocks"}
    known = "function (k) { return k === 'angular_frequency'; }"
    unknown = "function (k) { return false; }"
    assert run_store(
        f"[store.validDataset({json.dumps(good)}),"
        f" store.validDataset({json.dumps(good)}, {known}),"
        f" store.validDataset({json.dumps(good)}, {unknown}),"
        f" store.validDataset({json.dumps(short)}),"
        f" store.validDataset({json.dumps(no_blocks)}),"
        " store.validDataset(null), store.validDataset('text')]"
    ) == [True, True, False, False, False, False, False]


def test_followup_my_data_uses_the_stores_check():
    bad = _good_dataset()
    bad["blocks"][0]["columns"][0]["quantity"] = "bogus"
    assert run_mydata(
        f"[hooks.validDataset({json.dumps(_good_dataset())}),"
        f" hooks.validDataset({json.dumps(bad)})]"
    ) == [True, False]


def test_followup_the_graph_pages_filter_stored_datasets_through_the_check():
    source = STUDIO_JS.read_text(encoding="utf-8")
    assert "store.validDataset(dataset" in source
    assert "could not be read and is left out" in source
