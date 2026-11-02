"""The Rheometer workspace's import: quantity registry, table finding, readers, API, page.

The fixtures in tests/data/rheometry are synthetic (see generate_fixtures.py there): they follow
the commonly documented TRIOS and RheoCompass layouts but are not real exports.
"""

import math
import re
import zipfile
from pathlib import Path

import pytest

from lada_web.rheometry import quantities as q
from lada_web.rheometry import tabular
from lada_web.rheometry.derive import derive
from lada_web.rheometry.detect import detect_test_type
from lada_web.rheometry.model import Block, Column
from lada_web.rheometry.readers import Limits, read_file

RHEO = Path(__file__).resolve().parent / "data" / "rheometry"


def column(block, quantity, derived=None):
    for c in block.columns:
        if c.quantity == quantity and (derived is None or c.derived == derived):
            return c
    raise AssertionError(f"{block.name} has no {quantity} column")


# --- Quantity registry -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "quantity"),
    [
        ("Storage modulus", "storage_modulus"),
        ("G'", "storage_modulus"),
        ("Storage Modulus [Pa]", "storage_modulus"),
        ("Speichermodul", "storage_modulus"),
        ('G"', "loss_modulus"),
        ("Loss Factor", "tan_delta"),
        ("Tan(delta)", "tan_delta"),
        ("Complex viscosity", "complex_viscosity"),
        ("Viscosity (mPa.s)", "viscosity"),
        ("Angular frequency", "angular_frequency"),
        ("Shear Strain", "strain"),
        ("Shear rate (1/s)", "shear_rate"),
        ("Step time", "time"),
        ("Temperature", "temperature"),
        ("G(t)", "relaxation_modulus"),
        ("J(t)", "compliance"),
        ("Torque", "torque"),
        ("Point No.", None),
        ("Gap", None),
    ],
)
def test_headers_are_recognised_by_whole_name(header, quantity):
    assert q.recognise(header) == quantity


def test_complex_viscosity_is_never_read_as_viscosity():
    assert q.recognise("Complex viscosity") == "complex_viscosity"
    assert q.recognise("Viscosity") == "viscosity"


@pytest.mark.parametrize(
    ("header", "name", "unit"),
    [
        ("Storage modulus (Pa)", "Storage modulus", "Pa"),
        ("G' [kPa]", "G'", "kPa"),
        ("Viscosity in mPa.s", "Viscosity", "mPa.s"),
        ("Tan(delta)", "Tan(delta)", None),
        ("J(t)", "J(t)", None),
    ],
)
def test_a_unit_written_into_the_header_is_split_off(header, name, unit):
    assert q.split_header(header) == (name, unit)


def test_units_convert_to_the_canonical_unit():
    assert q.to_canonical([2.0], q.parse_unit("kPa", "storage_modulus")) == [2000.0]
    assert q.to_canonical([500.0], q.parse_unit("mPa.s", "viscosity")) == [0.5]
    assert q.to_canonical([373.15], q.parse_unit("K", "temperature"))[0] == pytest.approx(100.0)
    assert q.to_canonical([0.5], q.parse_unit("", "strain")) == [50.0]
    assert q.parse_unit("Pa", "temperature") is None


def test_the_registry_the_browser_gets_converts_the_same_way():
    for entry in q.registry_json():
        spec = q.QUANTITY_BY_KEY[entry["key"]]
        assert entry["unit"] == spec.canonical.symbol
        assert entry["units"][0] == {"symbol": spec.canonical.symbol, "factor": 1.0, "offset": 0.0}


# --- Table finding ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "comma", "value"),
    [
        ("1.5", False, 1.5),
        ("1,5", True, 1.5),
        ("1,23E+5", True, 1.23e5),
        ("1,5", False, 1.5),
        ("-2.5e-3", False, -2.5e-3),
        ("", False, None),
        ("NaN", False, None),
        ("abc", False, None),
    ],
)
def test_numbers_are_read_with_either_decimal_mark(cell, comma, value):
    assert tabular.parse_number(cell, comma) == value


@pytest.mark.parametrize(
    ("text", "delimiter"),
    [
        ("a\tb\tc\n1\t2\t3\n4\t5\t6", "\t"),
        ("a;b;c\n1,5;2,5;3,5\n4;5;6", ";"),
        ("a,b,c\n1,2,3\n4,5,6", ","),
        ("a b c\n1 2 3\n4 5 6", None),
        ("a b\n1,5 2,5\n3,5 4,5\n5,5 6,5", None),
    ],
)
def test_the_delimiter_is_sniffed(text, delimiter):
    assert tabular.sniff_delimiter(text) == delimiter


def test_a_units_row_under_the_header_is_found_and_applied():
    rows = [["Angular frequency", "Storage modulus"], ["rad/s", "kPa"], ["1", "2"], ["10", "3"]]
    (table,) = tabular.find_tables(rows, decimal_comma=False)
    columns = tabular.build_columns(table, False)
    assert table.units == ["rad/s", "kPa"]
    assert columns[1].values == [2000.0, 3000.0]
    assert columns[1].source_unit == "kPa"


def test_an_unknown_unit_keeps_the_column_unconverted_and_unrecognised():
    rows = [["Storage modulus"], ["furlongs"], ["1"], ["2"]]
    (table,) = tabular.find_tables(rows, decimal_comma=False)
    (col,) = tabular.build_columns(table, False)
    assert col.quantity is None
    assert col.values == [1.0, 2.0]


def test_separate_tables_in_one_file_are_kept_apart():
    rows = [["x", "y"], ["1", "2"], ["3", "4"], [], ["Second"], ["a", "b"], ["5", "6"], ["7", "8"]]
    tables = tabular.find_tables(rows, decimal_comma=False)
    assert [t.header for t in tables] == [["x", "y"], ["a", "b"]]


# --- Readers ---------------------------------------------------------------------------------


def test_a_trios_export_splits_into_its_steps():
    ds = read_file(RHEO / "trios_frequency_sweeps.txt", "trios_frequency_sweeps.txt")
    assert ds.format == "trios"
    assert ds.metadata["Instrument name"] == "DHR-2"
    assert [b.name for b in ds.blocks] == [
        "Frequency sweep (150 °C) - 1",
        "Frequency sweep (160 °C) - 2",
        "Frequency sweep (170 °C) - 3",
    ]
    assert [b.test_type for b in ds.blocks] == ["frequency_sweep"] * 3
    assert [b.temperature for b in ds.blocks] == [150.0, 160.0, 170.0]
    assert all(b.n_points == 13 for b in ds.blocks)


def test_derived_quantities_match_their_definitions():
    block = read_file(RHEO / "trios_frequency_sweeps.txt", "t.txt").blocks[0]
    omega = column(block, "angular_frequency").values
    g1 = column(block, "storage_modulus").values
    g2 = column(block, "loss_modulus").values
    gstar = column(block, "complex_modulus", derived=True).values
    eta = column(block, "complex_viscosity", derived=True).values
    freq = column(block, "frequency", derived=True).values
    for i in range(len(omega)):
        assert gstar[i] == pytest.approx(math.hypot(g1[i], g2[i]))
        assert eta[i] == pytest.approx(gstar[i] / omega[i])
        assert freq[i] == pytest.approx(omega[i] / (2 * math.pi))
    # tan δ came from the file, so it is not derived a second time.
    assert [c.derived for c in block.columns if c.quantity == "tan_delta"] == [False]


def test_a_rheocompass_export_reads_decimal_commas_and_bracketed_units():
    ds = read_file(RHEO / "anton_paar_amplitude_sweep.csv", "ap.csv")
    assert ds.format == "anton_paar"
    assert ds.metadata["Project"] == "Gel study"
    (block,) = ds.blocks
    assert block.name == "Gel A, Interval 1"
    assert block.test_type == "amplitude_sweep"
    assert block.temperature == 25.0
    g1 = column(block, "storage_modulus")
    assert g1.source_unit == "kPa" and g1.unit == "Pa"
    assert g1.values[0] == pytest.approx(12.5 / (1 + 0.01 / 50) * 1000, rel=1e-3)
    assert column(block, "strain").values[0] == pytest.approx(0.01)
    index = [c for c in block.columns if c.is_index]
    assert [c.label for c in index] == ["Point No."]
    assert any("decimal commas" in w for w in ds.warnings)


def test_a_plain_csv_flow_curve_with_units_in_the_header():
    ds = read_file(RHEO / "flow_curve.csv", "flow_curve.csv")
    assert ds.format == "delimited"
    assert ds.metadata == {"Sample": "PEO 2 %"}
    (block,) = ds.blocks
    assert block.test_type == "flow_curve"
    eta = column(block, "viscosity")
    assert eta.source_unit == "mPa.s" and eta.values[0] == pytest.approx(1.0, rel=1e-2)


def test_a_whitespace_table_with_comment_lines():
    ds = read_file(RHEO / "relaxation.dat", "relaxation.dat")
    (block,) = ds.blocks
    assert block.test_type == "stress_relaxation"
    assert block.temperature == 25.0  # from the "# Temperature: 25 C" comment
    assert column(block, "relaxation_modulus").values[0] == pytest.approx(1e4 * math.exp(-0.01))


def test_a_workbook_reads_every_sheet():
    ds = read_file(RHEO / "two_samples.xlsx", "two_samples.xlsx")
    assert ds.format == "excel"
    assert [b.name for b in ds.blocks] == ["Sample A", "Sample B"]
    assert [b.test_type for b in ds.blocks] == ["frequency_sweep", "frequency_sweep"]
    # Sample B gives its moduli in MPa in a units row.
    g1_b = column(ds.blocks[1], "storage_modulus").values
    assert g1_b[0] == pytest.approx(0.001 * 0.1 * 1e6)
    # Sample A gives "ω (rad/s)" as its header.
    assert column(ds.blocks[0], "angular_frequency").values[0] == pytest.approx(0.1)


def test_a_file_without_a_table_says_what_a_readable_one_looks_like():
    with pytest.raises(ValueError, match="No table of numbers"):
        read_file(RHEO / "not_a_table.txt", "not_a_table.txt")


def test_utf16_exports_are_decoded(tmp_path):
    path = tmp_path / "utf16.txt"
    text = (RHEO / "flow_curve.csv").read_text(encoding="utf-8")
    path.write_bytes(text.encode("utf-16"))
    assert read_file(path, "utf16.txt").blocks[0].test_type == "flow_curve"


def test_too_many_points_are_refused():
    with pytest.raises(ValueError, match="data points"):
        read_file(RHEO / "trios_frequency_sweeps.txt", "t.txt", Limits(max_rows=10))


def test_a_workbook_that_unpacks_too_large_is_refused_before_it_is_opened():
    with pytest.raises(ValueError, match="unpacks to more than"):
        read_file(RHEO / "two_samples.xlsx", "x.xlsx", Limits(max_xlsx_uncompressed_bytes=10))


def test_a_damaged_workbook_gives_a_readable_error(tmp_path):
    for name, payload in (("bad.xlsx", b"PK\x03\x04 not really"), ("bad.xls", b"nope")):
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="not a readable"):
            read_file(path, name)
    assert zipfile.is_zipfile(RHEO / "two_samples.xlsx")


def _block(**cols):
    return Block(
        id="b",
        name="b",
        n_points=3,
        columns=[Column(key=k, label=k, quantity=k, values=v) for k, v in cols.items()],
    )


@pytest.mark.parametrize(
    ("cols", "expected"),
    [
        ({"storage_modulus": [1, 2, 3], "strain": [0.1, 1, 10]}, "amplitude_sweep"),
        ({"storage_modulus": [1, 2, 3], "temperature": [100, 120, 140]}, "temperature_sweep"),
        ({"storage_modulus": [1, 2, 3], "angular_frequency": [0.1, 1, 10]}, "frequency_sweep"),
        ({"compliance": [1, 2, 3], "time": [1, 10, 100]}, "creep"),
        ({"viscosity": [3, 2, 1], "shear_rate": [0.1, 1, 10]}, "flow_curve"),
        ({"torque": [1, 2, 3]}, "unknown"),
    ],
)
def test_test_types_follow_what_varies(cols, expected):
    assert detect_test_type(derive(_block(**cols))) == expected


# --- API -------------------------------------------------------------------------------------


def _files(*names):
    return [("files", (name, (RHEO / name).read_bytes())) for name in names]


def test_parse_reads_several_files_and_reports_the_unreadable_one_by_name(client):
    r = client.post(
        "/api/rheometry/parse",
        files=_files("trios_frequency_sweeps.txt", "not_a_table.txt", "two_samples.xlsx"),
    )
    assert r.status_code == 200
    body = r.json()
    assert [d["filename"] for d in body["datasets"]] == [
        "trios_frequency_sweeps.txt",
        "two_samples.xlsx",
    ]
    assert [e["filename"] for e in body["errors"]] == ["not_a_table.txt"]
    assert "No table of numbers" in body["errors"][0]["message"]


def test_parse_without_files_is_a_validation_error(client):
    assert client.post("/api/rheometry/parse").status_code == 422


def test_the_quantities_route_carries_units_formats_and_test_types(client):
    body = client.get("/api/rheometry/quantities").json()
    keys = {entry["key"] for entry in body["quantities"]}
    assert {"storage_modulus", "loss_modulus", "complex_viscosity", "shear_rate"} <= keys
    assert body["formats"]["trios"] == "TA Instruments TRIOS export"
    assert body["test_types"]["frequency_sweep"] == "Frequency sweep"
    assert ".xlsx" in body["accepted"] and ".dat" in body["accepted"]


def _block_from(client, name):
    body = client.post("/api/rheometry/parse", files=_files(name)).json()
    return body["datasets"][0]["blocks"][0]


def _override(key, quantity, unit=None):
    return {"key": key, "quantity": quantity, "unit": unit}


def test_rederive_converts_a_column_into_the_unit_the_user_names(client):
    block = _block_from(client, "anton_paar_amplitude_sweep.csv")
    time = next(c for c in block["columns"] if c["quantity"] == "time")
    assert time["values"][:2] == [10.0, 20.0]
    r = client.post(
        "/api/rheometry/rederive",
        json={"block": block, "overrides": [_override(time["key"], "time", "min")]},
    )
    assert r.status_code == 200
    updated = next(c for c in r.json()["columns"] if c["key"] == time["key"])
    assert updated["values"][:2] == [600.0, 1200.0]
    assert updated["source_unit"] == "min"


def test_rederive_recomputes_what_depends_on_a_corrected_column(client):
    block = _block_from(client, "flow_curve.csv")
    eta = next(c for c in block["columns"] if c["quantity"] == "viscosity")
    r = client.post(
        "/api/rheometry/rederive",
        json={"block": block, "overrides": [_override(eta["key"], None)]},
    )
    body = r.json()
    assert body["test_type"] == "unknown"
    plain = next(c for c in body["columns"] if c["key"] == eta["key"])
    # Back to the numbers as the file wrote them, in mPa·s.
    assert plain["quantity"] is None
    assert plain["values"][0] == pytest.approx(1000 / 1.01**0.6, rel=1e-4)


def test_rederive_refuses_a_unit_that_does_not_belong_to_the_quantity(client):
    block = _block_from(client, "flow_curve.csv")
    key = block["columns"][0]["key"]
    r = client.post(
        "/api/rheometry/rederive",
        json={"block": block, "overrides": [_override(key, "time", "Pa")]},
    )
    assert r.status_code == 422
    assert "not a unit of time" in r.json()["detail"]


def test_rederive_refuses_a_derived_or_missing_column(client):
    block = _block_from(client, "trios_frequency_sweeps.txt")
    derived = next(c for c in block["columns"] if c["derived"])
    for key in (derived["key"], "nope"):
        r = client.post(
            "/api/rheometry/rederive",
            json={"block": block, "overrides": [_override(key, None)]},
        )
        assert r.status_code == 422


def test_the_my_data_page_offers_a_real_file_input_and_loads_its_scripts(client):
    r = client.get("/rheometer/my-data")
    assert r.status_code == 200
    assert re.search(r'<input type="file" id="rheo-files" multiple', r.text)
    for script in ("js/rheometry/store.js", "js/rheometry/mydata.js"):
        assert re.search(rf'src="/static/{re.escape(script)}\?v=[0-9a-f]{{10}}"', r.text)
        assert client.get(f"/static/{script}").status_code == 200
    assert "This tool is not ready yet." not in r.text


# --- Graph studio ----------------------------------------------------------------------------

GRAPH_PAGES = {
    "frequency-sweep": "frequency_sweep",
    "amplitude-sweep": "amplitude_sweep",
    "temperature-sweep": "temperature_sweep",
    "flow-curve": "flow_curve",
    "creep": "creep",
    "stress-relaxation": "stress_relaxation",
    "master-curve": "master_curve",
    "custom-plot": "custom",
}


@pytest.mark.parametrize(("slug", "preset"), GRAPH_PAGES.items())
def test_every_graph_page_is_the_studio_with_its_own_preset(client, slug, preset):
    r = client.get(f"/rheometer/{slug}")
    assert r.status_code == 200
    assert f'<article class="calc studio" data-preset="{preset}">' in r.text
    for part in ('id="studio-data"', 'id="studio-plot"', 'id="studio-form"'):
        assert part in r.text
    for script in ("store.js", "presets.js", "studio.js"):
        assert re.search(rf'src="/static/js/rheometry/{script}\?v=[0-9a-f]{{10}}"', r.text)
    assert "This tool is not ready yet." not in r.text


def test_every_page_preset_is_defined_in_the_presets_script(client):
    js = client.get("/static/js/rheometry/presets.js").text
    block = js[js.index("var PRESETS = {") : js.index("var PALETTES")]
    defined = set(re.findall(r"^    ([a-z_]+): \{", block, re.M))
    assert set(GRAPH_PAGES.values()) == defined


def test_preset_quantities_exist_in_the_registry(client):
    """A preset that names a quantity the importer never produces would plot nothing."""
    js = client.get("/static/js/rheometry/presets.js").text
    block = js[js.index("var PRESETS = {") : js.index("var PALETTES")]
    named = set(re.findall(r'"([a-z_]+)"', block)) - {
        "",
        "frequency_sweep",
        "amplitude_sweep",
        "temperature_sweep",
        "flow_curve",
        "creep",
        "stress_relaxation",
        "master_curve",
    }
    assert named <= set(q.QUANTITY_BY_KEY), named - set(q.QUANTITY_BY_KEY)


def test_the_studio_keeps_its_choices_but_never_the_data_in_local_storage(client):
    js = client.get("/static/js/rheometry/studio.js").text
    assert '"lada-rheo-spec:"' in js
    # The datasets are only ever read from the IndexedDB store.
    assert "store.all()" in js and "localStorage.setItem(SPEC_KEY" in js
