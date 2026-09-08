import re

import pytest
from starlette.testclient import TestClient

from lada_web.catalog import CALCULATIONS
from lada_web.config import Settings
from lada_web.main import create_app
from lada_web.rendering import templates

from .conftest import upload


def test_index_lists_every_calculation(client):
    r = client.get("/")
    assert r.status_code == 200
    for calc in CALCULATIONS:
        assert calc.path in r.text
        assert calc.title in r.text


def test_index_explains_the_flow_and_the_input_each_page_needs(client):
    r = client.get("/")
    assert "Upload" in r.text and "Download" in r.text
    assert "Needs a LAMMPS dump file" in r.text
    assert "Needs a NumPy .npz trajectory" in r.text


def test_reference_page_has_form_contract(client):
    r = client.get("/parsers/log")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'hx-post="/api/parsers/log/table"' in r.text
    assert 'id="results"' in r.text
    assert 'name="file"' in r.text
    assert 'hx-encoding="multipart/form-data"' in r.text
    form_tag = re.search(r"<form[^>]*>", r.text).group(0)
    assert "action=" not in form_tag


@pytest.mark.parametrize("calc", CALCULATIONS, ids=lambda c: f"{c.section}-{c.slug}")
def test_every_calculation_page_renders(client, calc):
    r = client.get(calc.path)
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert '<label for="file">' in r.text
    assert r.text.count("<form") == 1
    assert "hx-post=" in r.text or "formaction=" in r.text


def test_unknown_page_is_404_json_and_html(client):
    r = client.get("/parsers/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "There is no page at this address."
    r = client.get("/nope/nope", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "There is no page at this address." in r.text


def test_error_page_and_error_partial_are_siblings(client, htmx, data_dir):
    """One look for a fault, whether it arrives as a whole page or swapped into #results."""
    page = client.get("/nope/nope", headers={"Accept": "text/html"})
    partial = client.post(
        "/api/parsers/acf/table", files=upload(data_dir / "sample_log.lammps"), headers=htmx
    )
    assert partial.status_code == 422
    for r in (page, partial):
        assert 'class="result result-error"' in r.text
    # Only the swapped-in partial is a live region; the page announces itself with its h1.
    assert 'role="alert"' in partial.text
    assert 'role="alert"' not in page.text


def test_static_assets_served(client):
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200


def test_api_docs_available(client):
    assert client.get("/api/docs").status_code == 200


def test_settings_are_injected(app):
    assert app.state.settings.max_upload_mb == 1


# --- Shared template contracts ---------------------------------------------


def macros():
    return templates.env.get_template("_macros.html").module


FIELD_MACROS = [
    ("file", lambda m: m.file_field("file", "LAMMPS log file", ".log", "The file to read.")),
    ("count", lambda m: m.number_field("count", "Count", 3, step=1, min=1, help="How many.")),
    ("cols", lambda m: m.text_field("cols", "Columns", "x,y,z", help="Three names.")),
    (
        "section",
        lambda m: m.select_field(
            "section", "Section", [("Atoms", "Atoms")], value="Atoms", help="Which section."
        ),
    ),
    ("custom", lambda m: m.textarea_field("custom", "Chains", "0,7", help="One chain per line.")),
    ("log_x", lambda m: m.checkbox_field("log_x", "Log x", checked=True, help="Log scale.")),
]


@pytest.mark.parametrize(("name", "render"), FIELD_MACROS, ids=[m[0] for m in FIELD_MACROS])
def test_every_field_macro_labels_its_own_control(name, render):
    """Each field macro must emit a label bound to the control id it also emits."""
    html = render(macros())
    assert f'<label for="{name}"' in html
    assert f'id="{name}"' in html
    assert f'aria-describedby="{name}-help"' in html
    assert f'id="{name}-help"' in html


def test_calculate_button_and_results_area_keep_their_htmx_contract():
    m = macros()
    button = m.calculate_button("Read thermo table", "/api/parsers/log/table")
    assert 'type="submit"' in button
    assert 'hx-post="/api/parsers/log/table"' in button
    assert 'hx-target="#results"' in button
    assert 'hx-disabled-elt="this"' in button
    area = m.results_area()
    assert 'id="results-indicator"' in area and 'id="results"' in area


def test_truncated_result_says_how_many_rows_are_shown(temp_root, data_dir, htmx):
    settings = Settings(max_upload_mb=1, max_table_rows=2, temp_root=temp_root)
    client = TestClient(create_app(settings), raise_server_exceptions=False)
    r = client.post(
        "/api/parsers/log/table", files=upload(data_dir / "sample_log.lammps"), headers=htmx
    )
    assert r.status_code == 200
    assert "Showing the first 2 of 5 rows" in r.text
    assert "Downloads contain every row." in r.text


def test_an_untruncated_result_has_no_note(client, data_dir, htmx):
    r = client.post(
        "/api/parsers/log/table", files=upload(data_dir / "sample_log.lammps"), headers=htmx
    )
    assert r.status_code == 200
    assert "Showing the first" not in r.text


# --- Quality floor ---------------------------------------------------------

POSITIVE_FIELDS = [
    ("/dynamics/segment-acf", ["time_per_frame"]),
    ("/dynamics/rouse-mode", ["time_per_frame"]),
    ("/dynamics/isf", ["time_per_frame", "q_magnitude"]),
    ("/rheology/stress-relaxation", ["volume", "temperature"]),
    ("/rheology/dynamic-moduli", ["volume", "temperature", "t_min", "omega_min", "omega_max"]),
]


@pytest.mark.parametrize(("path", "names"), POSITIVE_FIELDS, ids=[f[0] for f in POSITIVE_FIELDS])
def test_fields_the_server_needs_positive_do_not_offer_zero(client, path, names):
    """The browser should say no to 0 inline instead of the form coming back as a 422."""
    r = client.get(path)
    for name in names:
        field = re.search(rf'<input[^>]*name="{name}"[^>]*>', r.text)
        assert field is not None, f"{name} is not on {path}"
        assert 'min="0"' not in field.group(0), f"{path}: {name} still accepts 0"
        assert 'min="1e-12"' in field.group(0), f"{path}: {name} has no positive minimum"


def test_alpine_panes_stay_hidden_until_alpine_boots(client):
    page = client.get("/dynamics/segment-acf")
    assert "x-cloak" in page.text
    css = client.get("/static/css/app.css")
    assert "[x-cloak]" in css.text


def test_a_wide_table_can_be_scrolled_from_the_keyboard(client, data_dir, htmx):
    r = client.post(
        "/api/parsers/dump/table", files=upload(data_dir / "sample_dump.dump"), headers=htmx
    )
    assert r.status_code == 200
    scroller = re.search(r'<div class="table-scroll"[^>]*>', r.text).group(0)
    assert 'tabindex="0"' in scroller
    assert "aria-label=" in scroller
