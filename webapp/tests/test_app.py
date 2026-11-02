import html
import os
import re

import pytest
from starlette.testclient import TestClient

from lada_web.catalog import (
    CALCULATIONS,
    RHEOMETER_CALCULATIONS,
    SECTION_BY_KEY,
    SECTIONS,
    SIMULATION_CALCULATIONS,
    WORKSPACE_BY_KEY,
    WORKSPACES,
)
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
    r = client.get("/simulations")
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


# conformations/dynamics/rheology accept several files to average; parsers/modifiers stay
# single-file.
MULTI_FILE_SECTIONS = {"conformations", "dynamics", "rheology"}


# conformations/dynamics/rheology also use the job-based Calculate button (real progress bar)
# instead of a plain htmx submit.
JOB_BASED_SECTIONS = {"conformations", "dynamics", "rheology"}


@pytest.mark.parametrize("calc", SIMULATION_CALCULATIONS, ids=lambda c: f"{c.section}-{c.slug}")
def test_every_calculation_page_renders(client, calc):
    r = client.get(calc.path)
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    field = "files" if calc.section in MULTI_FILE_SECTIONS else "file"
    assert f'<label for="{field}">' in r.text
    assert r.text.count("<form") == 1
    if calc.section in JOB_BASED_SECTIONS:
        assert 'onclick="window.ladaRunJob(this)"' in r.text
        assert "data-start-url=" in r.text
    else:
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


ACF_DOWNLOAD = "/api/parsers/acf/table/download?fmt=csv"
BACK_TO_ACF = '<a href="/parsers/acf">Back to Stress ACF file</a>'


def failing_download(client, data_dir, referer: str | None):
    """A download that 422s, sent the way a browser sends one: a native form submit."""
    headers = {"Accept": "text/html"}
    if referer is not None:
        headers["Referer"] = referer
    return client.post(ACF_DOWNLOAD, files=upload(data_dir / "sample_log.lammps"), headers=headers)


def test_the_error_page_offers_a_way_back_to_the_page_you_came_from(client, data_dir):
    """A download button is a native submit, so an error replaces the whole page."""
    r = failing_download(client, data_dir, "http://testserver/parsers/acf")
    assert r.status_code == 422
    assert BACK_TO_ACF in r.text
    assert "Back to the calculation list" in r.text


@pytest.mark.parametrize(
    "referer",
    [
        None,
        "",
        "http://evil.example/parsers/acf",
        "https://testserver/parsers/acf",
        "http://testserver/parsers/nope",
        "http://testserver/",
        "javascript:alert(1)//parsers/acf",
        "http://[abc/parsers/acf",
    ],
    ids=[
        "absent",
        "empty",
        "other-host",
        "other-scheme",
        "unknown-page",
        "index",
        "javascript",
        "malformed",
    ],
)
def test_the_error_page_falls_back_when_the_referer_is_not_one_of_our_pages(
    client, data_dir, referer
):
    """A malformed referer (e.g. an unclosed IPv6 bracket) must not turn the 422 into a 500."""
    r = failing_download(client, data_dir, referer)
    assert r.status_code == 422
    assert BACK_TO_ACF not in r.text
    assert "Back to the calculation list" in r.text


def test_the_error_partial_and_the_json_view_are_unchanged_by_the_back_link(client, data_dir, htmx):
    referer = {"Referer": "http://testserver/parsers/acf"}
    partial = client.post(
        "/api/parsers/acf/table",
        files=upload(data_dir / "sample_log.lammps"),
        headers=htmx | referer,
    )
    assert partial.status_code == 422 and "Back to" not in partial.text
    body = client.post(
        "/api/parsers/acf/table", files=upload(data_dir / "sample_log.lammps"), headers=referer
    )
    assert body.status_code == 422 and set(body.json()) == {"detail"}


def test_static_assets_served(client):
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200


def test_api_docs_available(client):
    assert client.get("/api/docs").status_code == 200


# The 13 view routes: every calculation except the download-only modifier, plus the dump
# preview, which has no download sibling of its own.
VIEW_ROUTES = [
    "/api/parsers/log/table",
    "/api/parsers/dump/preview",
    "/api/parsers/dump/table",
    "/api/parsers/data/table",
    "/api/parsers/acf/table",
    "/api/conformations/rg",
    "/api/conformations/ree",
    "/api/conformations/ree-vectors",
    "/api/dynamics/segment-acf",
    "/api/dynamics/rouse-mode",
    "/api/dynamics/isf",
    "/api/rheology/stress-relaxation",
    "/api/rheology/dynamic-moduli",
]


def test_the_result_models_reach_the_openapi_schema(app):
    """/api/docs is linked from every page, so the shape a route returns has to be in it."""
    schemas = app.openapi()["components"]["schemas"]
    assert "TableResult" in schemas
    assert "PlotSpec" in schemas


@pytest.mark.parametrize("path", VIEW_ROUTES)
def test_every_view_route_documents_its_200_body(app, path):
    ok = app.openapi()["paths"][path]["post"]["responses"]["200"]
    schema = ok["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/TableResult"), path


@pytest.mark.parametrize(
    "path",
    ["/api/parsers/log/table/download", "/api/modifiers/rewrite-end-beads"],
    ids=["download", "modifier"],
)
def test_file_routes_document_a_file_body_not_a_table(app, path):
    ok = app.openapi()["paths"][path]["post"]["responses"]["200"]
    assert "application/json" not in ok["content"]
    assert set(ok["content"]) >= {"text/csv", "text/plain"}


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
    assert "Showing the first 2 of 5 rows." in r.text
    assert "downloads carry every row at full precision." in r.text


def test_an_untruncated_result_does_not_claim_rows_were_hidden(client, data_dir, htmx):
    r = client.post(
        "/api/parsers/log/table", files=upload(data_dir / "sample_log.lammps"), headers=htmx
    )
    assert r.status_code == 200
    assert "Showing the first" not in r.text
    assert "downloads carry full precision." in r.text


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


# --- Navigation and page layout --------------------------------------------


def test_the_sidebar_lists_every_section_open(client):
    """No menus to open: every section and tool is in the page, in catalog order, always."""
    r = client.get("/parsers/log")
    positions = []
    for section in SECTIONS:
        group = re.search(rf'<div class="nav-group[^"]*"\s+data-section="{section.key}">', r.text)
        assert group is not None, f"no sidebar group for {section.key}"
        assert f'id="nav-section-{section.key}"' in r.text, section.key
        assert html.escape(section.title) in r.text
        positions.append(group.start())
    assert positions == sorted(positions), "sidebar sections are out of catalog order"
    sidebar = re.search(r'<nav id="site-nav".*?</nav>', r.text, re.S).group(0)
    assert "aria-expanded" not in sidebar, "a sidebar section can be collapsed"


def test_every_calculation_link_lives_in_its_section_group(client):
    r = client.get("/")
    for calc in CALCULATIONS:
        group = re.search(
            rf'data-section="{calc.section}">.*?</ul>',
            re.search(r'<nav id="site-nav".*?</nav>', r.text, re.S).group(0),
            re.S,
        )
        assert group is not None, calc.section
        assert f'href="{calc.path}"' in group.group(0), calc.path


@pytest.mark.parametrize("calc", CALCULATIONS, ids=lambda c: f"{c.section}-{c.slug}")
def test_the_navigation_names_the_section_the_page_and_the_way_back(client, calc):
    """Statements of place: workspace and section colour, the page in the sidebar, a breadcrumb."""
    r = client.get(calc.path)
    assert r.status_code == 200
    section = SECTION_BY_KEY[calc.section]
    workspace = WORKSPACE_BY_KEY[section.workspace]
    section_title = html.escape(section.title)

    # The whole page takes the section's colour and the workspace from these attributes.
    assert f'<body data-section="{calc.section}" data-workspace="{workspace.key}">' in r.text

    current_group = re.search(r'<div class="nav-group is-current"[^>]*>.*?</div>', r.text, re.S)
    assert current_group is not None, f"{calc.path} marks no current section"
    assert section_title in current_group.group(0)

    here = re.search(rf'href="{re.escape(calc.path)}"\s+aria-current="page"', r.text)
    assert here is not None, f"{calc.path} is not marked current in the sidebar"

    crumbs = re.search(r'<nav class="breadcrumb".*?</nav>', r.text, re.S).group(0)
    assert f'<a href="{workspace.home}">{workspace.title}</a>' in crumbs
    assert section_title in crumbs
    assert f'<span class="breadcrumb-current">{html.escape(calc.title)}</span>' in crumbs
    assert 'class="needs-chip"' in crumbs


def test_any_tool_is_a_keystroke_away(client):
    """The jump box is on every page and is filled from the sidebar, so the two never differ."""
    r = client.get("/conformations/rg")
    assert 'id="jump"' in r.text and 'id="jump-input"' in r.text
    assert "data-jump-open" in r.text
    js = client.get("/static/js/app.js").text
    assert "collectTools" in js and "showModal" in js
    assert '"lada-recent"' in js


@pytest.mark.parametrize("calc", SIMULATION_CALCULATIONS, ids=lambda c: f"{c.section}-{c.slug}")
def test_the_form_sits_left_of_the_results(client, calc):
    """Inputs and what came back share one screen, so a changed parameter shows its effect."""
    r = client.get(calc.path)
    assert 'class="workbench' in r.text
    assert r.text.index("workbench-inputs") < r.text.index('id="calc-form"')
    if calc.section == "modifiers":
        # This page answers with a file, so it has no results column to sit beside.
        assert "workbench-single" in r.text
        assert 'id="results"' not in r.text
    else:
        assert r.text.index('id="calc-form"') < r.text.index("workbench-output")
        assert r.text.index("workbench-output") < r.text.index('id="results"')


def test_the_results_column_says_what_will_appear_in_it(client):
    r = client.get("/rheology/dynamic-moduli")
    results = re.search(r'<div id="results"[^>]*>(.*?)</div>', r.text, re.S).group(1)
    assert "results-empty" in results
    assert "calculate" in results.lower()


def test_one_web_font_is_loaded_with_a_system_fallback(client):
    """One legibility-first family (sans plus its mono), swap, and an offline-safe stack."""
    page = client.get("/").text
    link = re.search(r'href="https://fonts\.googleapis\.com/css2\?[^"]*"', page).group(0)
    assert "family=Atkinson+Hyperlegible+Next" in link
    assert "family=Atkinson+Hyperlegible+Mono" in link
    assert "display=swap" in link
    assert page.count("fonts.googleapis.com/css2") == 1

    css = client.get("/static/css/app.css").text
    stack = re.search(r"--font-sans:(.*?);", css, re.S).group(1)
    assert "Atkinson Hyperlegible Next" in stack
    assert "system-ui" in stack and "sans-serif" in stack
    mono = re.search(r"--font-mono:(.*?);", css, re.S).group(1)
    assert "ui-monospace" in mono and "monospace" in mono
    # The numeric tables have to keep their columns lined up under the new face.
    assert "font-variant-numeric: tabular-nums" in css
    assert "fonts.googleapis.com" not in css


def _section_colour_var(css: str, key: str) -> str:
    rule = re.search(rf'\[data-section="{key}"\] \{{(.*?)\}}', css, re.S)
    assert rule is not None, key
    return re.search(r"--hue: var\(--([a-z-]+)\)", rule.group(1)).group(1)


def test_every_section_has_its_own_colour(client):
    """Each section has a hue that reaches the actions; within a workspace no two share one,
    except the workspace-general My data and Custom plot, which share neutral graphite."""
    css = client.get("/static/css/app.css").text
    for workspace in WORKSPACES:
        hues = [_section_colour_var(css, s.key) for s in SECTIONS if s.workspace == workspace.key]
        distinct = [h for h in hues if h != "graphite"]
        assert len(distinct) == len(set(distinct)), workspace.key
    assert _section_colour_var(css, "mydata") == _section_colour_var(css, "custom") == "graphite"
    body_rule = re.search(r"\nbody \{(.*?)\n\}", css, re.S).group(1)
    assert "background-image" not in body_rule


def test_dark_mode_follows_the_system_and_can_be_overridden(client):
    """The theme is set before the stylesheet paints, and every section has a dark variant."""
    page = client.get("/dynamics/isf").text
    head = page[: page.index("</head>")]
    assert head.index('localStorage.getItem("lada-theme")') < head.index("/static/css/app.css")
    assert "prefers-color-scheme: dark" in head
    assert "data-theme-toggle" in page

    css = client.get("/static/css/app.css").text
    dark = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1)
    for token in ("--ground", "--surface", "--ink", "--line", "--hue-ink"):
        assert f"{token}:" in dark, token
    for section in SECTIONS:
        hue = _section_colour_var(css, section.key)
        assert f"--{hue}-ink:" in dark, section.key
        assert f"--{hue}-tint:" in dark, section.key

    js = client.get("/static/js/app.js").text
    assert "ladaRepaintPlots" in js


def test_stylesheet_and_script_urls_change_when_the_files_do(client, tmp_path, monkeypatch):
    """Fingerprinted URLs, so no browser (Safari especially) can keep an outdated copy."""
    from lada_web import rendering

    page = client.get("/").text
    for path in ("css/app.css", "js/app.js"):
        url = re.search(rf'"(/static/{re.escape(path)}\?v=[0-9a-f]{{10}})"', page)
        assert url is not None, path
        assert client.get(url.group(1)).status_code == 200

    asset = tmp_path / "css" / "x.css"
    asset.parent.mkdir()
    asset.write_text("a { color: red; }")
    monkeypatch.setattr(rendering, "STATIC_DIR", tmp_path)
    before = rendering.static_url("css/x.css")
    asset.write_text("a { color: blue; }")
    os.utime(asset, ns=(asset.stat().st_atime_ns, asset.stat().st_mtime_ns + 1_000_000))
    after = rendering.static_url("css/x.css")
    assert before != after
    assert rendering.static_url("css/missing.css") == "/static/css/missing.css"


# --- Workspaces ----------------------------------------------------------------


def test_home_offers_one_door_per_workspace(client):
    r = client.get("/")
    assert r.status_code == 200
    doors = re.findall(r'<a class="door" href="([^"]+)" data-workspace="([^"]+)">', r.text)
    assert doors == [(w.home, w.key) for w in WORKSPACES]


@pytest.mark.parametrize("workspace", WORKSPACES, ids=lambda w: w.key)
def test_each_workspace_overview_lists_only_its_own_tools(client, workspace):
    r = client.get(workspace.home)
    assert r.status_code == 200
    main = re.search(r'<main id="main".*?</main>', r.text, re.S).group(0)
    for calc in CALCULATIONS:
        listed = f'href="{calc.path}"' in main
        assert listed == (SECTION_BY_KEY[calc.section].workspace == workspace.key), calc.path
    # The sidebar opens on this workspace and says so on the switch.
    assert f'aria-current="page">{workspace.title} overview</a>' in r.text
    pressed = re.search(r'data-workspace-switch="([a-z]+)"[^>]*aria-pressed="true"', r.text, re.S)
    assert pressed.group(1) == workspace.key


@pytest.mark.parametrize("calc", CALCULATIONS, ids=lambda c: f"{c.section}-{c.slug}")
def test_the_sidebar_opens_on_the_pages_own_workspace(client, calc):
    r = client.get(calc.path)
    own = SECTION_BY_KEY[calc.section].workspace
    active = re.search(r'<div class="nav-workspace is-active"\s+data-workspace="([a-z]+)"', r.text)
    assert active.group(1) == own
    switch = re.search(r'<div class="workspace-switch".*?</div>', r.text, re.S).group(0)
    assert [w.key for w in WORKSPACES] == re.findall(r'data-workspace-switch="([a-z]+)"', switch)


def test_rheometer_tools_live_under_one_prefix_and_simulation_urls_are_unchanged(client):
    for calc in RHEOMETER_CALCULATIONS:
        assert calc.path.startswith("/rheometer/"), calc.path
        assert client.get(calc.path).status_code == 200
    for calc in SIMULATION_CALCULATIONS:
        assert calc.path == f"/{calc.section}/{calc.slug}"
    assert client.get("/rheometer/nope").status_code == 404


def test_an_unbuilt_rheometer_tool_says_so(client):
    r = client.get("/rheometer/frequency-sweep")
    assert "This tool is not ready yet." in r.text
    assert 'href="/rheometer/my-data"' in r.text
