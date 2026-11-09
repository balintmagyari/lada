"""Pages, navigation and the shared look: what every tool page has in common."""

import html
import os
import re

import pytest

from lada_web.catalog import CALCULATIONS, SECTION_BY_KEY, SECTIONS, WORKSPACE_BY_KEY, WORKSPACES

PARSE_URL = "/api/rheometry/parse"


def test_version_one_has_only_the_rheometer_workspace():
    assert [w.key for w in WORKSPACES] == ["rheometer"]
    assert all(s.workspace == "rheometer" for s in SECTIONS)


@pytest.mark.parametrize("path", ["/", "/rheometer"])
def test_the_overview_lists_every_tool_and_the_input_it_needs(client, path):
    r = client.get(path)
    assert r.status_code == 200
    for calc in CALCULATIONS:
        assert f'href="{calc.path}"' in r.text
        assert html.escape(calc.title) in r.text
    assert "Import" in r.text and "Export" in r.text
    assert "Uses the files you imported on My data" in r.text


def test_nothing_of_the_simulation_workspace_is_left(client):
    page = client.get("/").text
    for gone in ("/simulations", "/parsers/", "/dynamics/", "LAMMPS", ".npz"):
        assert gone not in page, gone
    assert "workspace-switch" not in page
    for path in ("/simulations", "/parsers/log", "/dynamics/isf", "/rheology/dynamic-moduli"):
        assert client.get(path).status_code == 404, path
    assert client.post("/api/parsers/log/table").status_code in (404, 405)


@pytest.mark.parametrize("calc", CALCULATIONS, ids=lambda c: c.slug)
def test_every_tool_page_renders(client, calc):
    r = client.get(calc.path)
    assert r.status_code == 200
    assert calc.path.startswith("/rheometer/")
    assert f"<h1>{html.escape(calc.title)}</h1>" in r.text


def test_unknown_page_is_404_json_and_html(client):
    r = client.get("/rheometer/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "There is no page at this address."
    r = client.get("/nope/nope", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "There is no page at this address." in r.text


BACK_TO_MY_DATA = '<a href="/rheometer/my-data">Back to Import files</a>'


def failing_import(client, referer: str | None):
    """An import with no file, sent the way a browser form would send it."""
    headers = {"Accept": "text/html"}
    if referer is not None:
        headers["Referer"] = referer
    return client.post(PARSE_URL, headers=headers)


def test_the_error_page_offers_a_way_back_to_the_page_you_came_from(client):
    r = failing_import(client, "http://testserver/rheometer/my-data")
    assert r.status_code == 422
    assert BACK_TO_MY_DATA in r.text
    assert "Back to the tool list" in r.text


@pytest.mark.parametrize(
    "referer",
    [
        None,
        "",
        "http://evil.example/rheometer/my-data",
        "https://testserver/rheometer/my-data",
        "http://testserver/rheometer/nope",
        "http://testserver/",
        "javascript:alert(1)//rheometer/my-data",
        "http://[abc/rheometer/my-data",
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
def test_the_error_page_falls_back_when_the_referer_is_not_one_of_our_pages(client, referer):
    """A malformed referer (e.g. an unclosed IPv6 bracket) must not turn the 422 into a 500."""
    r = failing_import(client, referer)
    assert r.status_code == 422
    assert BACK_TO_MY_DATA not in r.text
    assert "Back to the tool list" in r.text


def test_the_json_answer_is_unchanged_by_the_back_link(client):
    r = client.post(PARSE_URL, headers={"Referer": "http://testserver/rheometer/my-data"})
    assert r.status_code == 422 and "detail" in r.json()


def test_static_assets_and_api_docs_are_served(client):
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/api/docs").status_code == 200
    paths = set(client.get("/api/openapi.json").json()["paths"])
    assert paths == {
        "/api/rheometry/parse",
        "/api/rheometry/quantities",
        "/api/rheometry/rederive",
        "/api/rheometry/figure",
        "/api/rheometry/pgfplots",
        "/api/rheometry/tts",
    }


def test_settings_are_injected(app):
    assert app.state.settings.max_upload_mb == 1


def test_no_cdn_script_is_loaded_that_the_pages_do_not_use(client):
    """htmx and Alpine drove the simulation forms; the rheometer pages need only Plotly."""
    page = client.get("/rheometer/frequency-sweep").text
    assert "htmx.org" not in page and "alpinejs" not in page
    assert "plotly" in page


# --- Navigation and page layout --------------------------------------------------------------


def test_the_sidebar_lists_every_section_open(client):
    """No menus to open: every section and tool is in the page, in catalog order, always."""
    r = client.get("/rheometer/creep")
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


def test_the_brand_chain_has_one_bead_per_section(client):
    page = client.get("/").text
    chain = re.search(r'<svg class="brand-chain".*?</svg>', page, re.S).group(0)
    assert re.findall(r'class="bead-([a-z]+)"', chain) == [s.key for s in SECTIONS]
    css = client.get("/static/css/app.css").text
    for section in SECTIONS:
        assert f".bead-{section.key} {{ fill:" in css, section.key


@pytest.mark.parametrize("calc", CALCULATIONS, ids=lambda c: c.slug)
def test_the_navigation_names_the_section_the_page_and_the_way_back(client, calc):
    r = client.get(calc.path)
    section = SECTION_BY_KEY[calc.section]
    workspace = WORKSPACE_BY_KEY[section.workspace]
    section_title = html.escape(section.title)

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
    r = client.get("/rheometer/master-curve")
    assert 'id="jump"' in r.text and 'id="jump-input"' in r.text
    assert "data-jump-open" in r.text
    js = client.get("/static/js/app.js").text
    assert "collectTools" in js and "showModal" in js
    assert '"lada-recent"' in js


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
    assert "fonts.googleapis.com" not in css


def _section_colour_var(css: str, key: str) -> str:
    rule = re.search(rf'\[data-section="{key}"\] \{{(.*?)\}}', css, re.S)
    assert rule is not None, key
    return re.search(r"--hue: var\(--([a-z-]+)\)", rule.group(1)).group(1)


def test_every_section_has_its_own_colour(client):
    """No two sections share a hue, except the general My data and Custom plot, which share
    neutral graphite."""
    css = client.get("/static/css/app.css").text
    hues = [_section_colour_var(css, s.key) for s in SECTIONS]
    distinct = [h for h in hues if h != "graphite"]
    assert len(distinct) == len(set(distinct))
    assert _section_colour_var(css, "mydata") == _section_colour_var(css, "custom") == "graphite"


def test_dark_mode_follows_the_system_and_can_be_overridden(client):
    """The theme is set before the stylesheet paints, and every section has a dark variant."""
    page = client.get("/rheometer/flow-curve").text
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

    # The studio redraws its plots when the theme changes; app.js is what calls it.
    assert "ladaRepaintPlots" in client.get("/static/js/app.js").text


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


# --- HEAD and 405 ------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/rheometer", "/rheometer/master-curve"])
def test_pages_answer_head(client, path):
    r = client.head(path)
    assert r.status_code == 200
    assert r.content == b""


def test_unknown_page_is_still_404_for_head(client):
    assert client.head("/no/such").status_code == 404


def test_405_keeps_its_allow_header(client):
    r = client.get("/api/rheometry/tts")
    assert r.status_code == 405
    assert r.headers["allow"] == "POST"


def test_405_keeps_its_allow_header_on_the_html_error_page(client):
    r = client.put("/", headers={"accept": "text/html"})
    assert r.status_code == 405
    assert set(r.headers["allow"].replace(" ", "").split(",")) == {"GET", "HEAD"}
