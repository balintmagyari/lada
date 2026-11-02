"""Publication export: figures (SVG, PDF, PNG) and the pgfplots archive."""

import io
import re
import shutil
import subprocess
import zipfile
from xml.dom import minidom

import numpy as np
import pytest
from PIL import Image

from lada_web.rheometry.figure import MM_PER_INCH, FigureSpec, latex, pgfplots_zip, render

OMEGA = [0.1 * 10 ** (i / 4) for i in range(13)]


def maxwell(shift=1.0):
    x = [w * shift for w in OMEGA]
    return [1e5 * v * v / (1 + v * v) for v in x], [1e5 * v / (1 + v * v) for v in x]


def sweep_spec(**overrides):
    """Two temperatures, G′ filled, G″ open, tan δ on the right, a marker key: what the
    frequency-sweep studio sends."""
    series = []
    for name, color, shift in (("150 °C", "#2f6fd0", 1.0), ("170 °C", "#d1495b", 0.01)):
        g1, g2 = maxwell(shift)
        series += [
            {"name": name, "x": OMEGA, "y": g1, "color": color},
            {
                "name": name,
                "x": OMEGA,
                "y": g2,
                "color": color,
                "filled": False,
                "show_legend": False,
            },
            {
                "axis": "y2",
                "x": OMEGA,
                "y": [b / a for a, b in zip(g1, g2, strict=True)],
                "color": color,
                "marker": "triangle-up",
                "show_legend": False,
            },
        ]
    series += [
        {"name": "G′", "key": True},
        {"name": "G″", "key": True, "filled": False},
        {"name": "tan δ", "key": True, "axis": "y2", "marker": "triangle-up"},
    ]
    spec = {
        "series": series,
        "x": {"title": "ω (rad/s)", "log": True},
        "y": {"title": "G′, G″ (Pa)", "log": True},
        "y2": {"title": "tan δ", "log": True},
        "filename": "frequency-sweep",
    }
    spec.update(overrides)
    return spec


def pt(mm):
    return mm / MM_PER_INCH * 72


# --- Formats and sizes -----------------------------------------------------------------------


def test_svg_is_well_formed_and_exactly_the_requested_size():
    data = render(FigureSpec(**sweep_spec()), "svg")
    root = minidom.parseString(data).documentElement
    assert float(root.getAttribute("width").removesuffix("pt")) == pytest.approx(pt(85), abs=0.01)
    assert float(root.getAttribute("height").removesuffix("pt")) == pytest.approx(pt(65), abs=0.01)


def test_pdf_page_is_exactly_the_requested_size():
    data = render(FigureSpec(**sweep_spec(width_mm=175, height_mm=100)), "pdf")
    assert data.startswith(b"%PDF")
    box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert float(box.group(1)) == pytest.approx(pt(175), abs=0.01)
    assert float(box.group(2)) == pytest.approx(pt(100), abs=0.01)


@pytest.mark.parametrize("dpi", [300, 600])
def test_png_has_the_pixels_its_size_and_resolution_call_for(dpi):
    data = render(FigureSpec(**sweep_spec()), "png", dpi=dpi)
    width, height = Image.open(io.BytesIO(data)).size
    assert width == pytest.approx(85 / MM_PER_INCH * dpi, abs=1)
    assert height == pytest.approx(65 / MM_PER_INCH * dpi, abs=1)


def test_exports_are_deterministic():
    spec = FigureSpec(**sweep_spec())
    assert render(spec, "svg") == render(spec, "svg")
    assert render(spec, "pdf") == render(spec, "pdf")


def test_serif_figures_render():
    assert render(FigureSpec(**sweep_spec(font="serif")), "svg").startswith(b"<?xml")


# --- Routes ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "media", "magic"),
    [
        ("svg", "image/svg+xml", b"<?xml"),
        ("pdf", "application/pdf", b"%PDF"),
        ("png", "image/png", b"\x89PNG"),
    ],
)
def test_the_figure_route_downloads_each_format(client, fmt, media, magic):
    r = client.post(f"/api/rheometry/figure?fmt={fmt}&dpi=300", json=sweep_spec())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(media)
    assert r.headers["content-disposition"] == f'attachment; filename="frequency-sweep.{fmt}"'
    assert r.content.startswith(magic)


def test_the_preview_is_inline(client):
    r = client.post("/api/rheometry/figure?fmt=svg&preview=true", json=sweep_spec())
    assert r.headers["content-disposition"] == "inline"
    assert "x-figure-notes" not in r.headers


def test_a_crowding_legend_comes_back_with_advice(client):
    spec = sweep_spec()
    for s in spec["series"]:
        if s.get("name") and not s.get("key"):
            s["name"] = "a very long sample name that the legend has to carry " + s["name"]
    for i in range(6):
        spec["series"].insert(
            0,
            {
                "name": f"another long sample name, number {i}",
                "x": OMEGA,
                "y": maxwell()[0],
                "color": "#7a5cc9",
            },
        )
    r = client.post("/api/rheometry/figure?fmt=svg&preview=true", json=spec)
    assert r.status_code == 200
    assert "legend takes up much of this figure" in r.headers["x-figure-notes"]


def test_a_filename_cannot_escape_its_header(client):
    r = client.post("/api/rheometry/figure?fmt=svg", json=sweep_spec(filename='../evil"; x=y'))
    assert r.headers["content-disposition"] == 'attachment; filename="evil-x-y.svg"'


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"series": []}, "nothing to draw"),
        ({"series": [{"name": "a", "x": [1, 2], "y": [1]}]}, "different number"),
        ({"series": [{"name": "a", "x": [1], "y": [1], "color": "red"}]}, "#rrggbb"),
        ({"width_mm": 5}, "greater than or equal"),
    ],
)
def test_a_figure_that_cannot_be_drawn_says_why(client, change, message):
    r = client.post("/api/rheometry/figure?fmt=svg", json=sweep_spec(**change))
    assert r.status_code == 422
    assert message in r.text


def test_a_png_too_large_to_make_is_refused(client):
    r = client.post(
        "/api/rheometry/figure?fmt=png&dpi=1200", json=sweep_spec(width_mm=600, height_mm=600)
    )
    assert r.status_code == 422
    assert "million pixels" in r.json()["detail"]


# --- pgfplots --------------------------------------------------------------------------------


def _archive(spec_dict, tmp_path):
    return zipfile.ZipFile(io.BytesIO(pgfplots_zip(FigureSpec(**spec_dict), tmp_path)))


def test_the_archive_holds_the_source_and_one_table_per_series(tmp_path):
    names = _archive(sweep_spec(), tmp_path).namelist()
    assert names[0] == "figure.tex"
    assert sorted(names[1:]) == [f"data/series-{n:02d}.dat" for n in range(1, 7)]


def test_the_tables_carry_the_plotted_numbers(tmp_path):
    archive = _archive(sweep_spec(), tmp_path)
    table = np.loadtxt(io.BytesIO(archive.read("data/series-01.dat")), skiprows=1)
    np.testing.assert_allclose(table[:, 0], OMEGA, rtol=1e-9)
    np.testing.assert_allclose(table[:, 1], maxwell()[0], rtol=1e-9)


def test_points_a_log_axis_cannot_show_are_left_out(tmp_path):
    spec = sweep_spec(series=[{"name": "a", "x": [0.0, 1, 10, None], "y": [1, -2, 3, 4]}], y2=None)
    archive = _archive(spec, tmp_path)
    table = np.loadtxt(io.BytesIO(archive.read("data/series-01.dat")), skiprows=1, ndmin=2)
    assert table.tolist() == [[10.0, 3.0]]


def test_the_source_has_one_legend_with_the_key_and_typeset_symbols(tmp_path):
    tex = _archive(sweep_spec(), tmp_path).read("figure.tex").decode()
    assert tex.count("\\begin{axis}") == 2  # the right axis is its own overlay
    first_axis, second_axis = tex.split("\\end{axis}")[:2]
    assert first_axis.count("\\addlegendentry") == 5  # two samples, G′, G″, tan δ
    assert "\\addlegendentry{$G'$}" in first_axis
    assert "\\addlegendentry{$\\tan\\delta$}" in first_axis
    assert "xlabel={$\\omega$ (rad/s)}" in tex
    assert "\\addlegendentry{150 \\textdegree{}C}" in tex
    assert "\\addlegendentry" not in second_axis


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("γ (%)", "$\\gamma$ (\\%)"),
        ("sample_1 & 2", "sample\\_1 \\& 2"),
        ("|η*| (Pa·s)", "$|\\eta^*|$ (Pa$\\cdot$s)"),
        ("γ̇ (1/s)", "$\\dot\\gamma$ (1/s)"),
        ("G″ ~ $5", "$G''$ \\textasciitilde{} \\$5"),
    ],
)
def test_interface_text_becomes_safe_latex(text, expected):
    assert latex(text) == expected


def test_the_pgfplots_route_returns_a_zip(client):
    r = client.post("/api/rheometry/pgfplots", json=sweep_spec())
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "frequency-sweep-pgfplots.zip" in r.headers["content-disposition"]
    assert "figure.tex" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_the_exported_source_compiles(tmp_path):
    _archive(sweep_spec(), tmp_path / "work").extractall(tmp_path / "out")
    (tmp_path / "out" / "main.tex").write_text(
        "\\documentclass{article}\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}"
        "\\begin{document}\\input{figure}\\end{document}\n"
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=tmp_path / "out",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert (tmp_path / "out" / "main.pdf").exists()


# --- The studio page -------------------------------------------------------------------------


def test_the_studio_offers_a_figure_tab_and_every_export(client):
    r = client.get("/rheometer/frequency-sweep")
    assert 'id="tab-figure"' in r.text and 'id="studio-figure"' in r.text
    for kind in ("svg", "pdf", "png", "pgfplots"):
        assert f'data-export="{kind}"' in r.text
    assert re.search(r'<option value="600" selected>600 dpi</option>', r.text)
