"""Regression tests for the figure and pgfplots export bugs found in the web-app audit."""

import io
import re
import shutil
import subprocess
import threading
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from lada_web.rheometry import figure as figure_module
from lada_web.rheometry.figure import (
    MAX_LEGEND_ENTRIES,
    MM_PER_INCH,
    FigureSpec,
    latex,
    pgfplots_zip,
    render,
    render_with_notes,
)

from .test_rheometry_export import sweep_spec

HAS_PDFLATEX = shutil.which("pdflatex") is not None


def _tex(spec_dict, tmp_path) -> str:
    data = pgfplots_zip(FigureSpec(**spec_dict), tmp_path / "work")
    return zipfile.ZipFile(io.BytesIO(data)).read("figure.tex").decode()


def _axes(tex: str) -> list[str]:
    """The option blocks of each axis, in order."""
    return re.findall(r"\\begin\{axis\}\[(.*?)\n\]", tex, re.S)


def _compile(spec_dict, tmp_path) -> subprocess.CompletedProcess:
    data = pgfplots_zip(FigureSpec(**spec_dict), tmp_path / "work")
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp_path / "out")
    (tmp_path / "out" / "main.tex").write_text(
        "\\documentclass{article}\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}"
        "\\begin{document}\\input{figure}\\end{document}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=tmp_path / "out",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )


def _option(block: str, key: str) -> str | None:
    match = re.search(rf"(?:^|,)\s*{re.escape(key)}=([^,\n]+)", block)
    return match.group(1).strip() if match else None


# --- B1: the right axis shares the bottom axis's x range ---------------------------------------


def _two_axis_spec(**x):
    xs = list(range(1, 101))
    return {
        "series": [
            {"name": "left", "x": xs[:50], "y": [float(v) for v in xs[:50]]},
            {"name": "right", "axis": "y2", "x": xs, "y": [float(v) for v in xs]},
        ],
        "x": {"title": "x", **x},
        "y2": {"title": "r"},
    }


@pytest.mark.parametrize("x", [{}, {"min": 10, "max": 30}, {"log": True}])
def test_b1_both_axes_have_the_same_x_range_and_mode(tmp_path, x):
    first, second = _axes(_tex(_two_axis_spec(**x), tmp_path))
    for key in ("xmode", "xmin", "xmax", "enlarge x limits"):
        assert _option(first, key) is not None, key
        assert _option(first, key) == _option(second, key), key
    # The automatic range covers the right-axis series too, not only the left one.
    if not x:
        assert float(_option(first, "xmax")) >= 100


# --- B3: legend=none draws no legend -----------------------------------------------------------


def test_b3_no_legend_means_no_legend_entries(tmp_path):
    tex = _tex(sweep_spec(legend="none"), tmp_path)
    assert "\\addlegendentry" not in tex
    assert "\\addlegendimage" not in tex


# --- B4: dollar signs in user text are never read as maths ------------------------------------


@pytest.mark.parametrize("legend", ["above", "outside", "inside-top-right"])
@pytest.mark.parametrize("name", ["$\\foo$", "a$^$b", "cost $5 and $6"])
def test_b4_dollar_signs_in_names_do_not_crash(legend, name):
    spec = sweep_spec(legend=legend)
    spec["series"][0]["name"] = name
    spec["x"]["title"] = name
    spec["y2"]["title"] = name
    for fmt in ("svg", "png"):
        render(FigureSpec(**spec), fmt, 100)


# --- B5: the cross marker is visible and the same shape everywhere ----------------------------


def _red_pixels(spec_dict) -> int:
    image = Image.open(io.BytesIO(render(FigureSpec(**spec_dict), "png", 150))).convert("RGB")
    a = np.asarray(image).astype(int)
    return int(((a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)).sum())


@pytest.mark.parametrize("filled", [True, False])
def test_b5_the_cross_marker_is_drawn(filled):
    def spec(marker):
        series = {"x": [1, 2, 3], "y": [1, 2, 3], "color": "#ff0000", "filled": filled}
        return {"series": [{**series, "marker": marker}], "marker_size": 16, "legend": "none"}

    circle = _red_pixels(spec("circle"))
    assert _red_pixels(spec("cross")) > 0.3 * circle


def test_b5_the_cross_is_a_plus_like_plotlys():
    # Plotly's "cross" is a plus sign; matplotlib's is "P", pgfplots' a declared plus.
    assert figure_module._MPL_MARKERS["cross"] == "P"
    filled, open_ = figure_module._PGF_MARKS["cross"]
    assert filled == "ladacross*" and open_ == "ladacross"


def test_b5_pgfplots_declares_the_cross_marks_it_uses(tmp_path):
    spec = {"series": [{"name": "c", "x": [1, 2], "y": [1, 2], "marker": "cross"}]}
    tex = _tex(spec, tmp_path)
    assert "mark=ladacross*" in tex
    assert "\\pgfdeclareplotmark{ladacross*}" in tex
    assert "\\pgfdeclareplotmark{ladacross}" in tex


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex is not installed")
def test_b5_the_cross_compiles(tmp_path):
    spec = {
        "series": [
            {"name": "c", "x": [1, 2], "y": [1, 2], "marker": "cross"},
            {"name": "o", "x": [1, 2], "y": [2, 1], "marker": "cross", "filled": False},
        ]
    }
    result = _compile(spec, tmp_path)
    assert result.returncode == 0, result.stdout[-2000:]


# --- B6: the outside legend clears the right axis ----------------------------------------------


def test_b6_outside_legend_is_placed_beyond_the_right_axis(tmp_path):
    tex = _tex(sweep_spec(legend="outside"), tmp_path)
    first, second = _axes(tex)
    # The right axis is drawn first and named, so the legend can sit past its labels.
    assert "name=ladaright" in first
    assert "at={(ladaright.outer north east)}" in second
    assert "outer north east" not in first


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex is not installed")
def test_b6_outside_legend_with_a_right_axis_compiles(tmp_path):
    result = _compile(sweep_spec(legend="outside"), tmp_path)
    assert result.returncode == 0, result.stdout[-2000:]


# --- B7: the above legend wraps --------------------------------------------------------------


def test_b7_a_long_above_legend_wraps_onto_rows(tmp_path):
    spec = sweep_spec()
    for i in range(8):
        spec["series"].insert(
            0, {"name": f"Sample batch {i}", "x": [1, 2], "y": [1e3, 2e3], "color": "#123456"}
        )
    tex = _tex(spec, tmp_path)
    columns = int(re.search(r"legend columns=(-?\d+)", tex).group(1))
    assert 2 <= columns < 13  # as the matplotlib figure wraps it at this size


def test_b7_a_short_above_legend_stays_on_one_row(tmp_path):
    spec = {"series": [{"name": n, "x": [1, 2], "y": [1, 2]} for n in "ab"]}
    tex = _tex(spec, tmp_path)
    assert "legend columns=2" in tex


# --- B8: unicode in names compiles -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x′", "x$'$"),
        ("x″", "x$''$"),
        ("μ", "$\\mu$"),
        ("α β Ω", "$\\alpha$ $\\beta$ $\\Omega$"),
        ("a − b", "a $-$ b"),
        ("T₀", "T$_0$"),
        ("x³", "x$^3$"),
        ("≈", "$\\approx$"),
    ],
)
def test_b8_unicode_becomes_latex(text, expected):
    assert latex(text) == expected


def test_b8_characters_latex_cannot_set_are_replaced():
    out = latex("日本 sample")
    assert out.isascii()
    assert out.endswith(" sample")


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex is not installed")
def test_b8_unicode_names_compile(tmp_path):
    text = "° µ μ γ̇ η ω ′ ″ · ⁻¹ ² ³ α β Ω ± × − – — ä é ₀ ≈ ≤ ∞ 日本 \u0307 ¤ ½"
    spec = sweep_spec()
    spec["series"][0]["name"] = text
    spec["x"]["title"] = text
    result = _compile(spec, tmp_path)
    assert result.returncode == 0, result.stdout[-2000:]


# --- B9: limits a log axis cannot take, and reversed limits ------------------------------------


def test_b9_non_positive_log_limits_are_dropped_with_a_note(tmp_path):
    spec = sweep_spec(x={"title": "w", "log": True, "min": 0, "max": 100})
    tex = _tex(spec, tmp_path)
    first = _axes(tex)[0]
    assert _option(first, "xmin") != "0"
    assert "% Note:" in tex
    _, notes = render_with_notes(FigureSpec(**spec), "svg")
    assert any("log" in n for n in notes)


def test_b9_reversed_limits_reverse_the_axis(tmp_path):
    spec = sweep_spec(x={"title": "w", "log": True, "min": 100, "max": 0.1})
    for block in _axes(_tex(spec, tmp_path)):
        assert float(_option(block, "xmin")) == pytest.approx(0.1)
        assert float(_option(block, "xmax")) == pytest.approx(100)
        assert "x dir=reverse" in block


def test_b9_reversed_y_limits_reverse_the_y_axis(tmp_path):
    spec = sweep_spec(y={"title": "G", "log": False, "min": 10, "max": 0})
    first = _axes(_tex(spec, tmp_path))[0]
    assert float(_option(first, "ymin")) == 0 and float(_option(first, "ymax")) == 10
    assert "y dir=reverse" in first


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex is not installed")
def test_b9_reversed_log_limits_compile_without_an_empty_axis(tmp_path):
    spec = sweep_spec(y2={"title": "t", "log": True, "min": 10, "max": 0.1})
    result = _compile(spec, tmp_path)
    assert result.returncode == 0, result.stdout[-2000:]
    log = (tmp_path / "out" / "main.log").read_text(errors="replace")
    assert "empty range" not in log
    assert "illegal" not in log


# --- B10: PNG pixel size ---------------------------------------------------------------------


@pytest.mark.parametrize(("w", "h", "dpi"), [(85, 65, 600), (85, 65, 300), (175, 100, 300)])
def test_b10_png_is_exactly_the_rounded_pixel_size(w, h, dpi):
    data = render(FigureSpec(**sweep_spec(width_mm=w, height_mm=h)), "png", dpi=dpi)
    size = Image.open(io.BytesIO(data)).size
    assert size == (round(w / MM_PER_INCH * dpi), round(h / MM_PER_INCH * dpi))


# --- E3: a legend with hundreds of entries -----------------------------------------------------


def _many(n):
    return {
        "series": [
            {"name": f"sample number {i} long", "x": [1, 2], "y": [1, 2], "color": "#123456"}
            for i in range(n)
        ]
    }


def test_e3_a_huge_legend_is_capped_and_fast():
    start = time.perf_counter()
    _, notes = render_with_notes(FigureSpec(**_many(300)), "svg")
    assert time.perf_counter() - start < 6
    assert any(str(MAX_LEGEND_ENTRIES) in n for n in notes)


def test_e3_the_pgfplots_legend_is_capped_the_same_way(tmp_path):
    tex = _tex(_many(100), tmp_path)
    assert tex.count("\\addlegendentry") == MAX_LEGEND_ENTRIES


def test_e3_a_busy_renderer_refuses_rather_than_queues(client, monkeypatch):
    monkeypatch.setattr(figure_module, "LOCK_TIMEOUT_S", 0.2)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with figure_module._STYLE_LOCK:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        held.wait(5)
        r = client.post("/api/rheometry/figure?fmt=svg", json=sweep_spec())
        assert r.status_code == 429
    finally:
        release.set()
        thread.join()
    assert client.post("/api/rheometry/figure?fmt=svg", json=sweep_spec()).status_code == 200
