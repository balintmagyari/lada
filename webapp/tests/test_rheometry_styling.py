"""Per-series styling, one-sided axis limits and a page's own choice of axes.

The graph studio sends the exported figure everything the user set: a marker size per series,
a colour for a marker key when its quantity has one, and limits where only one end is given.
"""

import io
import re
import zipfile

from lada_web.rheometry.figure import AxisSpec, FigureSpec, _limits, pgfplots_zip, render_with_notes

from .test_rheometry_export import sweep_spec


def _tex(spec_dict, tmp_path) -> str:
    data = pgfplots_zip(FigureSpec(**spec_dict), tmp_path / "work")
    return zipfile.ZipFile(io.BytesIO(data)).read("figure.tex").decode()


def test_one_end_of_an_axis_can_be_fixed_and_the_other_fits_the_data():
    assert _limits(AxisSpec(min=0)) == (0, None, False)
    assert _limits(AxisSpec(max=5)) == (None, 5, False)
    assert _limits(AxisSpec(min=2, max=1)) == (1, 2, True)
    assert _limits(AxisSpec()) is None
    # A log axis drops only the end it cannot take.
    assert _limits(AxisSpec(log=True, min=-1, max=100)) == (None, 100, False)
    assert _limits(AxisSpec(log=True, min=-1)) is None


def test_a_lower_limit_alone_reaches_both_renderers(tmp_path):
    spec = sweep_spec(y2={"title": "tan δ", "log": False, "min": 0})
    tex = _tex(spec, tmp_path)
    right = re.findall(r"\\begin\{axis\}\[(.*?)\n\]", tex, re.S)
    right_axis = next(block for block in right if "axis y line*=right" in block)
    assert "ymin=0" in right_axis and "ymax=" not in right_axis
    content, notes = render_with_notes(FigureSpec(**spec), "svg")
    assert content.startswith(b"<?xml") and notes == []


def test_a_dropped_log_limit_is_explained():
    spec = sweep_spec(x={"title": "ω", "log": True, "min": 0, "max": 100})
    _, notes = render_with_notes(FigureSpec(**spec), "svg")
    assert any("0 was ignored" in note for note in notes)


def test_each_series_can_have_its_own_marker_size(tmp_path):
    spec = sweep_spec()
    spec["series"][0]["marker_size"] = 14
    tex = _tex(spec, tmp_path)
    sizes = re.findall(r"mark size=([\d.]+)pt", tex)
    assert len(set(sizes)) == 2  # the one chosen, and the figure's default for the rest
    render_with_notes(FigureSpec(**spec), "pdf")


def test_a_marker_key_keeps_the_colour_its_quantity_was_given(tmp_path):
    spec = sweep_spec()
    key = next(s for s in spec["series"] if s.get("key"))
    key["color"] = "#cc0000"
    tex = _tex(spec, tmp_path)
    assert "cc0000" in tex.lower()
    # Without a colour, a key is grey as before.
    assert "6b6b6b" in _tex(sweep_spec(), tmp_path / "plain").lower()


def test_every_graph_page_offers_the_new_controls(client):
    js = client.get("/static/js/rheometry/studio.js").text
    for part in (
        '"Choose every axis yourself"',
        '"Logarithmic"',
        '"Choose colour"',
        '"Same for every sample"',
        'group("Series")',
        "markerPreview",
        "seriesPaint",
        "seedOverride",
        "limitHint",
    ):
        assert part in js
    presets = client.get("/static/js/rheometry/presets.js").text
    assert "override:" in presets and "styles:" in presets and "SHAPES" in presets


# --- Figure size and legend place ------------------------------------------------------------

LEGEND_PLACES = [
    "above",
    "below",
    "inside-top-left",
    "inside-top-center",
    "inside-top-right",
    "inside-center-left",
    "inside-center",
    "inside-center-right",
    "inside-bottom-left",
    "inside-bottom-center",
    "inside-bottom-right",
    "outside",
    "custom",
    "none",
]


import pytest  # noqa: E402


@pytest.mark.parametrize("place", LEGEND_PLACES)
def test_every_legend_place_draws_in_both_renderers(place, tmp_path):
    spec = sweep_spec(legend=place, legend_x=0.3, legend_y=0.6)
    content, _ = render_with_notes(FigureSpec(**spec), "svg")
    assert content.startswith(b"<?xml")
    tex = _tex(spec, tmp_path)
    if place == "none":
        assert "legend style" not in tex and "legend pos" not in tex
    elif place == "custom":
        assert "at={(0.300,0.600)},anchor=north west" in tex
    elif place == "below":
        assert re.search(r"at=\{\(0\.5,-0\.\d+\)\},anchor=north", tex)


def test_a_custom_size_is_exported_exactly(tmp_path):
    from PIL import Image

    spec = sweep_spec(width_mm=123.5, height_mm=47)
    png, _ = render_with_notes(FigureSpec(**spec), "png", dpi=300)
    size = Image.open(io.BytesIO(png)).size
    assert size == (round(123.5 / 25.4 * 300), round(47 / 25.4 * 300))
    assert "width=123.5mm" in _tex(spec, tmp_path) and "height=47.0mm" in _tex(spec, tmp_path)


def test_sizes_outside_the_range_are_refused(client):
    spec = sweep_spec(width_mm=10)
    assert client.post("/api/rheometry/figure?fmt=svg", json=spec).status_code == 422


def test_the_studio_offers_custom_size_and_every_legend_place(client):
    js = client.get("/static/js/rheometry/studio.js").text
    for place in LEGEND_PLACES:
        assert f'"{place}"' in js, place
    assert "edits: { legendPosition: true }" in js and "legendDropped" in js
    assert '"Width, mm"' in js and '"Height, mm"' in js
    presets = client.get("/static/js/rheometry/presets.js").text
    assert 'name: "Custom size"' in presets


def test_my_data_can_show_each_table_as_a_grid(client):
    js = client.get("/static/js/rheometry/mydata.js").text
    for part in ('"View data"', "dataTable", "Download as CSV", "TABLE_STEP"):
        assert part in js
