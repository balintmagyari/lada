"""Publication figures from the graph studio's figure spec.

The browser sends what it draws: every series already in display units, its colour, marker
and axis, and the figure's size and style. This module renders exactly that with matplotlib
(SVG, PDF or PNG) and writes the same figure as a pgfplots source for LaTeX. The browser's
"Figure" preview asks for the SVG, so what is previewed is what is exported.

Exports are always in print style, black on white, whatever theme the site is shown in.
"""

import io
import math
import re
import threading
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..errors import ServerBusy

MM_PER_INCH = 25.4
MAX_POINTS = 500_000
MAX_PIXELS = 80_000_000
# A legend longer than this is cut short (with a note): beyond it a legend is unreadable, and
# laying it out costs time while every other figure export waits for the style lock.
MAX_LEGEND_ENTRIES = 30
# How long a figure request waits for another one to finish before it is refused as busy.
LOCK_TIMEOUT_S = 8.0

Marker = Literal["circle", "square", "triangle-up", "triangle-down", "diamond", "cross"]


class AxisSpec(BaseModel):
    title: str = Field("", max_length=200)
    log: bool = False
    min: float | None = None
    max: float | None = None


class SeriesSpec(BaseModel):
    name: str = Field("", max_length=200)
    axis: Literal["y", "y2"] = "y"
    x: list[float | None] = Field(default_factory=list)
    y: list[float | None] = Field(default_factory=list)
    # None: black, or grey for a key entry.
    color: str | None = None
    marker: Marker = "circle"
    filled: bool = True
    # Overrides FigureSpec.marker_size for this series when set.
    marker_size: float | None = Field(None, ge=1, le=30)
    show_legend: bool = True
    # A key entry explains a marker; it has no data and is drawn in grey.
    key: bool = False

    @field_validator("color")
    @classmethod
    def _hex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError("A series colour must be written #rrggbb.")
        return value.lower()


class FigureSpec(BaseModel):
    width_mm: float = Field(85, ge=20, le=600)
    height_mm: float = Field(65, ge=20, le=600)
    font: Literal["sans", "serif"] = "sans"
    font_size: float = Field(9, ge=5, le=30)
    mode: Literal["markers", "both", "lines"] = "markers"
    marker_size: float = Field(7, ge=1, le=30)
    line_width: float = Field(1.5, ge=0.1, le=10)
    legend: Literal[
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
    ] = "above"
    # Where a "custom" legend's top left corner sits, as fractions of the plot area (0, 0 its
    # bottom left, 1, 1 its top right): where it was dragged to in the browser.
    legend_x: float = Field(0.02, ge=-1, le=2)
    legend_y: float = Field(0.98, ge=-1, le=2)
    grid: bool = False
    x: AxisSpec = Field(default_factory=AxisSpec)
    y: AxisSpec = Field(default_factory=AxisSpec)
    y2: AxisSpec | None = None
    series: list[SeriesSpec] = Field(default_factory=list, max_length=300)
    filename: str = Field("figure", max_length=80)

    @field_validator("filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.") or "figure"


def _check(spec: FigureSpec) -> None:
    total = sum(len(s.x) for s in spec.series)
    if total > MAX_POINTS:
        raise ValueError(
            f"This figure has {total:,} points, more than the {MAX_POINTS:,} it can be drawn with."
        )
    for s in spec.series:
        if len(s.x) != len(s.y):
            raise ValueError(f"The series “{s.name}” has a different number of x and y values.")
    if not any(s.x for s in spec.series if not s.key):
        raise ValueError("There is nothing to draw: no series has any points.")


def _usable(value: float | None, log: bool) -> float | None:
    if value is None or not math.isfinite(value) or (log and value <= 0):
        return None
    return value


def _limits(axis: AxisSpec) -> tuple[float | None, float | None, bool] | None:
    """The axis' requested limits as (low, high, reversed), or None to fit the data. Either end
    may be None: that end then fits the data. A limit a log axis cannot take (zero or negative)
    is dropped; a minimum above the maximum reverses the axis, as it does in the browser."""
    lo, hi = _usable(axis.min, axis.log), _usable(axis.max, axis.log)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        if lo == hi:
            return None
        return min(lo, hi), max(lo, hi), lo > hi
    return lo, hi, False


def _limit_notes(spec: FigureSpec) -> list[str]:
    notes = []
    axes = [("x", spec.x), ("left y", spec.y)] + ([("right y", spec.y2)] if spec.y2 else [])
    for name, axis in axes:
        dropped = [v for v in (axis.min, axis.max) if v is not None and v <= 0]
        if axis.log and dropped:
            notes.append(
                f"The {name} axis is logarithmic, so its limits must be positive; "
                f"{_fmt(dropped[0])} was ignored and that end fits the data."
            )
    return notes


_KEY_GREY = "#6b6b6b"


def _paint(series: SeriesSpec) -> str:
    """A series' colour: its own, else grey for a marker key and black for data."""
    return series.color or (_KEY_GREY if series.key else "#000000")


def _legend_cap_note(total: int) -> str:
    return (
        f"The legend shows the first {MAX_LEGEND_ENTRIES} of its {total} entries; "
        "a longer legend cannot be read."
    )


def _points(series: SeriesSpec, x_log: bool, y_log: bool) -> tuple[list[float], list[float]]:
    """The series' points that can be drawn: both values present and finite, and positive on
    a log axis."""
    xs, ys = [], []
    for x, y in zip(series.x, series.y, strict=True):
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            continue
        if (x_log and x <= 0) or (y_log and y <= 0):
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


# --- matplotlib ----------------------------------------------------------------------------

_MPL_MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle-up": "^",
    "triangle-down": "v",
    "diamond": "D",
    # Plotly's "cross" is a plus sign with thick arms: matplotlib's filled plus.
    "cross": "P",
}
# Plotly sizes markers and lines in screen pixels; a printed figure is measured in points,
# and on paper the same marker reads larger than on screen, so it is drawn a little smaller.
_PX_TO_PT = 0.6

_LEGEND_PLACES = {
    "inside-top-left": {"loc": "upper left"},
    "inside-top-center": {"loc": "upper center"},
    "inside-top-right": {"loc": "upper right"},
    "inside-center-left": {"loc": "center left"},
    "inside-center": {"loc": "center"},
    "inside-center-right": {"loc": "center right"},
    "inside-bottom-left": {"loc": "lower left"},
    "inside-bottom-center": {"loc": "lower center"},
    "inside-bottom-right": {"loc": "lower right"},
}


def _rc(spec: FigureSpec) -> dict:
    serif = spec.font == "serif"
    return {
        "font.family": "serif" if serif else "sans-serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stix" if serif else "dejavusans",
        "font.size": spec.font_size,
        "axes.linewidth": 0.8,
        "axes.labelpad": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": spec.font_size * 0.92,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.4,
        "svg.fonttype": "path",
        # A fixed salt for the ids inside an SVG: the same figure gives the same bytes, so an
        # exported figure can be kept under version control without spurious changes.
        "svg.hashsalt": "lada-web",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _style_axis(ax, axis: AxisSpec, which: str, grid: bool) -> None:
    from matplotlib.ticker import (
        FuncFormatter,
        LogLocator,
        MaxNLocator,
        NullFormatter,
        NullLocator,
        ScalarFormatter,
    )

    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    set_scale("log" if axis.log else "linear")
    set_label(axis.title, parse_math=False)
    limits = _limits(axis)
    if limits is not None:
        lo, hi, reverse = limits
        set_lim((hi, lo) if reverse else (lo, hi))
    target = ax.xaxis if which == "x" else ax.yaxis
    if axis.log:
        lo_shown, hi_shown = sorted(ax.get_xlim() if which == "x" else ax.get_ylim())
        decades = math.log10(hi_shown / lo_shown) if lo_shown > 0 else 99
        short = decades < 1.5
        target.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
        target.set_minor_formatter(NullFormatter())
        if decades < 0.3:
            # A sliver of a decade (a near-constant tan δ in a time sweep): no digit of the
            # decade may fall inside it, so label it evenly instead.
            target.set_major_locator(MaxNLocator(nbins=4))
            target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
            target.set_minor_locator(NullLocator())
        elif short:
            # Under a decade and a half, powers of ten alone leave one or two labels and
            # matplotlib falls back to "3 × 10⁰"; 1, 2 and 5 of each decade read better.
            subs = tuple(float(n) for n in range(1, 10)) if decades < 0.8 else (1.0, 2.0, 5.0)
            target.set_major_locator(LogLocator(base=10, subs=subs, numticks=100))
            target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            target.set_major_locator(LogLocator(base=10, numticks=100))
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        target.set_major_formatter(formatter)
    if grid:
        ax.grid(True, which="major", axis=which, color="#d9d9d9", linewidth=0.5)


def _row_major(items: list, ncols: int) -> list:
    """Reorder legend items so they read along rows: matplotlib fills columns top to bottom."""
    return [items[i] for c in range(ncols) for i in range(c, len(items), ncols)]


def _make_legend(target, handles, labels, **kwargs):
    """A legend whose texts are plain text: a "$" in a name is never read as maths, neither
    when the legend is measured nor when it is drawn."""
    legend = target.legend(handles, labels, **kwargs)
    for text in legend.get_texts():
        text.set_parse_math(False)
    return legend


def _legend_above(fig, ax, handles, labels, below: bool = False):
    """A legend above the plot (or below it, under the axis title), wrapped onto as many rows
    as it needs to fit the axes' width, so a long legend never squeezes the plot to make room
    for itself. The widest column count that fits is found by bisection, so only a handful of
    trial layouts are measured."""
    renderer = fig.canvas.get_renderer()
    axes_width = ax.get_window_extent(renderer).width
    legend = None

    def attempt(ncols: int) -> bool:
        nonlocal legend
        if legend is not None:
            legend.remove()
        if below:
            # A figure legend in its own band under everything, so constrained layout makes
            # room for it below the axis title instead of drawing over it.
            place = {"loc": "outside lower center"}
        else:
            place = {"loc": "lower left", "bbox_to_anchor": (0, 1.01), "borderaxespad": 0.2}
        legend = _make_legend(
            fig if below else ax,
            _row_major(handles, ncols),
            _row_major(labels, ncols),
            ncols=ncols,
            columnspacing=1.0,
            **place,
        )
        return legend.get_window_extent(renderer).width <= axes_width * 1.02

    low, high = 1, len(labels)  # the answer lies in [low, high]; one column always "fits"
    while low < high:
        middle = (low + high + 1) // 2
        if attempt(middle):
            low = middle
        else:
            high = middle - 1
    # A legend's width is not strictly monotonic in its column count; settle on one that fits.
    ncols = low
    while not attempt(ncols) and ncols > 1:
        ncols -= 1
    if legend is not None and legend.get_window_extent(renderer).height > fig.bbox.height / 3:
        fig._lada_notes.append(
            "The legend takes up much of this figure. Shorter legend names, a legend inside "
            "the plot, or the double-column size would leave more room for the data."
        )
    return legend


def _legend(fig, ax, ax2, handles, labels, spec: FigureSpec) -> None:
    if spec.legend in ("above", "below"):
        _legend_above(fig, ax, handles, labels, below=spec.legend == "below")
    elif spec.legend == "custom":
        _make_legend(
            ax2 or ax,
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(spec.legend_x, spec.legend_y),
            borderaxespad=0,
        )
    elif spec.legend == "outside":
        _make_legend(
            ax, handles, labels, loc="upper left", bbox_to_anchor=(1.18 if ax2 else 1.02, 1)
        )
    else:
        # Drawn on the top axes so the right-axis series cannot cover it.
        _make_legend(ax2 or ax, handles, labels, **_LEGEND_PLACES[spec.legend])


def _draw(spec: FigureSpec, dpi: float | None = None):
    """The matplotlib Figure for a spec. Built with the object API, never pyplot. Must run
    inside `matplotlib.rc_context(_rc(spec))`, held open until the figure is saved: tick marks
    are created when the figure is drawn, from the style settings in force at that moment."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _check(spec)
    # Constrained layout fits labels and legend inside the requested size, so a figure asked
    # for at 85 mm comes out exactly 85 mm wide, as a journal column needs.
    width, height = spec.width_mm / MM_PER_INCH, spec.height_mm / MM_PER_INCH
    if dpi is not None:
        # A raster is cut to whole pixels by truncation: size it to the nearest whole pixel
        # (plus a fraction, against rounding error) so it is not one pixel short.
        width = (round(width * dpi) + 0.25) / dpi
        height = (round(height * dpi) + 0.25) / dpi
    fig = Figure(figsize=(width, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    FigureCanvasAgg(fig)  # a renderer, so the legend can be measured while it is laid out
    fig._lada_notes = _limit_notes(spec)  # advice about the figure, passed back to the studio
    ax = fig.add_subplot()
    has_y2 = spec.y2 is not None and any(s.axis == "y2" for s in spec.series)
    ax2 = ax.twinx() if has_y2 else None

    handles, labels = [], []
    for series in spec.series:
        target = ax2 if series.axis == "y2" and ax2 is not None else ax
        axis_spec = spec.y2 if target is ax2 else spec.y
        color = _paint(series)
        marker = _MPL_MARKERS[series.marker]
        show_markers = series.key or spec.mode in ("markers", "both")
        show_line = not series.key and spec.mode in ("lines", "both")
        kwargs = {
            "color": color,
            "marker": marker if show_markers else "None",
            "markersize": (series.marker_size or spec.marker_size) * _PX_TO_PT,
            "markerfacecolor": color if series.filled else "none",
            "markeredgecolor": color,
            "markeredgewidth": 0 if series.filled else 1.0,
            "linestyle": (":" if series.axis == "y2" else "-") if show_line else "None",
            "linewidth": spec.line_width * _PX_TO_PT,
        }
        if series.key:
            (line,) = target.plot([], [], **kwargs)
        else:
            xs, ys = _points(series, spec.x.log, axis_spec.log)
            (line,) = target.plot(xs, ys, **kwargs)
        if series.show_legend and series.name:
            handles.append(line)
            labels.append(series.name)

    _style_axis(ax, spec.x, "x", spec.grid)
    _style_axis(ax, spec.y, "y", spec.grid)
    if ax2 is not None:
        _style_axis(ax2, spec.y2, "y", False)
        ax.tick_params(top=True, right=False, which="both")
    else:
        ax.tick_params(top=True, right=True, which="both")

    if spec.legend != "none" and handles:
        if len(labels) > MAX_LEGEND_ENTRIES:
            fig._lada_notes.append(_legend_cap_note(len(labels)))
            handles, labels = handles[:MAX_LEGEND_ENTRIES], labels[:MAX_LEGEND_ENTRIES]
        _legend(fig, ax, ax2, handles, labels, spec)
    return fig


MEDIA_TYPES = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}

# matplotlib's style settings are process-wide, and rc_context changes them for the whole
# process while it is open; one render at a time keeps two requests on different worker
# threads from drawing with each other's fonts and sizes.
_STYLE_LOCK = threading.Lock()


class FigureBusy(ServerBusy):
    """Another figure is being drawn and did not finish in time; answered with 429."""

    def __init__(self) -> None:
        super().__init__("Another figure is being drawn. Try again in a moment.")


def render(spec: FigureSpec, fmt: Literal["svg", "pdf", "png"], dpi: int = 600) -> bytes:
    """The figure as SVG, PDF or PNG bytes."""
    return render_with_notes(spec, fmt, dpi)[0]


def render_with_notes(
    spec: FigureSpec, fmt: Literal["svg", "pdf", "png"], dpi: int = 600
) -> tuple[bytes, list[str]]:
    """The figure's bytes, and any advice about its layout (a legend crowding the plot)."""
    if fmt == "png":
        pixels = (spec.width_mm / MM_PER_INCH * dpi) * (spec.height_mm / MM_PER_INCH * dpi)
        if pixels > MAX_PIXELS:
            raise ValueError(
                f"A PNG this size at {dpi} dpi would have {pixels / 1e6:.0f} million pixels; "
                "choose a lower resolution or a smaller figure."
            )
    import matplotlib

    buffer = io.BytesIO()
    metadata = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}[fmt]
    # Wait a few seconds at most: a request queued behind a long render would otherwise pin a
    # worker thread; refused, it can be retried.
    if not _STYLE_LOCK.acquire(timeout=LOCK_TIMEOUT_S):
        raise FigureBusy()
    try:
        with matplotlib.rc_context(_rc(spec)):
            fig = _draw(spec, dpi if fmt == "png" else None)
            fig.savefig(buffer, format=fmt, dpi=dpi, metadata=metadata)
    finally:
        _STYLE_LOCK.release()
    return buffer.getvalue(), list(fig._lada_notes)


# --- pgfplots ------------------------------------------------------------------------------

_PGF_MARKS = {
    "circle": ("*", "o"),
    "square": ("square*", "square"),
    "triangle-up": ("triangle*", "triangle"),
    "triangle-down": (
        "triangle*,mark options={rotate=180}",
        "triangle,mark options={rotate=180}",
    ),
    "diamond": ("diamond*", "diamond"),
    # Plotly's "cross" is a plus with thick arms; pgfplots has none, so figure.tex declares it.
    "cross": ("ladacross*", "ladacross"),
}
# A plus sign with arms a third as wide as they are long, filled or drawn in outline.
_PGF_CROSS = (
    "\\pgfpathmoveto{\\pgfqpoint{0.333\\pgfplotmarksize}{\\pgfplotmarksize}}"
    + "".join(
        f"\\pgfpathlineto{{\\pgfqpoint{{{x}\\pgfplotmarksize}}{{{y}\\pgfplotmarksize}}}}"
        for x, y in (
            (-0.333, 1),
            (-0.333, 0.333),
            (-1, 0.333),
            (-1, -0.333),
            (-0.333, -0.333),
            (-0.333, -1),
            (0.333, -1),
            (0.333, -0.333),
            (1, -0.333),
            (1, 0.333),
            (0.333, 0.333),
        )
    )
    + "\\pgfpathclose"
)
_PGF_CROSS_MARKS = [
    f"\\pgfdeclareplotmark{{ladacross*}}{{{_PGF_CROSS}\\pgfusepathqfillstroke}}",
    f"\\pgfdeclareplotmark{{ladacross}}{{{_PGF_CROSS}\\pgfusepathqstroke}}",
]
_PGF_LEGEND = {
    "inside-top-left": "legend pos=north west",
    "inside-top-center": "legend style={at={(0.5,0.97)},anchor=north}",
    "inside-top-right": "legend pos=north east",
    "inside-center-left": "legend style={at={(0.03,0.5)},anchor=west}",
    "inside-center": "legend style={at={(0.5,0.5)},anchor=center}",
    "inside-center-right": "legend style={at={(0.97,0.5)},anchor=east}",
    "inside-bottom-left": "legend pos=south west",
    "inside-bottom-center": "legend style={at={(0.5,0.03)},anchor=south}",
    "inside-bottom-right": "legend pos=south east",
    "outside": "legend pos=outer north east",
}
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_SYMBOLS = [
    ("aT·", r"$a_T$"),
    ("|η*|", r"$|\eta^*|$"),
    ("|G*|", r"$|G^*|$"),
    ("G′", r"$G'$"),
    ("G″", r"$G''$"),
    ("tan δ", r"$\tan\delta$"),
    ("γ̇", r"$\dot\gamma$"),
    ("°C", r"\textdegree{}C"),
    ("°F", r"\textdegree{}F"),
    ("°", r"\textdegree{}"),
    ("·", r"$\cdot$"),
    ("µ", r"$\mu$"),
    ("′", r"$'$"),
    ("″", r"$''$"),
    ("‴", r"$'''$"),
    ("−", r"$-$"),
    ("–", "--"),
    ("—", "---"),
    ("±", r"$\pm$"),
    ("×", r"$\times$"),
    ("÷", r"$\div$"),
    ("≈", r"$\approx$"),
    ("≠", r"$\neq$"),
    ("≤", r"$\leq$"),
    ("≥", r"$\geq$"),
    ("∞", r"$\infty$"),
    ("∝", r"$\propto$"),
    ("∂", r"$\partial$"),
    ("∆", r"$\Delta$"),
    ("→", r"$\rightarrow$"),
    ("←", r"$\leftarrow$"),
    ("…", r"\ldots{}"),
    ("‘", "`"),
    ("’", "'"),
    ("“", "``"),
    ("”", "''"),
    ("Å", r"\AA{}"),
]
_LATEX_SYMBOLS.append(("Ω", r"$\Omega$"))  # the ohm sign
# The Greek alphabet, U+03B1 to U+03C9 and U+0391 to U+03A9, in LaTeX. Capitals that look
# like Latin letters have no command of their own and are set as those letters.
_GREEK_SMALL = re.findall(
    r"\S+",
    r"\alpha \beta \gamma \delta \epsilon \zeta \eta \theta \iota \kappa \lambda \mu \nu \xi o "
    r"\pi \rho \varsigma \sigma \tau \upsilon \phi \chi \psi \omega",
)
_GREEK_CAPITAL = re.findall(
    r"\S+",
    r"A B \Gamma \Delta E Z H \Theta I K \Lambda M N \Xi O \Pi P - \Sigma T \Upsilon \Phi X "
    r"\Psi \Omega",
)
for _offset, (_small, _capital) in enumerate(zip(_GREEK_SMALL, _GREEK_CAPITAL, strict=True)):
    _LATEX_SYMBOLS.append((chr(0x3B1 + _offset), f"${_small}$"))
    if _capital != "-":  # U+03A2 is unassigned (there is no capital final sigma)
        _capital_tex = f"${_capital}$" if _capital.startswith("\\") else _capital
        _LATEX_SYMBOLS.append((chr(0x391 + _offset), _capital_tex))
_LATEX_SYMBOLS += [("ϑ", r"$\vartheta$"), ("ϕ", r"$\phi$"), ("ϵ", r"$\epsilon$")]
_SUPERSCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ", "0123456789+-=()n", strict=True))
_SUBSCRIPTS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎", "0123456789+-=()", strict=True))
# Accents LaTeX sets on any letter, by the combining character Unicode decomposes them into.
_ACCENTS = {
    "̀": "`",
    "́": "'",
    "̂": "^",
    "̃": "~",
    "̄": "=",
    "̆": "u",
    "̇": ".",
    "̈": '"',
    "̊": "r",
    "̋": "H",
    "̌": "v",
    "̧": "c",
    "̨": "k",
}


def _latex_char(char: str) -> str:
    """One character with no rheology meaning: as is, escaped, accented, or "?" when pdflatex
    with its default fonts cannot set it."""
    if char in _LATEX_SPECIAL:
        return _LATEX_SPECIAL[char]
    if char.isascii() or "À" <= char <= "ÿ":
        return char  # ASCII and the Latin-1 letters compile with LaTeX's default utf8 input
    import unicodedata

    parts = unicodedata.normalize("NFD", char)
    if len(parts) == 2 and parts[0].isascii() and parts[0].isalpha() and parts[1] in _ACCENTS:
        return f"\\{_ACCENTS[parts[1]]}{{{parts[0]}}}"
    plain = unicodedata.normalize("NFKD", char)
    if plain.isascii() and plain.isprintable() and plain:
        return "".join(_LATEX_SPECIAL.get(c, c) for c in plain)
    return "?"


def latex(text: str) -> str:
    """Plain interface text as LaTeX: specials escaped, rheology symbols typeset, and any
    other character pdflatex cannot set by default spelled out or replaced."""
    out = []
    i = 0
    while i < len(text):
        for table, mark in ((_SUPERSCRIPTS, "^"), (_SUBSCRIPTS, "_")):
            if text[i] in table:
                j = i
                while j < len(text) and text[j] in table:
                    j += 1
                run = "".join(table[c] for c in text[i:j])
                out.append(f"${mark}{run}$" if len(run) == 1 else f"${mark}{{{run}}}$")
                i = j
                break
        else:
            for symbol, replacement in _LATEX_SYMBOLS:
                if text.startswith(symbol, i):
                    out.append(replacement)
                    i += len(symbol)
                    break
            else:
                out.append(_latex_char(text[i]))
                i += 1
    # Two symbols side by side ("aT·ω") would give "$a_T$$\omega$", and "$$" opens display
    # maths: join neighbouring segments into one. An escaped "\$" is left alone.
    return re.sub(r"(?<!\\)\$\$", "", "".join(out))


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def _pgf_axis_options(
    axis: AxisSpec, which: str, limits: tuple[float, float, bool] | None
) -> list[str]:
    options = [f"{which}mode={'log' if axis.log else 'normal'}"]
    if axis.title:
        options.append(f"{which}label={{{latex(axis.title)}}}")
    if limits is not None:
        lo, hi, reverse = limits
        options += [f"{which}min={_fmt(lo)}"] if lo is not None else []
        options += [f"{which}max={_fmt(hi)}"] if hi is not None else []
        if reverse:
            options.append(f"{which} dir=reverse")
    return options


def _data_range(xs: list[float], log: bool) -> tuple[float, float, bool] | None:
    """The x range the data needs, with matplotlib's 5% margin either side, so both axes of a
    two-axis figure can be given the same one."""
    if not xs:
        return None
    lo, hi = min(xs), max(xs)
    if log:
        lo, hi = math.log10(lo), math.log10(hi)
    pad = (hi - lo) * 0.05 if hi > lo else max(abs(lo) * 0.05, 0.5)
    lo, hi = lo - pad, hi + pad
    if log:
        lo, hi = 10**lo, 10**hi
    return lo, hi, False


def _pgf_plot_options(series: SeriesSpec, color: str, spec: FigureSpec) -> str:
    filled, open_ = _PGF_MARKS[series.marker]
    mark = filled if series.filled else open_
    size = f"mark size={(series.marker_size or spec.marker_size) * _PX_TO_PT / 2:.2f}pt"
    if series.key or spec.mode == "markers":
        return f"only marks, mark={mark}, {size}, color={color}"
    style = "densely dotted" if series.axis == "y2" else "solid"
    width = f"line width={spec.line_width * _PX_TO_PT:.2f}pt"
    if spec.mode == "lines":
        return f"no markers, {style}, {width}, color={color}"
    return f"mark={mark}, {size}, {style}, {width}, color={color}"


# The pgfplots legend is set in the document's font, unknown here: its layout is estimated
# for 10 pt text, in ems.
_MM_PER_EM = 10 / 72.27 * MM_PER_INCH


def _pgf_legend_columns(names: list[str], spec: FigureSpec, has_y2: bool) -> int:
    """How many columns a legend above the plot can have and still fit the plot's width, the
    way the matplotlib figure wraps it. pgfplots fills a legend row by row, and each column is
    as wide as its widest entry."""
    # pgfplots takes about 45 pt of the width for the axis labels, whatever the figure has.
    available = (spec.width_mm - 16) / _MM_PER_EM
    # A legend image and the gap after it, then about 0.45 em a character.
    widths = [1.8 + 0.45 * len(name) for name in names]
    for ncols in range(len(widths), 1, -1):
        if sum(max(widths[c::ncols]) for c in range(ncols)) <= available:
            return ncols
    return 1


def pgfplots_zip(spec: FigureSpec, workdir: Path) -> bytes:
    """A zip with figure.tex and one .dat table per series, ready for \\input{figure}."""
    from lada.exporters import write_pgfplots_table

    _check(spec)
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    colors: dict[str, str] = {}
    plots: dict[str, list[str]] = {"y": [], "y2": []}
    files: list[Path] = []
    all_xs: list[float] = []
    legend_on = spec.legend != "none"
    entries: list[str] = []  # the legend's names, in order, at most MAX_LEGEND_ENTRIES
    wanted = 0  # how many entries the legend would have without the cap
    for series in spec.series:
        hexcode = _paint(series).lstrip("#")
        name = colors.setdefault(hexcode, f"ladac{len(colors) + 1}")
        options = _pgf_plot_options(series, name, spec)
        axis = "y2" if series.axis == "y2" and spec.y2 is not None else "y"
        legend = legend_on and bool(series.key or (series.show_legend and series.name))
        wanted += legend
        legend = legend and len(entries) < MAX_LEGEND_ENTRIES
        if legend:
            entries.append(series.name)
        # Every legend entry lives on the main axis, so there is one legend: right-axis
        # series are described there with \addlegendimage and drawn unlabelled on their own.
        if series.key:
            if legend:
                plots["y"].append(f"\\addlegendimage{{{options}}}")
                plots["y"].append(f"\\addlegendentry{{{latex(series.name)}}}")
            continue
        axis_spec = spec.y2 if axis == "y2" else spec.y
        xs, ys = _points(series, spec.x.log, axis_spec.log)
        all_xs += xs
        path = data_dir / f"series-{len(files) + 1:02d}.dat"
        write_pgfplots_table(pd.DataFrame({"x": xs, "y": ys}), str(path), fmt="%.10g")
        files.append(path)
        here = legend and axis == "y"
        plots[axis].append(
            f"\\addplot[{options}{'' if here else ', forget plot'}] "
            f"table[x=x, y=y] {{data/{path.name}}};"
        )
        if legend and axis == "y2":
            plots["y"].append(f"\\addlegendimage{{{options}}}")
        if legend:
            plots["y"].append(f"\\addlegendentry{{{latex(series.name)}}}")

    has_y2 = bool(plots["y2"]) and spec.y2 is not None
    x_limits = _limits(spec.x)
    if has_y2 and (x_limits is None or None in x_limits[:2]):
        # Two overlaid axes line up only if both are told the same x range: an end left to
        # fit the data is filled in from the data.
        fitted = _data_range(all_xs, spec.x.log)
        if x_limits is None or fitted is None:
            x_limits = x_limits if fitted is None else fitted
        else:
            x_limits = (
                fitted[0] if x_limits[0] is None else x_limits[0],
                fitted[1] if x_limits[1] is None else x_limits[1],
                False,
            )
    x_options = _pgf_axis_options(spec.x, "x", x_limits)
    if has_y2:
        x_options.append("enlarge x limits=false")
    # The right axis is drawn first, and named, when the legend has to sit beyond its labels.
    right_first = has_y2 and spec.legend == "outside"

    common = [f"width={spec.width_mm:.1f}mm", f"height={spec.height_mm:.1f}mm", "tick align=inside"]
    main = common + x_options + _pgf_axis_options(spec.y, "y", _limits(spec.y))
    if spec.grid:
        main.append("grid=major")
    if has_y2:
        main.append("axis y line*=left")
    if entries and spec.legend == "above":
        columns = _pgf_legend_columns(entries, spec, has_y2)
        main.append(
            f"legend style={{at={{(0,1.02)}},anchor=south west,legend columns={columns},"
            "/tikz/every even column/.append style={column sep=0.6em}}"
        )
    elif entries and spec.legend == "below":
        # Under the tick labels and the axis title: about 4.4 em of a 10 pt document font,
        # as a fraction of the axis height.
        drop = 44 / (spec.height_mm * 72 / MM_PER_INCH)
        columns = _pgf_legend_columns(entries, spec, has_y2)
        main.append(
            f"legend style={{at={{(0.5,{-drop:.3f})}},anchor=north,legend columns={columns},"
            "/tikz/every even column/.append style={column sep=0.6em}}"
        )
    elif entries and spec.legend == "custom":
        main.append(
            f"legend style={{at={{({spec.legend_x:.3f},{spec.legend_y:.3f})}},anchor=north west}}"
        )
    elif entries and right_first:
        main.append("legend style={at={(ladaright.outer north east)},anchor=north west}")
    elif entries and spec.legend in _PGF_LEGEND:
        main.append(_PGF_LEGEND[spec.legend])
    if entries:
        main.append("legend cell align=left")  # names read from a common left edge, as in the app

    lines = [
        "% Figure exported from LaDa web.",
        "% Preamble: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "% Use with \\input{figure} from the folder this archive was unpacked into.",
        f"% Drawn at {spec.font_size:g} pt in the app; here the document's font size applies.",
    ]
    notes = _limit_notes(spec)
    if wanted > len(entries) and legend_on:
        notes.append(_legend_cap_note(wanted))
    lines += [f"% Note: {note}" for note in notes]
    if any(s.marker == "cross" for s in spec.series):
        lines += _PGF_CROSS_MARKS
    for hexcode, name in colors.items():
        lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{hexcode.upper()}}}")
    lines.append("\\begin{tikzpicture}")
    right: list[str] = []
    if has_y2:
        second = (
            (["name=ladaright"] if right_first else [])
            + common
            + [
                "axis y line*=right",
                "axis x line=none",
                *x_options,
                *_pgf_axis_options(spec.y2, "y", _limits(spec.y2)),
            ]
        )
        right = ["\\begin{axis}[", "  " + ",\n  ".join(second), "]"]
        right += ["  " + p for p in plots["y2"]]
        right.append("\\end{axis}")
    if right_first:
        lines += right
    lines += ["\\begin{axis}[", "  " + ",\n  ".join(main), "]"]
    lines += ["  " + p for p in plots["y"]]
    lines.append("\\end{axis}")
    if not right_first:
        lines += right
    lines.append("\\end{tikzpicture}")
    (workdir / "figure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(workdir / "figure.tex", "figure.tex")
        for path in files:
            archive.write(path, f"data/{path.name}")
    return buffer.getvalue()
