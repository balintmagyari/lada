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

MM_PER_INCH = 25.4
MAX_POINTS = 500_000
MAX_PIXELS = 80_000_000

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
    color: str = "#000000"
    marker: Marker = "circle"
    filled: bool = True
    show_legend: bool = True
    # A key entry explains a marker; it has no data and is drawn in grey.
    key: bool = False

    @field_validator("color")
    @classmethod
    def _hex(cls, value: str) -> str:
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
        "inside-top-right",
        "inside-top-left",
        "inside-bottom-right",
        "inside-bottom-left",
        "outside",
        "none",
    ] = "above"
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
    "cross": "x",
}
_KEY_GREY = "#6b6b6b"
# Plotly sizes markers and lines in screen pixels; a printed figure is measured in points,
# and on paper the same marker reads larger than on screen, so it is drawn a little smaller.
_PX_TO_PT = 0.6

_LEGEND_PLACES = {
    "inside-top-right": {"loc": "upper right"},
    "inside-top-left": {"loc": "upper left"},
    "inside-bottom-right": {"loc": "lower right"},
    "inside-bottom-left": {"loc": "lower left"},
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
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter

    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_label = ax.set_xlabel if which == "x" else ax.set_ylabel
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    set_scale("log" if axis.log else "linear")
    set_label(axis.title, parse_math=False)
    lo, hi = axis.min, axis.max
    if lo is not None and hi is not None and lo != hi and (not axis.log or (lo > 0 and hi > 0)):
        set_lim(lo, hi)
    target = ax.xaxis if which == "x" else ax.yaxis
    if axis.log:
        lo_shown, hi_shown = sorted(ax.get_xlim() if which == "x" else ax.get_ylim())
        decades = math.log10(hi_shown / lo_shown) if lo_shown > 0 else 99
        short = decades < 1.5
        target.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
        target.set_minor_formatter(NullFormatter())
        if short:
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


def _legend_above(fig, ax, handles, labels):
    """A legend above the plot, wrapped onto as many rows as it needs to fit the axes' width,
    so a long legend never squeezes the plot to make room for itself."""
    renderer = fig.canvas.get_renderer()
    axes_width = ax.get_window_extent(renderer).width
    legend = None
    for ncols in range(len(labels), 0, -1):
        if legend is not None:
            legend.remove()
        legend = ax.legend(
            _row_major(handles, ncols),
            _row_major(labels, ncols),
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncols=ncols,
            borderaxespad=0.2,
            columnspacing=1.0,
        )
        if legend.get_window_extent(renderer).width <= axes_width * 1.02:
            break
    if legend is not None and legend.get_window_extent(renderer).height > fig.bbox.height / 3:
        fig._lada_notes.append(
            "The legend takes up much of this figure. Shorter legend names, a legend inside "
            "the plot, or the double-column size would leave more room for the data."
        )
    return legend


def _legend(fig, ax, ax2, handles, labels, spec: FigureSpec) -> None:
    if spec.legend == "above":
        legend = _legend_above(fig, ax, handles, labels)
    elif spec.legend == "outside":
        legend = ax.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(1.18 if ax2 else 1.02, 1)
        )
    else:
        # Drawn on the top axes so the right-axis series cannot cover it.
        legend = (ax2 or ax).legend(handles, labels, **_LEGEND_PLACES[spec.legend])
    for text in legend.get_texts():
        text.set_parse_math(False)


def _draw(spec: FigureSpec):
    """The matplotlib Figure for a spec. Built with the object API, never pyplot. Must run
    inside `matplotlib.rc_context(_rc(spec))`, held open until the figure is saved: tick marks
    are created when the figure is drawn, from the style settings in force at that moment."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _check(spec)
    # Constrained layout fits labels and legend inside the requested size, so a figure asked
    # for at 85 mm comes out exactly 85 mm wide, as a journal column needs.
    fig = Figure(
        figsize=(spec.width_mm / MM_PER_INCH, spec.height_mm / MM_PER_INCH), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    FigureCanvasAgg(fig)  # a renderer, so the legend can be measured while it is laid out
    fig._lada_notes = []  # advice about the layout, passed back to the studio
    ax = fig.add_subplot()
    has_y2 = spec.y2 is not None and any(s.axis == "y2" for s in spec.series)
    ax2 = ax.twinx() if has_y2 else None

    handles, labels = [], []
    for series in spec.series:
        target = ax2 if series.axis == "y2" and ax2 is not None else ax
        axis_spec = spec.y2 if target is ax2 else spec.y
        color = _KEY_GREY if series.key else series.color
        marker = _MPL_MARKERS[series.marker]
        show_markers = series.key or spec.mode in ("markers", "both")
        show_line = not series.key and spec.mode in ("lines", "both")
        kwargs = {
            "color": color,
            "marker": marker if show_markers else "None",
            "markersize": spec.marker_size * _PX_TO_PT,
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
        _legend(fig, ax, ax2, handles, labels, spec)
    return fig


MEDIA_TYPES = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}

# matplotlib's style settings are process-wide, and rc_context changes them for the whole
# process while it is open; one render at a time keeps two requests on different worker
# threads from drawing with each other's fonts and sizes.
_STYLE_LOCK = threading.Lock()


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
    with _STYLE_LOCK, matplotlib.rc_context(_rc(spec)):
        fig = _draw(spec)
        fig.savefig(buffer, format=fmt, dpi=dpi, metadata=metadata)
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
    "cross": ("x", "x"),
}
_PGF_LEGEND = {
    "above": "legend style={at={(0,1.02)},anchor=south west,legend columns=-1,"
    "/tikz/every even column/.append style={column sep=0.6em}}",
    "inside-top-right": "legend pos=north east",
    "inside-top-left": "legend pos=north west",
    "inside-bottom-right": "legend pos=south east",
    "inside-bottom-left": "legend pos=south west",
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
    ("|η*|", r"$|\eta^*|$"),
    ("|G*|", r"$|G^*|$"),
    ("G′", r"$G'$"),
    ("G″", r"$G''$"),
    ("tan δ", r"$\tan\delta$"),
    ("γ̇", r"$\dot\gamma$"),
    ("ω", r"$\omega$"),
    ("δ", r"$\delta$"),
    ("η", r"$\eta$"),
    ("γ", r"$\gamma$"),
    ("τ", r"$\tau$"),
    ("σ", r"$\sigma$"),
    ("°C", r"\textdegree{}C"),
    ("°F", r"\textdegree{}F"),
    ("°", r"\textdegree{}"),
    ("·", r"$\cdot$"),
    ("µ", r"$\mu$"),
    ("²", r"$^2$"),
    ("⁻¹", r"$^{-1}$"),
]


def latex(text: str) -> str:
    """Plain interface text as LaTeX: specials escaped, rheology symbols typeset."""
    out = []
    i = 0
    while i < len(text):
        for symbol, replacement in _LATEX_SYMBOLS:
            if text.startswith(symbol, i):
                out.append(replacement)
                i += len(symbol)
                break
        else:
            out.append(_LATEX_SPECIAL.get(text[i], text[i]))
            i += 1
    return "".join(out)


def _pgf_axis_options(axis: AxisSpec, which: str) -> list[str]:
    options = [f"{which}mode={'log' if axis.log else 'normal'}"]
    if axis.title:
        options.append(f"{which}label={{{latex(axis.title)}}}")
    if axis.min is not None and axis.max is not None and axis.min != axis.max:
        options += [f"{which}min={axis.min:.10g}", f"{which}max={axis.max:.10g}"]
    return options


def _pgf_plot_options(series: SeriesSpec, color: str, spec: FigureSpec) -> str:
    filled, open_ = _PGF_MARKS[series.marker]
    mark = filled if series.filled else open_
    size = f"mark size={spec.marker_size * _PX_TO_PT / 2:.2f}pt"
    if series.key or spec.mode == "markers":
        return f"only marks, mark={mark}, {size}, color={color}"
    style = "densely dotted" if series.axis == "y2" else "solid"
    width = f"line width={spec.line_width * _PX_TO_PT:.2f}pt"
    if spec.mode == "lines":
        return f"no markers, {style}, {width}, color={color}"
    return f"mark={mark}, {size}, {style}, {width}, color={color}"


def pgfplots_zip(spec: FigureSpec, workdir: Path) -> bytes:
    """A zip with figure.tex and one .dat table per series, ready for \\input{figure}."""
    from lada.exporters import write_pgfplots_table

    _check(spec)
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    colors: dict[str, str] = {}
    plots: dict[str, list[str]] = {"y": [], "y2": []}
    files: list[Path] = []
    for series in spec.series:
        hexcode = "6b6b6b" if series.key else series.color.lstrip("#")
        name = colors.setdefault(hexcode, f"ladac{len(colors) + 1}")
        options = _pgf_plot_options(series, name, spec)
        axis = "y2" if series.axis == "y2" and spec.y2 is not None else "y"
        # Every legend entry lives on the main axis, so there is one legend: right-axis
        # series are described there with \addlegendimage and drawn unlabelled on their own.
        if series.key:
            plots["y"].append(f"\\addlegendimage{{{options}}}")
            plots["y"].append(f"\\addlegendentry{{{latex(series.name)}}}")
            continue
        axis_spec = spec.y2 if axis == "y2" else spec.y
        xs, ys = _points(series, spec.x.log, axis_spec.log)
        path = data_dir / f"series-{len(files) + 1:02d}.dat"
        write_pgfplots_table(pd.DataFrame({"x": xs, "y": ys}), str(path), fmt="%.10g")
        files.append(path)
        legend = bool(series.show_legend and series.name)
        here = legend and axis == "y"
        plots[axis].append(
            f"\\addplot[{options}{'' if here else ', forget plot'}] "
            f"table[x=x, y=y] {{data/{path.name}}};"
        )
        if legend and axis == "y2":
            plots["y"].append(f"\\addlegendimage{{{options}}}")
        if legend:
            plots["y"].append(f"\\addlegendentry{{{latex(series.name)}}}")

    common = [f"width={spec.width_mm:.1f}mm", f"height={spec.height_mm:.1f}mm", "tick align=inside"]
    main = common + _pgf_axis_options(spec.x, "x") + _pgf_axis_options(spec.y, "y")
    if spec.grid:
        main.append("grid=major")
    if plots["y2"]:
        main.append("axis y line*=left")
    if spec.legend in _PGF_LEGEND:
        main.append(_PGF_LEGEND[spec.legend])

    lines = [
        "% Figure exported from LaDa web.",
        "% Preamble: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "% Use with \\input{figure} from the folder this archive was unpacked into.",
        f"% Drawn at {spec.font_size:g} pt in the app; here the document's font size applies.",
    ]
    for hexcode, name in colors.items():
        lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{hexcode.upper()}}}")
    lines += ["\\begin{tikzpicture}", "\\begin{axis}[", "  " + ",\n  ".join(main), "]"]
    lines += ["  " + p for p in plots["y"]]
    lines.append("\\end{axis}")
    if plots["y2"] and spec.y2 is not None:
        second = common + [
            "axis y line*=right",
            "axis x line=none",
            _pgf_axis_options(spec.x, "x")[0],
            *_pgf_axis_options(spec.y2, "y"),
        ]
        lines += ["\\begin{axis}[", "  " + ",\n  ".join(second), "]"]
        lines += ["  " + p for p in plots["y2"]]
        lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")
    (workdir / "figure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(workdir / "figure.tex", "figure.tex")
        for path in files:
            archive.write(path, f"data/{path.name}")
    return buffer.getvalue()
