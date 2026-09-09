"""Template setup and the one place that decides between an HTML partial and JSON."""

import math
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates

from . import __version__
from .catalog import SECTIONS, calculations_by_section
from .schemas.common import Cell, TableResult

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

# How many significant figures the on-screen table and the plot hover readout show.
# Downloads and the JSON API stay bit-exact: rounding here is presentation only.
SIGNIFICANT_FIGURES = 6

# Past 2**53 a float64 cannot hold consecutive integers, so spelling out every digit of one
# would claim precision it does not have. Below it, an integral float is a whole number.
_EXACT_INTEGER_LIMIT = 2**53


def sigfig(value: Cell) -> str:
    """Render one table cell at SIGNIFICANT_FIGURES, leaving whole numbers and text alone.

    A plain '%.6g' would turn the timestep 10000000.0 into '1e+07', so an integral float
    prints as an integer instead. Strings pass through untouched and None is an empty cell.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value.is_integer() and abs(value) < _EXACT_INTEGER_LIMIT:
            return str(int(value))
        return f"{value:.{SIGNIFICANT_FIGURES}g}"
    return str(value)


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# `sections` lets a template name the section a calculation belongs to (the breadcrumb
# needs the title, and a Calculation carries only the key).
templates.env.globals.update(
    nav=calculations_by_section(),
    sections={section.key: section for section in SECTIONS},
    app_version=__version__,
    significant_figures=SIGNIFICANT_FIGURES,
)
templates.env.filters["sigfig"] = sigfig


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def table_response(request: Request, result: TableResult, download_url: str | None) -> Response:
    """HTML partial for htmx requests, JSON for everything else."""
    if is_htmx(request):
        context = {
            "result": result,
            "result_json": result.model_dump_json(),
            "download_url": download_url,
        }
        return templates.TemplateResponse(request, "partials/table_result.html", context)
    return JSONResponse(result.model_dump(mode="json"))
