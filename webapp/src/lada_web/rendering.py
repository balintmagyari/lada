"""Template setup and the one place that decides between an HTML partial and JSON."""

from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates

from . import __version__
from .catalog import SECTIONS, calculations_by_section
from .schemas.common import TableResult

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# `sections` lets a template name the section a calculation belongs to (the breadcrumb
# needs the title, and a Calculation carries only the key).
templates.env.globals.update(
    nav=calculations_by_section(),
    sections={section.key: section for section in SECTIONS},
    app_version=__version__,
)


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
