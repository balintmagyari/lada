"""HTML pages: the home page, one overview per workspace, and one page per catalog entry."""

from fastapi import APIRouter, HTTPException, Request

from ..catalog import (
    WORKSPACE_BY_KEY,
    WORKSPACES,
    calculations_by_section,
    get_calculation_by_path,
    workspace_of,
)
from ..rendering import templates

router = APIRouter(include_in_schema=False)

# HEAD as well as GET: link checkers, uptime monitors and `curl -I` all ask with HEAD, and a
# GET-only route answers them 405. The server drops the body of a HEAD response itself.
PAGE_METHODS = ["GET", "HEAD"]


@router.api_route("/", methods=PAGE_METHODS)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {"doors": WORKSPACES})


def _overview(request: Request, key: str):
    workspace = WORKSPACE_BY_KEY[key]
    context = {
        "workspace": workspace,
        "here_workspace": key,
        "groups": calculations_by_section(key),
    }
    return templates.TemplateResponse(request, "index.html", context)


@router.api_route("/simulations", methods=PAGE_METHODS)
def simulations_overview(request: Request):
    return _overview(request, "simulations")


@router.api_route("/rheometer", methods=PAGE_METHODS)
def rheometer_overview(request: Request):
    return _overview(request, "rheometer")


@router.api_route("/{first}/{second}", methods=PAGE_METHODS)
def calculation_page(request: Request, first: str, second: str):
    calc = get_calculation_by_path(f"/{first}/{second}")
    if calc is None:
        raise HTTPException(status_code=404, detail="There is no page at this address.")
    context = {"calc": calc, "here_workspace": workspace_of(calc).key}
    return templates.TemplateResponse(request, calc.template, context)
