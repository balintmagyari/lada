"""HTML pages: the overview, and one page per catalog entry."""

from fastapi import APIRouter, HTTPException, Request

from ..catalog import WORKSPACES, calculations_by_section, get_calculation_by_path, workspace_of
from ..rendering import templates

router = APIRouter(include_in_schema=False)

# HEAD as well as GET: link checkers, uptime monitors and `curl -I` all ask with HEAD, and a
# GET-only route answers them 405. The server drops the body of a HEAD response itself.
PAGE_METHODS = ["GET", "HEAD"]

# Version 1 has one workspace, so its overview is the home page as well.
HOME_WORKSPACE = WORKSPACES[0]


def _overview(request: Request):
    context = {
        "workspace": HOME_WORKSPACE,
        "here_workspace": HOME_WORKSPACE.key,
        "groups": calculations_by_section(HOME_WORKSPACE.key),
    }
    return templates.TemplateResponse(request, "index.html", context)


@router.api_route("/", methods=PAGE_METHODS)
def home(request: Request):
    return _overview(request)


@router.api_route(HOME_WORKSPACE.home, methods=PAGE_METHODS)
def workspace_overview(request: Request):
    """The same overview at its own address, which every tool page's breadcrumb links to."""
    return _overview(request)


@router.api_route("/{first}/{second}", methods=PAGE_METHODS)
def calculation_page(request: Request, first: str, second: str):
    calc = get_calculation_by_path(f"/{first}/{second}")
    if calc is None:
        raise HTTPException(status_code=404, detail="There is no page at this address.")
    context = {"calc": calc, "here_workspace": workspace_of(calc).key}
    return templates.TemplateResponse(request, calc.template, context)
