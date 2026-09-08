"""HTML pages: the index and one page per catalog entry."""

from fastapi import APIRouter, HTTPException, Request

from ..catalog import calculations_by_section, get_calculation
from ..rendering import templates

router = APIRouter(include_in_schema=False)


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"groups": calculations_by_section()})


@router.get("/{section}/{slug}")
def calculation_page(request: Request, section: str, slug: str):
    calc = get_calculation(section, slug)
    if calc is None:
        raise HTTPException(status_code=404, detail="There is no page at this address.")
    return templates.TemplateResponse(request, calc.template, {"calc": calc})
