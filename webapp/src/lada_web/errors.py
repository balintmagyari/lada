"""Application errors and the handlers that turn them into partials, pages or JSON."""

from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .catalog import Calculation, get_calculation
from .rendering import is_htmx, templates, wants_html


class UploadTooLarge(Exception):
    def __init__(self, filename: str, limit_mb: float) -> None:
        super().__init__(f"'{filename}' is larger than the {limit_mb:g} MB upload limit.")


class TrajectoryTooLarge(Exception):
    def __init__(self, n_frames: int, limit: int) -> None:
        super().__init__(
            f"The trajectory has {n_frames} frames; this server accepts at most {limit}. "
            "Save fewer frames or increase LADA_WEB_MAX_TRAJECTORY_FRAMES."
        )


class TooManyJobs(Exception):
    """Raised when the background-job concurrency cap is already exhausted."""

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"Too many calculations are running at once (limit {limit}). Try again shortly."
        )


class TrajectoryTooLargeInMemory(Exception):
    """A small compressed archive can still declare an array too big to load."""

    def __init__(self, needed_bytes: int, limit_bytes: int) -> None:
        mb = 1024 * 1024
        super().__init__(
            f"The trajectory needs {needed_bytes / mb:,.0f} MB in memory once decompressed; "
            f"this server allows {limit_bytes / mb:g} MB. Save fewer frames or fewer atoms, "
            "or increase LADA_WEB_MAX_TRAJECTORY_MEMORY_MB."
        )


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


def _page_came_from(request: Request) -> Calculation | None:
    """The calculation page this request was submitted from, if the referer names one of ours.

    Download buttons are native form submits, so an error replaces the whole page and takes
    the rendered table and the filled-in form with it. Offering the way back costs nothing.
    Only a same-origin path that matches a real catalog entry is used — the referer itself is
    never reflected into the page.
    """
    try:
        referer = urlsplit(request.headers.get("referer", ""))
    except ValueError:
        # e.g. an unclosed IPv6 bracket ("Invalid IPv6 URL") — no real browser sends this,
        # but the error page must not itself 500 while rendering a 4xx.
        return None
    if referer.scheme and referer.scheme not in ("http", "https"):
        return None
    origin = urlsplit(str(request.base_url))
    if referer.netloc and (referer.scheme, referer.netloc) != (origin.scheme, origin.netloc):
        return None
    parts = referer.path.strip("/").split("/")
    return get_calculation(*parts) if len(parts) == 2 else None


def error_response(
    request: Request, status_code: int, message: str, *, title: str = "That didn't work"
) -> Response:
    context = {"title": title, "message": message}
    if is_htmx(request):
        return templates.TemplateResponse(
            request, "partials/error.html", context, status_code=status_code
        )
    if wants_html(request):
        back = _page_came_from(request)
        page_context = context | {
            "back_path": back.path if back else None,
            "back_title": back.title if back else None,
        }
        return templates.TemplateResponse(
            request, "error_page.html", page_context, status_code=status_code
        )
    return JSONResponse({"detail": message}, status_code=status_code)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    async def domain_error(request: Request, exc: Exception) -> Response:
        return error_response(
            request, 422, _message(exc), title="The file or settings could not be processed"
        )

    @app.exception_handler(UploadTooLarge)
    @app.exception_handler(TrajectoryTooLarge)
    @app.exception_handler(TrajectoryTooLargeInMemory)
    async def too_large(request: Request, exc: Exception) -> Response:
        return error_response(request, 413, _message(exc), title="Too large")

    @app.exception_handler(TooManyJobs)
    async def too_many_jobs(request: Request, exc: Exception) -> Response:
        return error_response(request, 429, _message(exc), title="Server busy")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        if is_htmx(request) or wants_html(request):
            problems = [
                f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
                for e in exc.errors()
            ]
            return error_response(request, 422, "; ".join(problems), title="Check the form")
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return error_response(request, exc.status_code, str(exc.detail), title="Not available")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        return error_response(
            request,
            500,
            f"Unexpected error while processing the request: {type(exc).__name__}: {exc}",
            title="Something went wrong",
        )
