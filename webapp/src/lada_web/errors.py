"""Application errors and the handlers that turn them into pages or JSON."""

import logging
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .catalog import Calculation, get_calculation_by_path
from .rendering import templates, wants_html

_log = logging.getLogger(__name__)


class UploadTooLarge(Exception):
    def __init__(self, filename: str, limit_mb: float) -> None:
        super().__init__(f"'{filename}' is larger than the {limit_mb:g} MiB upload limit.")


class ServerBusy(Exception):
    """The server cannot take this request right now; answered with 429."""


_SAFE_ERRORS = (ValueError, KeyError, UploadTooLarge, ServerBusy)


def _message(exc: Exception) -> str:
    # UnicodeDecodeError is a ValueError whose args[0] is only the codec name ("utf-8"), which
    # tells a user nothing about what went wrong.
    if isinstance(exc, UnicodeDecodeError):
        return (
            "The file could not be read as text — it looks binary. Check that this is a "
            "rheometer export or a spreadsheet."
        )
    return str(exc.args[0]) if exc.args else str(exc)


def client_safe_message(exc: Exception) -> str:
    """What an unexpected exception is allowed to tell a client.

    Errors raised deliberately by this app carry messages written for the user, so they pass
    through unchanged. Anything else — a full disk, a permission problem, a bug — arrives with
    text that routinely embeds server paths (`/private/tmp/lada-web-.../000/export.csv`), and
    an instance meant to be reachable by other people should not hand out its filesystem
    layout. Those get a fixed message instead; the real one goes to the server log.
    """
    if isinstance(exc, _SAFE_ERRORS):
        return _message(exc)
    return (
        "The request stopped because of an unexpected server error. The details were "
        "written to the server log."
    )


def _page_came_from(request: Request) -> Calculation | None:
    """The tool page this request came from, if the referer names one of ours.

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
    return get_calculation_by_path(referer.path) if referer.path.count("/") == 2 else None


def error_response(
    request: Request,
    status_code: int,
    message: str,
    *,
    title: str = "That didn't work",
    headers: dict[str, str] | None = None,
) -> Response:
    if wants_html(request):
        back = _page_came_from(request)
        context = {
            "title": title,
            "message": message,
            "back_path": back.path if back else None,
            "back_title": back.title if back else None,
        }
        return templates.TemplateResponse(
            request, "error_page.html", context, status_code=status_code, headers=headers
        )
    return JSONResponse({"detail": message}, status_code=status_code, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    async def domain_error(request: Request, exc: Exception) -> Response:
        return error_response(
            request, 422, _message(exc), title="The file or settings could not be processed"
        )

    @app.exception_handler(UploadTooLarge)
    async def too_large(request: Request, exc: Exception) -> Response:
        return error_response(request, 413, _message(exc), title="Too large")

    @app.exception_handler(ServerBusy)
    async def server_busy(request: Request, exc: Exception) -> Response:
        return error_response(request, 429, _message(exc), title="Server busy")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        if wants_html(request):
            problems = [
                f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
                for e in exc.errors()
            ]
            return error_response(request, 422, "; ".join(problems), title="Check the form")
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        # The exception's headers carry what the status code promises: a 405 must say which
        # methods are allowed (`Allow`), and dropping them breaks well-behaved clients.
        return error_response(
            request,
            exc.status_code,
            str(exc.detail),
            title="Not available",
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        # Logged in full server-side; the response says only that it happened.
        _log.exception("Unhandled error for %s %s", request.method, request.url.path)
        return error_response(
            request,
            500,
            client_safe_message(exc),
            title="Something went wrong",
        )
