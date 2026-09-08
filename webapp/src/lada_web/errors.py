"""Application errors and the handlers that turn them into partials, pages or JSON."""

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

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


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


def error_response(
    request: Request, status_code: int, message: str, *, title: str = "That didn't work"
) -> Response:
    context = {"title": title, "message": message}
    if is_htmx(request):
        return templates.TemplateResponse(
            request, "partials/error.html", context, status_code=status_code
        )
    if wants_html(request):
        return templates.TemplateResponse(
            request, "error_page.html", context, status_code=status_code
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
    async def too_large(request: Request, exc: Exception) -> Response:
        return error_response(request, 413, _message(exc), title="Too large")

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
