"""Application factory."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .errors import error_response, register_error_handlers
from .rendering import STATIC_DIR
from .routers import pages, rheometry

# Headroom over the largest accepted upload for the multipart envelope and the other fields.
BODY_OVERHEAD_BYTES = 1024 * 1024

# The import route takes files; every other /api/ route takes a JSON document.
UPLOAD_PATHS = frozenset({"/api/rheometry/parse"})

# Content types an upload (or a plain form) arrives as. Anything else is held to the JSON
# ceiling, whatever route it is sent to.
FORM_CONTENT_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")

TOO_LARGE_MESSAGE = "The request body is larger than this server accepts."
NO_LENGTH_MESSAGE = (
    "This request declares no Content-Length, so its size cannot be checked before the "
    "server reads it. Send the upload without chunked transfer encoding."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="LaDa web",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    resolved = settings or Settings()
    app.state.settings = resolved

    # An import may carry up to max_files_per_request files in one body.
    upload_ceiling = (
        resolved.max_upload_bytes * resolved.max_files_per_request + BODY_OVERHEAD_BYTES
    )
    json_ceiling = resolved.max_json_bytes

    def body_ceiling(path: str, content_type: str) -> int:
        is_form = content_type.split(";", 1)[0].strip().lower() in FORM_CONTENT_TYPES
        if path in UPLOAD_PATHS and is_form:
            return upload_ceiling
        return json_ceiling

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse a body this server will not accept, on its declared length.

        This is the only limit that runs before the body is read: by the time a route (and
        `uploads.save_upload` with it) can see the upload, Starlette has already parsed the
        multipart form and spooled the file part to disk.

        A declared length is the only thing that can be checked this early, so it is not a
        guarantee: a body that declares none — a chunked upload, say — cannot be measured
        at all, and a POST to /api/ carrying one is refused rather than streamed in
        unbounded. Nothing here bounds a body that lies about its length; `save_upload`
        stops that one, later and at the cost of the bytes already spooled.
        """
        declared = request.headers.get("content-length", "")
        if declared.isdecimal():
            ceiling = body_ceiling(request.url.path, request.headers.get("content-type", ""))
            if int(declared) > ceiling:
                return error_response(request, 413, TOO_LARGE_MESSAGE, title="Too large")
        elif request.method == "POST" and request.url.path.startswith("/api/"):
            return error_response(request, 413, NO_LENGTH_MESSAGE, title="Too large")
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_error_handlers(app)
    app.include_router(rheometry.router)
    app.include_router(pages.router)  # last: it owns the catch-all /{first}/{second}
    return app


app = create_app()
