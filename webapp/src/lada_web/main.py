"""Application factory."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .errors import error_response, register_error_handlers
from .rendering import STATIC_DIR
from .routers import conformations, dynamics, modifiers, pages, parsers, rheology

# Headroom over the largest accepted upload for the multipart envelope and the other fields.
BODY_OVERHEAD_BYTES = 1024 * 1024


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
    max_body_bytes = resolved.max_trajectory_bytes + BODY_OVERHEAD_BYTES

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse an oversized body on its declared length, before anything reads it."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_body_bytes:
            return error_response(
                request,
                413,
                "The request body is larger than this server accepts.",
                title="Too large",
            )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_error_handlers(app)
    for router in (
        parsers.router,
        conformations.router,
        dynamics.router,
        rheology.router,
        modifiers.router,
        pages.router,  # last: it owns the catch-all /{section}/{slug}
    ):
        app.include_router(router)
    return app


app = create_app()
