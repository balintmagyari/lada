"""Application factory."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .errors import register_error_handlers
from .rendering import STATIC_DIR
from .routers import conformations, dynamics, modifiers, pages, parsers, rheology


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="LaDa web",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings or Settings()
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
