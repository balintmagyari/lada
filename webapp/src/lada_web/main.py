"""Application factory."""

import threading
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .errors import error_response, register_error_handlers
from .jobs import JobStore
from .rendering import STATIC_DIR
from .routers import conformations, dynamics, jobs, modifiers, pages, parsers, rheology

# How often the background sweep thread prunes jobs older than settings.job_ttl_seconds. Capped
# at a quarter of the TTL so a short TTL is actually honoured: at a fixed minute, a 10-second TTL
# would leave finished jobs fetchable for up to a minute, with nothing swept at all until the
# first tick. Floored so a tiny TTL cannot spin the thread.
JOB_SWEEP_INTERVAL_SECONDS = 60
MIN_JOB_SWEEP_INTERVAL_SECONDS = 1.0


def sweep_interval(ttl_seconds: float) -> float:
    return max(MIN_JOB_SWEEP_INTERVAL_SECONDS, min(JOB_SWEEP_INTERVAL_SECONDS, ttl_seconds / 4))


# Headroom over the largest accepted upload for the multipart envelope and the other fields.
BODY_OVERHEAD_BYTES = 1024 * 1024

# Only these routes take an .npz trajectory; everything else takes a text upload, and must be
# held to the much smaller text budget rather than to the trajectory one.
TRAJECTORY_PREFIX = "/api/dynamics/"

# These routes accept several files (repeat seeds, or regime files) to average into one
# calculation; parsers/modifiers stay single-file, so they keep the tighter one-file ceiling.
MULTI_FILE_TEXT_PREFIXES = ("/api/conformations/", "/api/rheology/")

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

    app.state.job_store = JobStore()
    app.state.job_semaphore = threading.Semaphore(resolved.max_concurrent_jobs)

    def sweep_loop() -> None:
        while True:
            time.sleep(sweep_interval(resolved.job_ttl_seconds))
            app.state.job_store.sweep(resolved.job_ttl_seconds)

    threading.Thread(target=sweep_loop, daemon=True).start()

    # Dynamics and the multi-file text routes may carry up to max_files_per_request uploads in
    # one body; parsers/modifiers stay single-file, so they keep the tighter one-file ceiling.
    upload_ceiling = resolved.max_upload_bytes + BODY_OVERHEAD_BYTES
    multi_upload_ceiling = (
        resolved.max_upload_bytes * resolved.max_files_per_request + BODY_OVERHEAD_BYTES
    )
    trajectory_ceiling = (
        resolved.max_trajectory_bytes * resolved.max_files_per_request + BODY_OVERHEAD_BYTES
    )

    def body_ceiling(path: str) -> int:
        if path.startswith(TRAJECTORY_PREFIX):
            return trajectory_ceiling
        if path.startswith(MULTI_FILE_TEXT_PREFIXES):
            return multi_upload_ceiling
        return upload_ceiling

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse a body this server will not accept, on its declared length.

        This is the only limit that runs before the body is read: by the time a route (and
        `uploads.save_upload` with it) can see the upload, Starlette has already parsed the
        multipart form and spooled the file part to disk. The ceiling is the one belonging
        to the route family being addressed, since the text routes accept far less than the
        trajectory routes do.

        A declared length is the only thing that can be checked this early, so it is not a
        guarantee: a body that declares none — a chunked upload, say — cannot be measured
        at all, and a POST to /api/ carrying one is refused rather than streamed in
        unbounded. Nothing here bounds a body that lies about its length; `save_upload`
        stops that one, later and at the cost of the bytes already spooled.
        """
        declared = request.headers.get("content-length", "")
        if declared.isdecimal():
            if int(declared) > body_ceiling(request.url.path):
                return error_response(request, 413, TOO_LARGE_MESSAGE, title="Too large")
        elif request.method == "POST" and request.url.path.startswith("/api/"):
            return error_response(request, 413, NO_LENGTH_MESSAGE, title="Too large")
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_error_handlers(app)
    for router in (
        parsers.router,
        conformations.router,
        dynamics.router,
        rheology.router,
        modifiers.router,
        jobs.router,
        pages.router,  # last: it owns the catch-all /{section}/{slug}
    ):
        app.include_router(router)
    return app


app = create_app()
