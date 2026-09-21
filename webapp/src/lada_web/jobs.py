"""In-memory background job tracking for real progress reporting.

A calculation that reports progress (see the `progress_callback` parameter on the core `lada`
functions) can run on a background thread instead of blocking the request: the route starts a
job and returns immediately (202 + job id), and the browser polls `GET /api/jobs/{job_id}`
until it finishes. No Redis/Celery — this is a single-process app, so an in-memory store with a
TTL sweep and a concurrency cap are enough.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Request, UploadFile

from .config import Settings
from .errors import TooManyJobs, client_safe_message
from .uploads import cleanup_workspace, new_workspace, save_uploads

_log = logging.getLogger(__name__)


def _cleanup_quietly(workspace: Path) -> None:
    """Remove a job's workspace, logging rather than raising if it cannot be removed."""
    try:
        cleanup_workspace(workspace)
    except Exception:
        _log.exception("Could not clean up workspace %s", workspace)


JobStatus = Literal["pending", "running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "pending"
    progress: float = 0.0
    message: str = ""
    result: Any | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class JobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, fraction: float, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "running"
                job.progress = fraction
                job.message = message

    def set_result(self, job_id: str, result: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "done"
                job.progress = 1.0
                job.result = result

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.error = message

    def sweep(self, ttl_seconds: float) -> None:
        """Drop jobs older than `ttl_seconds`, whether or not their result was ever fetched."""
        cutoff = time.monotonic() - ttl_seconds
        with self._lock:
            stale = [job_id for job_id, job in self._jobs.items() if job.created_at < cutoff]
            for job_id in stale:
                del self._jobs[job_id]


def throttled(
    store: JobStore, job_id: str, *, min_interval: float = 0.1
) -> Callable[[int, int], None]:
    """Wrap a raw per-iteration progress callback into a rate-limited one.

    Core `lada` functions call their callback unconditionally on every iteration — ISF's inner
    loop can do this up to `n_vectors * n_frames` times — so this collapses updates down to a
    UI-relevant frequency instead of hammering the store's lock once per iteration.
    """
    last_update = [0.0]

    def callback(current: int, total: int) -> None:
        now = time.monotonic()
        if current == total or now - last_update[0] >= min_interval:
            last_update[0] = now
            store.update_progress(job_id, current / total, f"{current}/{total}")

    return callback


def run_job(
    store: JobStore,
    workspace: Path,
    target: Callable[[Callable[[int, int], None]], Any],
    semaphore: threading.Semaphore,
    limit: int,
) -> Job:
    """Start `target` on a background thread, passing it a throttled progress callback.

    `workspace` is torn down once the job finishes (success or error) — it's no longer needed
    after the result has been extracted from it, unlike a view route's request-scoped workspace,
    since the job outlives the request that started it. Raises `TooManyJobs` (which the route
    lets propagate as a 429) if the concurrency cap is already exhausted, without starting a
    thread or touching the workspace's lifetime further than cleaning it up immediately.
    """
    if not semaphore.acquire(blocking=False):
        _cleanup_quietly(workspace)
        raise TooManyJobs(limit)

    job = store.create()

    def body() -> None:
        try:
            result = target(throttled(store, job.id))
            store.set_result(job.id, result)
        except Exception as exc:
            store.set_error(job.id, client_safe_message(exc))
            _log.exception("Job %s failed", job.id)
        finally:
            # The release must not hinge on the cleanup surviving: a cleanup that raised here
            # would strand this permit for the life of the process, and enough of those would
            # take the server's concurrency to zero with nothing visible to any client.
            _cleanup_quietly(workspace)
            semaphore.release()

    threading.Thread(target=body, daemon=True).start()
    return job


def start_workspace(
    settings: Settings, files: list[UploadFile], max_bytes: int, max_files: int
) -> tuple[Path, list[Path]]:
    """Create a job's workspace and save its uploads into it; cleans up on failure.

    A job's workspace can't use the request-scoped `get_workspace` dependency (it must outlive
    the request that starts the job, and is torn down by `run_job` once the background thread
    finishes instead), so it's created directly here — with the same "clean up on failure"
    guarantee `download_workspace` gives the download routes.
    """
    ws = new_workspace(settings)
    try:
        paths = save_uploads(files, ws, max_bytes, max_files)
    except Exception:
        cleanup_workspace(ws)
        raise
    return ws, paths


def per_file_progress(
    overall_callback: Callable[[int, int], None] | None, file_index: int, n_files: int
) -> Callable[[int, int], None] | None:
    """Fold one file's (current, total) into an overall fraction across `n_files` files.

    File totals (e.g. frame counts) can differ slightly between repeat files, so this
    normalizes each file to a fraction of 1.0 rather than assuming a shared total, then scales
    that fraction into a fixed-resolution (current, total) pair the job store can still treat
    the same way as a single-file call's.
    """
    if overall_callback is None:
        return None
    scale = 10_000

    def callback(current: int, total: int) -> None:
        fraction = (file_index + current / total) / n_files
        overall_callback(round(fraction * scale), scale)

    return callback


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_job_semaphore(request: Request) -> threading.Semaphore:
    return request.app.state.job_semaphore
