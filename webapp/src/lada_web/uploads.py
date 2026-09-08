"""Temporary workspaces, size-limited uploads and file downloads.

Nothing an uploader sends is kept: every request computes inside its own temporary
directory, which is removed when the response has been produced.
"""

import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .config import Settings, get_settings
from .errors import UploadTooLarge

CHUNK_SIZE = 1024 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str | None, default: str = "upload") -> str:
    """Basename only, restricted to [A-Za-z0-9._-]; never empty."""
    cleaned = _UNSAFE.sub("_", Path(name or "").name).strip("._")
    return cleaned or default


def new_workspace(settings: Settings) -> Path:
    if settings.temp_root is not None:
        settings.temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="lada-web-", dir=settings.temp_root))


def cleanup_workspace(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def save_upload(upload: UploadFile, directory: Path, max_bytes: int) -> Path:
    """Stream an upload into `directory` in 1 MiB chunks; reject it once it exceeds max_bytes."""
    name = safe_filename(upload.filename)
    limit_mb = max_bytes / (1024 * 1024)
    if upload.size is not None and upload.size > max_bytes:
        raise UploadTooLarge(name, limit_mb)
    dest = directory / name
    written = 0
    try:
        with dest.open("wb") as out:
            for chunk in iter(lambda: upload.file.read(CHUNK_SIZE), b""):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLarge(name, limit_mb)
                out.write(chunk)
    except UploadTooLarge:
        dest.unlink(missing_ok=True)
        raise
    return dest


def get_workspace(settings: Annotated[Settings, Depends(get_settings)]) -> Iterator[Path]:
    """Per-request temp directory for view routes; removed when the request finishes."""
    ws = new_workspace(settings)
    try:
        yield ws
    finally:
        cleanup_workspace(ws)


@contextmanager
def download_workspace(settings: Settings) -> Iterator[Path]:
    """Temp directory for download routes: removed on error here, otherwise by file_download."""
    ws = new_workspace(settings)
    try:
        yield ws
    except BaseException:
        cleanup_workspace(ws)
        raise


def file_download(path: Path, workspace: Path, media_type: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        background=BackgroundTask(cleanup_workspace, workspace),
    )
