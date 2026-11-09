"""Template setup, static-file URLs, and how a request says it wants HTML."""

import hashlib
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

from . import __version__
from .catalog import INPUT_KINDS, SECTION_BY_KEY, WORKSPACE_BY_KEY, navigation

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


_fingerprints: dict[str, tuple[int, str]] = {}


def static_url(path: str) -> str:
    """The URL of a file under static/, with a fingerprint of its contents as `?v=`.

    The static files are served without a Cache-Control header, so a browser is free to guess
    how long its copy stays fresh, and Safari in particular keeps using an old stylesheet
    without asking the server again. A URL that changes whenever the file's bytes do means
    no browser can show a stale copy. The hash is cached per modification time, so it is
    recomputed only after an edit.
    """
    file = STATIC_DIR / path
    try:
        mtime = file.stat().st_mtime_ns
    except OSError:
        return f"/static/{path}"
    cached = _fingerprints.get(path)
    if cached is None or cached[0] != mtime:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()[:10]
        cached = (mtime, digest)
        _fingerprints[path] = cached
    return f"/static/{path}?v={cached[1]}"


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# `sections` and `workspaces` let a template name the section and workspace a page belongs to
# (the breadcrumb needs titles, and a Calculation carries only the section key).
templates.env.globals.update(
    nav=navigation(),
    sections=SECTION_BY_KEY,
    workspaces=WORKSPACE_BY_KEY,
    input_kinds=INPUT_KINDS,
    static_url=static_url,
    app_version=__version__,
)


def wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")
