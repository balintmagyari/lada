"""A request body larger than the server accepts is refused before it is read.

The per-file limit in `uploads.save_upload` only applies once the multipart body has been
streamed in — Starlette has already spooled the file part to disk by then. The middleware in
`create_app` answers earlier, on the declared `Content-Length`: the import route may carry
`max_files_per_request` files of `max_upload_mb` each, every other route only a JSON document.
A body that declares no length cannot be measured that way at all, so a POST to `/api/`
without one is refused outright.
"""

from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.main import create_app

from .conftest import files_upload

PARSE_URL = "/api/rheometry/parse"
FIGURE_URL = "/api/rheometry/figure?fmt=svg"
MIB = 1024 * 1024
BOUNDARY = "bodylimitboundary"

# The test settings allow 1 MiB per file and 10 files, so the import ceiling is 11 MiB.
OVER_THE_LIMIT = b"x" * int(11.5 * MIB)

TOO_LARGE = "The request body is larger than this server accepts."


def multipart(payload: bytes, filename: str = "huge.csv") -> bytes:
    """One file part, hand-rolled, so it can be sent without a Content-Length."""
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    return head.encode() + payload + f"\r\n--{BOUNDARY}--\r\n".encode()


def multipart_headers() -> dict[str, str]:
    return {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}


def nothing_left(temp_root) -> bool:
    return not temp_root.exists() or list(temp_root.iterdir()) == []


def test_body_over_the_limit_is_413_and_leaves_nothing_behind(client, temp_root):
    r = client.post(PARSE_URL, files=[("files", ("huge.csv", OVER_THE_LIMIT))])
    assert r.status_code == 413
    assert r.json()["detail"] == TOO_LARGE
    assert nothing_left(temp_root)


def test_oversized_browser_post_gets_the_error_page(client):
    r = client.post(
        PARSE_URL,
        files=[("files", ("huge.csv", OVER_THE_LIMIT))],
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 413
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text
    assert "larger than this server accepts" in r.text


def test_a_normal_import_still_works(client, rheo_data):
    r = client.post(PARSE_URL, files=files_upload(rheo_data / "flow_curve.csv"))
    assert r.status_code == 200, r.text
    assert [d["filename"] for d in r.json()["datasets"]] == ["flow_curve.csv"]


def test_pages_and_static_files_still_work(client):
    assert client.get("/").status_code == 200
    assert client.get("/rheometer/frequency-sweep").status_code == 200
    assert client.get("/static/css/app.css").status_code == 200


def test_a_file_under_the_middleware_ceiling_reaches_the_per_file_limit(client):
    """1.5 MiB clears the 11 MiB request ceiling, so `save_upload` is what refuses it."""
    r = client.post(PARSE_URL, files=[("files", ("big.csv", b"x" * int(1.5 * MIB)))])
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]


def test_a_json_route_keeps_the_small_json_ceiling(client):
    """The figure, pgfplots, rederive and superposition routes take JSON, never a file."""
    r = client.post(
        FIGURE_URL,
        content=b"{" + b" " * (6 * MIB) + b"}",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert r.json()["detail"] == TOO_LARGE


def test_a_json_route_keeps_its_ceiling_whatever_the_content_type_claims(client):
    r = client.post(
        FIGURE_URL,
        content=b"x" * (6 * MIB),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413


def test_the_json_ceiling_is_configurable(temp_root):
    small = TestClient(
        create_app(Settings(max_json_mb=0.0001, temp_root=temp_root)),
        raise_server_exceptions=False,
    )
    r = small.post("/api/rheometry/tts", json={"sweeps": [], "padding": "x" * 200})
    assert r.status_code == 413


# --- Bodies that cannot be measured ----------------------------------------------------------


def test_a_body_with_no_declared_length_is_refused(client, temp_root):
    """A chunked upload carries no Content-Length, so its size cannot be checked at all."""
    r = client.post(
        PARSE_URL, content=iter([multipart(OVER_THE_LIMIT)]), headers=multipart_headers()
    )
    assert r.status_code == 413
    assert "Content-Length" in r.json()["detail"]
    assert nothing_left(temp_root)


def test_a_body_with_no_declared_length_is_refused_however_small_it_is(client):
    """The check is on the framing, not the size: an unmeasurable body is never accepted."""
    r = client.post(
        PARSE_URL, content=iter([multipart(b"a,b\n1,2\n")]), headers=multipart_headers()
    )
    assert r.status_code == 413
    assert "Content-Length" in r.json()["detail"]


def test_a_get_without_a_body_is_not_caught_by_that_rule(client):
    """Only POSTs to /api/ need a declared length; a page request has no body to measure."""
    assert client.get("/rheometer/my-data").status_code == 200
    assert client.get("/api/rheometry/quantities").status_code == 200
