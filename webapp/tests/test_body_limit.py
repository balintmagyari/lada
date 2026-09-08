"""A request body larger than the server accepts is refused before it is read.

The per-file limits in `uploads.save_upload` only apply once the multipart body has been
streamed in — Starlette has already spooled the file part to disk by then. The middleware in
`create_app` answers earlier, on the declared `Content-Length`, with the ceiling that belongs
to the route family being addressed. A body that declares no length cannot be measured that
way at all, so a POST to `/api/` without one is refused outright.
"""

from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.main import create_app

from .conftest import upload

LOG_URL = "/api/parsers/log/table"
SEGMENT_URL = "/api/dynamics/segment-acf"
MIB = 1024 * 1024
BOUNDARY = "bodylimitboundary"

# The test settings allow 1 MB uploads, so the middleware's ceiling is 1 MiB + 1 MiB.
OVER_THE_LIMIT = b"x" * int(2.1 * MIB)

TOO_LARGE = "The request body is larger than this server accepts."


def multipart(payload: bytes, filename: str = "huge.lammps") -> bytes:
    """One file part, hand-rolled, so it can be sent without a Content-Length."""
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    return head.encode() + payload + f"\r\n--{BOUNDARY}--\r\n".encode()


def multipart_headers() -> dict[str, str]:
    return {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}


def test_body_over_the_limit_is_413_and_leaves_nothing_behind(client, temp_root):
    r = client.post(LOG_URL, files={"file": ("huge.lammps", OVER_THE_LIMIT)})
    assert r.status_code == 413
    assert r.json()["detail"] == TOO_LARGE
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_body_over_the_limit_is_refused_on_every_route(client, temp_root):
    r = client.post(
        "/api/modifiers/rewrite-end-beads",
        files={"file": ("huge.data", OVER_THE_LIMIT)},
        data={"new_end_type": "3", "base_type": "1"},
    )
    assert r.status_code == 413
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_oversized_htmx_post_gets_the_error_partial(client, htmx):
    r = client.post(LOG_URL, files={"file": ("huge.lammps", OVER_THE_LIMIT)}, headers=htmx)
    assert r.status_code == 413
    assert r.headers["content-type"].startswith("text/html")
    assert 'role="alert"' in r.text
    assert "Too large" in r.text


def test_oversized_browser_post_gets_the_error_page(client):
    r = client.post(
        LOG_URL, files={"file": ("huge.lammps", OVER_THE_LIMIT)}, headers={"Accept": "text/html"}
    )
    assert r.status_code == 413
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text
    assert "larger than this server accepts" in r.text


def test_a_normal_request_still_works(client, data_dir):
    r = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 200, r.text
    assert r.json()["n_rows_total"] == 5


def test_pages_and_static_files_still_work(client):
    assert client.get("/").status_code == 200
    assert client.get("/parsers/log").status_code == 200
    assert client.get("/static/css/app.css").status_code == 200


def test_a_download_still_cleans_up_behind_the_middleware(client, data_dir, temp_root):
    r = client.post(
        "/api/parsers/log/table/download?fmt=csv", files=upload(data_dir / "sample_log.lammps")
    )
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith("attachment")
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_a_body_under_the_middleware_ceiling_reaches_the_upload_limit(client):
    """1.9 MiB clears the 2 MiB ceiling, so `save_upload` is what refuses it, not the middleware."""
    r = client.post(LOG_URL, files={"file": ("big.lammps", b"x" * int(1.9 * MIB))})
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]


# --- The two holes the middleware used to have -----------------------------


def test_a_body_with_no_declared_length_is_refused(client, temp_root):
    """A chunked upload carries no Content-Length, so its size cannot be checked at all.

    Starlette used to spool the whole file part to disk before any route saw it, with
    nothing measuring it on the way in.
    """
    r = client.post(LOG_URL, content=iter([multipart(OVER_THE_LIMIT)]), headers=multipart_headers())
    assert r.status_code == 413
    assert "Content-Length" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_a_body_with_no_declared_length_is_refused_however_small_it_is(client):
    """The check is on the framing, not the size: an unmeasurable body is never accepted."""
    r = client.post(LOG_URL, content=iter([multipart(b"Step\n0\n")]), headers=multipart_headers())
    assert r.status_code == 413
    assert "Content-Length" in r.json()["detail"]


def test_a_get_without_a_body_is_not_caught_by_that_rule(client):
    """Only POSTs to /api/ need a declared length; a page request has no body to measure."""
    assert client.get("/parsers/log").status_code == 200


def test_the_ceiling_is_per_route_family(tmp_path):
    """Text routes take 50 MB by default; only the .npz routes get the trajectory budget."""
    settings = Settings(
        max_upload_mb=1,
        max_trajectory_mb=20,
        max_trajectory_frames=100,
        max_table_rows=20,
        temp_root=tmp_path / "uploads",
    )
    client = TestClient(create_app(settings), raise_server_exceptions=False)
    body = b"x" * int(4 * MIB)

    text = client.post(LOG_URL, files={"file": ("big.lammps", body)})
    assert text.status_code == 413
    assert text.json()["detail"] == TOO_LARGE

    # The same body addressed to a trajectory route is inside that route's larger ceiling,
    # so the middleware passes it on and the .npz reader is what turns it down.
    npz = client.post(
        SEGMENT_URL,
        files={"file": ("big.npz", body)},
        data={
            "mode": "uniform",
            "n_chains": "4",
            "beads_per_chain": "8",
            "first_index": "0",
            "time_per_frame": "1.0",
        },
    )
    assert npz.status_code == 422
    assert ".npz" in npz.json()["detail"]
