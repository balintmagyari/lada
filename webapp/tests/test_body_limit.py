"""A request body larger than the server accepts is refused before it is read.

The per-file limits in `uploads.save_upload` only apply once the multipart body has been
streamed in. The middleware in `create_app` answers on the declared `Content-Length`
instead, so an oversized post costs neither disk nor parse time.
"""

from .conftest import upload

LOG_URL = "/api/parsers/log/table"
MIB = 1024 * 1024

# The test settings allow 1 MB trajectories, so the middleware's ceiling is 1 MiB + 1 MiB.
OVER_THE_LIMIT = b"x" * int(2.1 * MIB)


def test_body_over_the_limit_is_413_and_leaves_nothing_behind(client, temp_root):
    r = client.post(LOG_URL, files={"file": ("huge.lammps", OVER_THE_LIMIT)})
    assert r.status_code == 413
    assert r.json()["detail"] == "The request body is larger than this server accepts."
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


def test_a_body_just_under_the_limit_is_let_through(client):
    """1.9 MB is under the 2 MiB ceiling: the middleware passes it to the upload limit."""
    r = client.post(LOG_URL, files={"file": ("big.lammps", b"x" * int(1.9 * MIB))})
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]
