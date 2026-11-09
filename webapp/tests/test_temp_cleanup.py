"""Nothing an uploader sends may outlive its request.

Every route that touches the disk works inside a temporary directory under
`settings.temp_root`. These tests drive one client through the whole surface — success,
validation failure, unreadable file, oversize, export and unexpected error — and assert after
*each* request that the temp root is gone or empty.
"""

from pathlib import Path

from lada_web.routers import rheometry

from .conftest import files_upload
from .test_rheometry_export import sweep_spec

PARSE_URL = "/api/rheometry/parse"
MIB = 1024 * 1024


def leftovers(temp_root: Path) -> list[str]:
    """Whatever is still sitting in the temp root; empty when nothing leaked."""
    if not temp_root.exists():
        return []
    return sorted(child.name for child in temp_root.iterdir())


def explode(*args, **kwargs):
    raise RuntimeError("the reader exploded")


def post_after_breaking_the_reader(monkeypatch, client, path: Path):
    """An error nobody planned for: the reader raises something that is not a ParseError."""
    monkeypatch.setattr(rheometry, "read_file", explode)
    return client.post(PARSE_URL, files=files_upload(path))


def test_no_request_leaves_a_temporary_workspace_behind(client, rheo_data, temp_root, monkeypatch):
    flow = rheo_data / "flow_curve.csv"
    trios = rheo_data / "trios_frequency_sweeps.txt"
    workbook = rheo_data / "two_samples.xlsx"
    not_a_table = rheo_data / "not_a_table.txt"
    orchestrator = rheo_data / "orchestrator" / "TA_Orchestrator_FS.RSD02"

    battery = [
        ("one export", 200, lambda: client.post(PARSE_URL, files=files_upload(flow))),
        (
            "several exports at once",
            200,
            lambda: client.post(PARSE_URL, files=files_upload(trios, workbook, orchestrator)),
        ),
        (
            "an unreadable file",
            200,
            lambda: client.post(PARSE_URL, files=files_upload(not_a_table)),
        ),
        ("no file at all", 422, lambda: client.post(PARSE_URL)),
        (
            "a file over the size limit",
            413,
            lambda: client.post(PARSE_URL, files=[("files", ("big.csv", b"x" * (MIB + 1)))]),
        ),
        (
            "an svg figure",
            200,
            lambda: client.post("/api/rheometry/figure?fmt=svg", json=sweep_spec()),
        ),
        (
            "a pgfplots archive",
            200,
            lambda: client.post("/api/rheometry/pgfplots", json=sweep_spec()),
        ),
        (
            "a pgfplots archive of a figure that cannot be drawn",
            422,
            lambda: client.post("/api/rheometry/pgfplots", json=sweep_spec(series=[])),
        ),
        (
            "an unexpected server error",
            500,
            lambda: post_after_breaking_the_reader(monkeypatch, client, flow),
        ),
    ]

    seen = {}
    for label, expected_status, request in battery:
        response = request()
        seen[label] = response
        assert response.status_code == expected_status, f"{label}: {response.text[:400]}"
        assert leftovers(temp_root) == [], f"{label} left {leftovers(temp_root)} behind"

    # The battery really exercised the paths it claims to.
    assert len(seen["several exports at once"].json()["datasets"]) == 3
    assert seen["an unreadable file"].json()["errors"][0]["filename"] == "not_a_table.txt"
    assert seen["a pgfplots archive"].headers["content-type"] == "application/zip"
    # The body must not name the exception or any server path — that detail goes to the log.
    unexpected = seen["an unexpected server error"].json()["detail"]
    assert "unexpected server error" in unexpected
    assert "RuntimeError" not in unexpected and "/" not in unexpected


def test_workspaces_really_live_under_the_temp_root(client, rheo_data, temp_root, monkeypatch):
    """Guard the guard: with cleanup disabled the battery's assertion has to fail."""
    monkeypatch.setattr("lada_web.uploads.cleanup_workspace", lambda path: None)
    r = client.post(PARSE_URL, files=files_upload(rheo_data / "flow_curve.csv"))
    assert r.status_code == 200
    assert [name for name in leftovers(temp_root) if name.startswith("lada-web-")]
