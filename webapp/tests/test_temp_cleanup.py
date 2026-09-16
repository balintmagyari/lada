"""Nothing an uploader sends may outlive its request.

Every route computes inside a temporary directory under `settings.temp_root`. These tests
drive one client through the whole surface — success, validation failure, domain error,
oversize, guardrail and unexpected error — and assert after *each* request that the temp
root is gone or empty.
"""

from pathlib import Path

import numpy as np

from lada_web.services import parsers_service

from .conftest import files_upload, upload

LOG_URL = "/api/parsers/log/table"
DUMP_DOWNLOAD_URL = "/api/parsers/dump/table/download"
RG_URL = "/api/conformations/rg"
SEGMENT_URL = "/api/dynamics/segment-acf"
STRESS_URL = "/api/rheology/stress-relaxation"
MODULI_DOWNLOAD_URL = "/api/rheology/dynamic-moduli/download"
REWRITE_URL = "/api/modifiers/rewrite-end-beads"

MIB = 1024 * 1024
UNIFORM = {"mode": "uniform", "n_chains": "4", "beads_per_chain": "8", "first_index": "0"}
THERMO_STATE = {"volume": "1000", "temperature": "1"}
END_BEADS = {"new_end_type": "3", "base_type": "1"}


def leftovers(temp_root: Path) -> list[str]:
    """Whatever is still sitting in the temp root; empty when nothing leaked."""
    if not temp_root.exists():
        return []
    return sorted(child.name for child in temp_root.iterdir())


def write_npz(path: Path, n_frames: int, n_atoms: int) -> Path:
    coords = np.arange(n_frames * n_atoms * 3, dtype=float).reshape(n_frames, n_atoms, 3)
    np.savez(path, coords=coords)
    return path


def explode(*args, **kwargs):
    raise RuntimeError("the parser exploded")


def post_after_breaking_the_parser(monkeypatch, client, path: Path):
    """An error nobody planned for: the service raises where the route expects a DataFrame."""
    monkeypatch.setattr(parsers_service, "log_table", explode)
    return client.post(LOG_URL, files=upload(path))


def test_no_request_leaves_a_temporary_workspace_behind(
    client, data_dir, temp_root, tmp_path, htmx, monkeypatch
):
    log = data_dir / "sample_log.lammps"
    dump = data_dir / "sample_dump.dump"
    npz = data_dir / "sample_trajectory.npz"
    acf = data_dir / "sample_acf.txt"
    stress_acf = data_dir / "sample_stress_acf.txt"
    molecular = data_dir / "sample_molecular.data"
    atomic = data_dir / "sample.data"
    too_many_frames = write_npz(tmp_path / "too_many_frames.npz", n_frames=101, n_atoms=32)

    battery = [
        ("log view", 200, lambda: client.post(LOG_URL, files=upload(log))),
        ("log without a file", 422, lambda: client.post(LOG_URL)),
        (
            "dump table csv download",
            200,
            lambda: client.post(f"{DUMP_DOWNLOAD_URL}?fmt=csv", files=upload(dump)),
        ),
        ("rg view", 200, lambda: client.post(RG_URL, files=files_upload(dump))),
        (
            "rg with a column the dump does not have",
            422,
            lambda: client.post(RG_URL, files=files_upload(dump), data={"mass_col": "mass"}),
        ),
        (
            "upload over the size limit",
            413,
            lambda: client.post(LOG_URL, files={"file": ("big.lammps", b"x" * (MIB + 1))}),
        ),
        (
            "segment acf view",
            200,
            lambda: client.post(
                SEGMENT_URL, files=files_upload(npz), data=UNIFORM | {"time_per_frame": "1.0"}
            ),
        ),
        (
            "segment acf over the frame guardrail",
            413,
            lambda: client.post(
                SEGMENT_URL,
                files=files_upload(too_many_frames),
                data=UNIFORM | {"time_per_frame": "1.0"},
            ),
        ),
        (
            "stress relaxation with the wrong acf schema",
            422,
            lambda: client.post(STRESS_URL, files=files_upload(acf), data=THERMO_STATE),
        ),
        (
            "dynamic moduli pgfplots download",
            200,
            lambda: client.post(
                f"{MODULI_DOWNLOAD_URL}?fmt=pgfplots",
                files=files_upload(stress_acf),
                data=THERMO_STATE,
            ),
        ),
        (
            "rewrite end beads",
            200,
            lambda: client.post(REWRITE_URL, files=upload(molecular), data=END_BEADS),
        ),
        (
            "rewrite end beads on an atomic-style file",
            422,
            lambda: client.post(REWRITE_URL, files=upload(atomic), data=END_BEADS),
        ),
        ("log view via htmx", 200, lambda: client.post(LOG_URL, files=upload(log), headers=htmx)),
        (
            "unexpected server error",
            500,
            lambda: post_after_breaking_the_parser(monkeypatch, client, log),
        ),
    ]

    seen = {}
    for label, expected_status, request in battery:
        response = request()
        seen[label] = response
        assert response.status_code == expected_status, f"{label}: {response.text[:400]}"
        assert leftovers(temp_root) == [], f"{label} left {leftovers(temp_root)} behind"

    # The battery really exercised the paths it claims to.
    assert "<table" in seen["log view via htmx"].text
    assert seen["dump table csv download"].headers["content-disposition"].startswith("attachment")
    assert "RuntimeError" in seen["unexpected server error"].json()["detail"]


def test_workspaces_really_live_under_the_temp_root(client, data_dir, temp_root, monkeypatch):
    """Guard the guard: with cleanup disabled the battery's assertion has to fail."""
    monkeypatch.setattr("lada_web.uploads.cleanup_workspace", lambda path: None)

    view = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"))
    assert view.status_code == 200
    assert [name for name in leftovers(temp_root) if name.startswith("lada-web-")]

    download = client.post(
        f"{DUMP_DOWNLOAD_URL}?fmt=csv", files=upload(data_dir / "sample_dump.dump")
    )
    assert download.status_code == 200
    assert len(leftovers(temp_root)) == 2
