"""JobStore unit tests, plus route-level start-then-poll integration tests."""

import threading
import time

import pytest

from lada_web.errors import TooManyJobs
from lada_web.jobs import JobStore, per_file_progress, run_job, throttled

from .conftest import files_upload

# --- JobStore ----------------------------------------------------------------


def test_create_starts_pending_with_zero_progress():
    store = JobStore()
    job = store.create()
    assert job.status == "pending"
    assert job.progress == 0.0
    assert store.get(job.id) is job


def test_get_missing_job_returns_none():
    store = JobStore()
    assert store.get("does-not-exist") is None


def test_update_progress_moves_to_running():
    store = JobStore()
    job = store.create()
    store.update_progress(job.id, 0.5, "5/10")
    assert job.status == "running"
    assert job.progress == 0.5
    assert job.message == "5/10"


def test_update_progress_on_missing_job_is_a_no_op():
    store = JobStore()
    store.update_progress("nope", 0.5)  # must not raise


def test_set_result_marks_done_at_full_progress():
    store = JobStore()
    job = store.create()
    store.set_result(job.id, {"columns": ["x"]})
    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result == {"columns": ["x"]}


def test_set_error_marks_error_and_keeps_the_message():
    store = JobStore()
    job = store.create()
    store.set_error(job.id, "boom")
    assert job.status == "error"
    assert job.error == "boom"


def test_sweep_drops_jobs_older_than_the_ttl():
    store = JobStore()
    job = store.create()
    job.created_at -= 1000  # simulate an old job without sleeping in the test
    store.sweep(ttl_seconds=1)
    assert store.get(job.id) is None


def test_sweep_keeps_jobs_within_the_ttl():
    store = JobStore()
    job = store.create()
    store.sweep(ttl_seconds=1000)
    assert store.get(job.id) is not None


# --- throttled -----------------------------------------------------------------


def test_throttled_always_reports_the_final_call():
    store = JobStore()
    job = store.create()
    cb = throttled(store, job.id, min_interval=1000)  # so nothing but the final call gets through
    for i in range(1, 6):
        cb(i, 5)
    assert job.progress == 1.0
    assert job.message == "5/5"


def test_throttled_reports_the_first_call_of_a_slow_series():
    store = JobStore()
    job = store.create()
    cb = throttled(store, job.id, min_interval=1000)
    cb(1, 100)
    assert job.progress == pytest.approx(0.01)


# --- per_file_progress -----------------------------------------------------


def test_per_file_progress_scales_into_the_files_own_slice():
    calls = []
    combined = per_file_progress(lambda c, t: calls.append((c, t)), file_index=1, n_files=4)
    combined(5, 10)  # halfway through file index 1 of 4 -> overall fraction (1 + 0.5) / 4 = 0.375
    current, total = calls[0]
    assert current / total == pytest.approx(0.375)


def test_per_file_progress_is_none_when_no_overall_callback_is_given():
    assert per_file_progress(None, 0, 3) is None


# --- run_job / TooManyJobs ---------------------------------------------------


def wait_for(predicate, timeout: float = 2.0) -> None:
    """Bounded retry for a background-thread side effect (status, then cleanup, then release)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("condition was never met")


def test_run_job_releases_the_semaphore_and_reports_the_result(tmp_path):
    store = JobStore()
    semaphore = threading.Semaphore(1)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    job = run_job(store, workspace, lambda cb: "ok", semaphore, limit=1)

    wait_for(lambda: store.get(job.id).status != "pending")
    assert store.get(job.id).status == "done"
    assert store.get(job.id).result == "ok"

    # Cleanup and the semaphore release happen in the background thread's `finally`, after
    # `set_result` — a moment after the status is observably "done", not atomically with it.
    wait_for(lambda: not workspace.exists())
    wait_for(lambda: semaphore.acquire(blocking=False))


def test_run_job_records_an_exception_as_an_error(tmp_path):
    store = JobStore()
    semaphore = threading.Semaphore(1)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def target(cb):
        raise ValueError("bad input")

    job = run_job(store, workspace, target, semaphore, limit=1)
    wait_for(lambda: store.get(job.id).status != "pending")
    assert store.get(job.id).status == "error"
    assert store.get(job.id).error == "bad input"
    wait_for(lambda: not workspace.exists())


def test_run_job_refuses_when_the_semaphore_is_exhausted(tmp_path):
    store = JobStore()
    semaphore = threading.Semaphore(1)
    semaphore.acquire()  # the one slot is already taken
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(TooManyJobs):
        run_job(store, workspace, lambda cb: "ok", semaphore, limit=1)
    assert not workspace.exists()  # cleaned up even though the job never started


# --- Route-level: start -> poll -> done -------------------------------------


def poll_until_finished(client, job_id: str, timeout: float = 5.0) -> dict:
    """Bounded retry loop mirroring the browser's own polling in app.js."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_rg_start_then_poll_matches_the_synchronous_route(client, data_dir):
    dump = data_dir / "sample_dump.dump"
    sync = client.post("/api/conformations/rg", files=files_upload(dump)).json()

    start = client.post("/api/conformations/rg/start", files=files_upload(dump))
    assert start.status_code == 202
    job_id = start.json()["job_id"]
    finished = poll_until_finished(client, job_id)
    assert finished["status"] == "done"
    assert finished["result"]["rows"] == sync["rows"]


def test_segment_acf_start_then_poll_matches_the_synchronous_route(client, data_dir):
    path = data_dir / "sample_trajectory.npz"
    form = {
        "mode": "uniform",
        "n_chains": "4",
        "beads_per_chain": "8",
        "first_index": "0",
        "time_per_frame": "1.0",
    }
    sync = client.post("/api/dynamics/segment-acf", files=files_upload(path), data=form).json()

    start = client.post("/api/dynamics/segment-acf/start", files=files_upload(path), data=form)
    assert start.status_code == 202
    finished = poll_until_finished(client, start.json()["job_id"])
    assert finished["status"] == "done"
    assert finished["result"]["rows"] == sync["rows"]
    assert finished["progress"] == 1.0


def test_segment_acf_start_accepts_a_data_file_topology(client, data_dir):
    """The data file is read before the job starts, and its workspace dies with the request.

    The background thread must therefore hold the resolved indices, not the file: if anything
    later moved that read into the thread, this is the test that would notice.
    """
    path = data_dir / "sample_trajectory.npz"
    topology = data_dir / "sample_chains_molecular.data"
    form = {"mode": "data_file", "time_per_frame": "1.0"}

    def files():
        return files_upload(path) + [("topology_file", (topology.name, topology.read_bytes()))]

    sync = client.post("/api/dynamics/segment-acf", files=files(), data=form).json()

    start = client.post("/api/dynamics/segment-acf/start", files=files(), data=form)
    assert start.status_code == 202, start.text
    finished = poll_until_finished(client, start.json()["job_id"])
    assert finished["status"] == "done"
    assert finished["result"]["rows"] == sync["rows"]
    assert finished["result"]["meta"]["topology"] == "4 chains of 8 beads, from molecule ids"


def test_segment_acf_start_rejects_a_mismatched_data_file_before_starting(
    client, data_dir, temp_root
):
    path = data_dir / "sample_trajectory.npz"
    topology = data_dir / "sample_molecular.data"
    r = client.post(
        "/api/dynamics/segment-acf/start",
        files=files_upload(path) + [("topology_file", (topology.name, topology.read_bytes()))],
        data={"mode": "data_file", "time_per_frame": "1.0"},
    )

    assert r.status_code == 422
    assert "12 atoms" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_stress_relaxation_start_then_poll_matches_the_synchronous_route(client, data_dir):
    path = data_dir / "sample_stress_acf.txt"
    form = {"volume": "1000", "temperature": "1"}
    sync = client.post(
        "/api/rheology/stress-relaxation", files=files_upload(path), data=form
    ).json()

    start = client.post(
        "/api/rheology/stress-relaxation/start", files=files_upload(path), data=form
    )
    assert start.status_code == 202
    finished = poll_until_finished(client, start.json()["job_id"])
    assert finished["status"] == "done"
    assert finished["result"]["rows"] == sync["rows"]


def test_conformations_job_reports_a_domain_error_via_polling(client, data_dir):
    """conformations has no cheap synchronous pre-check the way dynamics' topology validation
    does (which now also covers reading a data file), so a bad column is only discovered once
    the background thread runs the calculation — the /start response itself is still a 202, and
    the error surfaces through polling."""
    start = client.post(
        "/api/conformations/rg/start",
        files=files_upload(data_dir / "sample_dump.dump"),
        data={"mass_col": "mass"},
    )
    assert start.status_code == 202
    finished = poll_until_finished(client, start.json()["job_id"])
    assert finished["status"] == "error"
    assert finished["error"] == "Mass column 'mass' not found in data."


def test_job_start_leaves_no_workspace_behind_on_a_validation_error(client, data_dir, temp_root):
    r = client.post(
        "/api/dynamics/segment-acf/start",
        files=files_upload(data_dir / "sample_trajectory.npz"),
        data={
            "mode": "custom",
            "custom": "0,7\n8,40",  # index 40 is out of range for the 32-atom fixture
            "time_per_frame": "1.0",
        },
    )
    assert r.status_code == 422
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_unknown_job_id_is_404(client):
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_too_many_jobs_is_429(client, data_dir, app, monkeypatch):
    monkeypatch.setattr(app.state, "job_semaphore", threading.Semaphore(0))
    r = client.post(
        "/api/conformations/rg/start", files=files_upload(data_dir / "sample_dump.dump")
    )
    assert r.status_code == 429
    assert "Too many calculations" in r.json()["detail"]
