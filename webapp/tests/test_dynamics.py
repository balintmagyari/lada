import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lada.analysis import (
    calculate_segment_acf_from_trajectory,
)

from lada_web.errors import TrajectoryTooLarge
from lada_web.services.dynamics_service import (
    inspect_trajectory,
    parse_custom_lines,
    resolve_chain_indices,
    resolve_segment_pairs,
)

from .conftest import upload

SEGMENT_URL = "/api/dynamics/segment-acf"
ROUSE_URL = "/api/dynamics/rouse-mode"
ISF_URL = "/api/dynamics/isf"

# The fixture holds 4 chains of 8 beads, chain-major: atom index = chain * 8 + bead.
UNIFORM = {"mode": "uniform", "n_chains": "4", "beads_per_chain": "8", "first_index": "0"}
FIXTURE_PAIRS = np.array([[0, 7], [8, 15], [16, 23], [24, 31]])

# Decompressed-size budget for the direct inspect_trajectory calls; the cases that exercise
# the budget itself live in test_resource_limits.py.
MEMORY_BUDGET = 8 * 1024 * 1024


def trajectory(data_dir: Path) -> Path:
    return data_dir / "sample_trajectory.npz"


def write_npz(path: Path, n_frames: int, n_atoms: int) -> Path:
    coords = np.arange(n_frames * n_atoms * 3, dtype=float).reshape(n_frames, n_atoms, 3)
    np.savez(path, coords=coords)
    return path


# --- Topology resolution ---------------------------------------------------


def test_uniform_chain_indices_are_consecutive_blocks():
    indices = resolve_chain_indices(
        "uniform", n_chains=2, beads_per_chain=3, first_index=10, custom="", n_atoms=32
    )
    assert indices.tolist() == [[10, 11, 12], [13, 14, 15]]


def test_uniform_segment_pairs_are_the_block_ends():
    pairs = resolve_segment_pairs(
        "uniform", n_chains=2, beads_per_chain=3, first_index=10, custom="", n_atoms=32
    )
    assert pairs.tolist() == [[10, 12], [13, 15]]


def test_custom_pairs_ignore_blank_and_comment_lines():
    pairs = resolve_segment_pairs(
        "custom",
        n_chains=None,
        beads_per_chain=None,
        first_index=0,
        custom="0, 7\n8 15\n\n# comment\n",
        n_atoms=32,
    )
    assert pairs.tolist() == [[0, 7], [8, 15]]


def test_custom_full_bead_lists_give_indices_and_pairs():
    text = "0 1 2 3\n4 5 6 7"
    indices = resolve_chain_indices(
        "custom", n_chains=None, beads_per_chain=None, first_index=0, custom=text, n_atoms=32
    )
    assert indices.shape == (2, 4)
    assert indices.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    pairs = resolve_segment_pairs(
        "custom", n_chains=None, beads_per_chain=None, first_index=0, custom=text, n_atoms=32
    )
    assert pairs.tolist() == [[0, 3], [4, 7]]


def test_custom_non_numeric_line_names_the_line_number():
    with pytest.raises(ValueError) as excinfo:
        parse_custom_lines("0,7\nfoo,bar")
    assert str(excinfo.value).startswith("Line 2:")


def test_custom_single_index_line_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        parse_custom_lines("0,7\n8")
    assert str(excinfo.value).startswith("Line 2:")


def test_custom_negative_index_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        parse_custom_lines("0,-7")
    assert str(excinfo.value).startswith("Line 1:")


def test_custom_empty_text_is_rejected():
    with pytest.raises(ValueError, match="one chain per line"):
        parse_custom_lines("\n# nothing here\n")


def test_custom_chains_of_unequal_length_are_rejected_for_rouse_modes():
    with pytest.raises(ValueError, match="same number of beads"):
        resolve_chain_indices(
            "custom",
            n_chains=None,
            beads_per_chain=None,
            first_index=0,
            custom="0 1 2\n3 4",
            n_atoms=32,
        )


def test_uniform_without_counts_asks_for_both():
    with pytest.raises(ValueError) as excinfo:
        resolve_chain_indices(
            "uniform", n_chains=None, beads_per_chain=8, first_index=0, custom="", n_atoms=32
        )
    message = str(excinfo.value)
    assert "number of chains" in message and "beads per chain" in message


# --- Trajectory inspection -------------------------------------------------


def test_inspect_trajectory_reads_the_shape_of_the_fixture(data_dir):
    assert inspect_trajectory(trajectory(data_dir), 100, MEMORY_BUDGET) == (32, 32)


def test_inspect_trajectory_rejects_too_many_frames(tmp_path):
    path = write_npz(tmp_path / "long.npz", n_frames=101, n_atoms=4)
    with pytest.raises(TrajectoryTooLarge, match="at most 100"):
        inspect_trajectory(path, 100, MEMORY_BUDGET)


def test_inspect_trajectory_needs_a_coords_array(tmp_path):
    path = tmp_path / "wrong_key.npz"
    np.savez(path, other=np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="'coords'"):
        inspect_trajectory(path, 100, MEMORY_BUDGET)


def test_inspect_trajectory_rejects_a_file_that_is_not_an_archive(tmp_path):
    path = tmp_path / "not_a_trajectory.npz"
    path.write_text("timestep 0\n")
    with pytest.raises(ValueError, match=r"\.npz"):
        inspect_trajectory(path, 100, MEMORY_BUDGET)


def test_inspect_trajectory_rejects_the_wrong_shape(tmp_path):
    path = tmp_path / "flat.npz"
    np.savez(path, coords=np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"n_frames, n_atoms, 3"):
        inspect_trajectory(path, 100, MEMORY_BUDGET)


# --- Segment ACF route -----------------------------------------------------


def test_segment_acf_uniform_matches_the_library(client, data_dir):
    path = trajectory(data_dir)
    r = client.post(SEGMENT_URL, files=upload(path), data=UNIFORM | {"time_per_frame": "2.0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "acf"]
    assert body["n_rows_total"] == 32
    assert body["rows"][0] == [0.0, 1.0]
    assert body["download_stem"] == "sample_trajectory_segment_acf"
    assert body["meta"]["frames"] == 32 and body["meta"]["atoms"] == 32
    assert body["meta"]["chains"] == 4
    assert body["plot"]["x"] == "lag_time" and body["plot"]["y"] == ["acf"]
    assert body["plot"]["x_log"] is False and body["plot"]["y_log"] is False
    expected = calculate_segment_acf_from_trajectory(str(path), FIXTURE_PAIRS, 2.0)
    # The on-screen table is capped by max_table_rows (20 in the test settings); the plot
    # series always carry every frame.
    assert len(body["rows"]) == 20 and body["truncated"] is True
    assert np.allclose(np.array(body["rows"], dtype=float), expected[:20])
    assert np.allclose(body["plot"]["series"]["lag_time"], expected[:, 0])
    assert np.allclose(body["plot"]["series"]["acf"], expected[:, 1])


def test_segment_acf_custom_matches_the_library(client, data_dir):
    path = trajectory(data_dir)
    form = {"mode": "custom", "custom": "0,7\n8,15", "time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=upload(path), data=form)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["chains"] == 2
    expected = calculate_segment_acf_from_trajectory(str(path), np.array([[0, 7], [8, 15]]), 1.0)
    assert body["n_rows_total"] == 32
    assert np.allclose(np.array(body["rows"], dtype=float), expected[: len(body["rows"])])
    assert np.allclose(body["plot"]["series"]["acf"], expected[:, 1])


@pytest.mark.parametrize(
    "form",
    [
        {"mode": "custom", "custom": "0,7\n8,40", "time_per_frame": "1.0"},
        # 4 chains x 8 beads starting at index 8 runs to index 39; the fixture has 32 atoms.
        UNIFORM | {"first_index": "8", "time_per_frame": "1.0"},
    ],
    ids=["custom", "uniform"],
)
def test_segment_acf_rejects_an_index_beyond_the_last_atom(client, data_dir, form):
    r = client.post(SEGMENT_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    assert "32 atoms" in r.json()["detail"]


def test_segment_acf_htmx_partial_carries_the_plot(client, data_dir, htmx):
    r = client.post(
        SEGMENT_URL,
        files=upload(trajectory(data_dir)),
        data=UNIFORM | {"time_per_frame": "2.0"},
        headers=htmx,
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="result-plot"' in r.text
    assert 'formaction="/api/dynamics/segment-acf/download?fmt=csv"' in r.text


def test_segment_acf_csv_download_round_trips(client, data_dir):
    path = trajectory(data_dir)
    form = UNIFORM | {"time_per_frame": "2.0"}
    view = client.post(SEGMENT_URL, files=upload(path), data=form).json()
    r = client.post(SEGMENT_URL + "/download?fmt=csv", files=upload(path), data=form)
    assert r.status_code == 200
    assert (
        r.headers["content-disposition"]
        == 'attachment; filename="sample_trajectory_segment_acf.csv"'
    )
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert list(df.columns) == view["columns"]
    assert len(df) == view["n_rows_total"] == 32
    # Exact, not allclose: a download must carry the same float64 the JSON view carries.
    assert np.array_equal(df.to_numpy()[: len(view["rows"])], np.array(view["rows"], dtype=float))


def test_segment_acf_needs_a_positive_frame_time(client, data_dir):
    form = UNIFORM | {"time_per_frame": "0"}
    r = client.post(SEGMENT_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422


# --- Rouse mode route ------------------------------------------------------


def test_rouse_mode_acf_starts_at_one(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "1"}
    r = client.post(ROUSE_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "acf"]
    assert body["n_rows_total"] == 32
    assert body["rows"][0] == pytest.approx([0.0, 1.0])
    assert body["download_stem"] == "sample_trajectory_rouse_p1"


def test_rouse_mode_forwards_the_library_message_for_a_mode_that_is_too_high(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "8"}
    r = client.post(ROUSE_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "p=8" in detail and "N=8" in detail


def test_rouse_mode_pgfplots_download_names_the_mode(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "2"}
    r = client.post(
        ROUSE_URL + "/download?fmt=pgfplots", files=upload(trajectory(data_dir)), data=form
    )
    assert r.status_code == 200
    assert (
        r.headers["content-disposition"] == 'attachment; filename="sample_trajectory_rouse_p2.dat"'
    )
    assert r.text.splitlines()[0] == "lag_time acf"
    assert np.loadtxt(io.BytesIO(r.content), skiprows=1).shape == (32, 2)


# --- ISF route -------------------------------------------------------------


def test_isf_starts_at_one(client, data_dir):
    form = {"time_per_frame": "1.0", "q_magnitude": "1.0", "n_vectors": "5"}
    r = client.post(ISF_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "isf"]
    assert body["n_rows_total"] == 32
    assert body["rows"][0] == pytest.approx([0.0, 1.0])
    assert body["download_stem"] == "sample_trajectory_isf"
    assert body["meta"]["frames"] == 32 and body["meta"]["atoms"] == 32


def test_isf_needs_a_positive_q(client, data_dir):
    form = {"time_per_frame": "1.0", "q_magnitude": "0", "n_vectors": "5"}
    r = client.post(ISF_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422


def test_isf_rejects_a_single_q_direction(client, data_dir):
    # lada spreads the directions over n_vectors - 1 steps, so a single one has no meaning.
    form = {"time_per_frame": "1.0", "q_magnitude": "1.0", "n_vectors": "1"}
    r = client.post(ISF_URL, files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("n_vectors" in error["loc"] for error in detail)


def test_isf_csv_download(client, data_dir):
    form = {"time_per_frame": "1.0", "q_magnitude": "1.0", "n_vectors": "5"}
    r = client.post(ISF_URL + "/download?fmt=csv", files=upload(trajectory(data_dir)), data=form)
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_trajectory_isf.csv"'
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert list(df.columns) == ["lag_time", "isf"] and len(df) == 32


# --- Guardrails ------------------------------------------------------------


def test_too_many_frames_is_413(client, tmp_path):
    path = write_npz(tmp_path / "long.npz", n_frames=101, n_atoms=32)
    form = UNIFORM | {"time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=upload(path), data=form)
    assert r.status_code == 413
    assert "at most 100" in r.json()["detail"]


def test_oversized_trajectory_is_413(client, tmp_path, temp_root):
    path = write_npz(tmp_path / "big.npz", n_frames=50, n_atoms=1050)
    assert path.stat().st_size > 1024 * 1024
    form = UNIFORM | {"time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=upload(path), data=form)
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


# --- Pages -----------------------------------------------------------------


@pytest.mark.parametrize("slug", ["segment-acf", "rouse-mode", "isf"])
def test_page_renders(client, slug):
    r = client.get(f"/dynamics/{slug}")
    assert r.status_code == 200
    assert 'id="calc-form"' in r.text
    assert 'name="time_per_frame"' in r.text
    assert "10–20" in r.text


@pytest.mark.parametrize("slug", ["segment-acf", "rouse-mode"])
def test_topology_pages_carry_the_alpine_toggle(client, slug):
    r = client.get(f"/dynamics/{slug}")
    assert r.status_code == 200
    assert "x-data" in r.text
    assert 'name="mode"' in r.text
    assert 'name="custom"' in r.text
    assert "Atom order is assumed to match ascending atom ID" in r.text
    assert "x-show=\"mode === 'custom'\" x-cloak" in r.text


def test_rouse_page_binds_the_mode_number_to_the_bead_count(client):
    r = client.get("/dynamics/rouse-mode")
    assert ':max="beads - 1"' in r.text
    assert 'name="p"' in r.text


def test_segment_acf_page_has_no_mode_number_field(client):
    """The segment ACF route takes no `p`; a required input for it would block the form."""
    r = client.get("/dynamics/segment-acf")
    assert r.status_code == 200
    assert 'name="p"' not in r.text
