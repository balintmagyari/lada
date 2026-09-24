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

from .conftest import files_upload

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


DATA_FILE = {"mode": "data_file"}


def topology_upload(data_dir: Path, name: str):
    path = data_dir / name
    return ("topology_file", (path.name, path.read_bytes()))


def data_file_post(client, data_dir, url, name="sample_chains_molecular.data", **extra):
    """A dynamics request whose chain topology comes from an uploaded data file."""
    files = files_upload(trajectory(data_dir)) + [topology_upload(data_dir, name)]
    return client.post(url, files=files, data=DATA_FILE | {"time_per_frame": "1.0"} | extra)


# --- Topology resolution ---------------------------------------------------


def test_uniform_chain_indices_are_consecutive_blocks():
    indices = resolve_chain_indices(
        "uniform", n_chains=2, beads_per_chain=3, first_index=10, custom="", n_atoms=16
    )
    assert indices.tolist() == [[10, 11, 12], [13, 14, 15]]


def test_uniform_topology_that_covers_only_part_of_the_file_is_rejected():
    # A mistyped chain count used to compute quietly on a fraction of the system and report
    # success, which is worse than an error: the result looks ordinary and is wrong.
    with pytest.raises(ValueError, match="covers 6 atoms"):
        resolve_chain_indices(
            "uniform", n_chains=2, beads_per_chain=3, first_index=0, custom="", n_atoms=32
        )


def test_uniform_segment_pairs_are_the_block_ends():
    pairs = resolve_segment_pairs(
        "uniform", n_chains=2, beads_per_chain=3, first_index=10, custom="", n_atoms=16
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
    r = client.post(SEGMENT_URL, files=files_upload(path), data=UNIFORM | {"time_per_frame": "2.0"})
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
    r = client.post(SEGMENT_URL, files=files_upload(path), data=form)
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
    r = client.post(SEGMENT_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    assert "32 atoms" in r.json()["detail"]


def test_segment_acf_htmx_partial_carries_the_plot(client, data_dir, htmx):
    r = client.post(
        SEGMENT_URL,
        files=files_upload(trajectory(data_dir)),
        data=UNIFORM | {"time_per_frame": "2.0"},
        headers=htmx,
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="result-plot"' in r.text
    assert 'formaction="/api/dynamics/segment-acf/download?fmt=csv"' in r.text


def test_segment_acf_csv_download_round_trips(client, data_dir):
    path = trajectory(data_dir)
    form = UNIFORM | {"time_per_frame": "2.0"}
    view = client.post(SEGMENT_URL, files=files_upload(path), data=form).json()
    r = client.post(SEGMENT_URL + "/download?fmt=csv", files=files_upload(path), data=form)
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


def test_segment_acf_multiple_files_averages_across_repeats(client, data_dir):
    path = trajectory(data_dir)
    repeat2 = data_dir / "sample_trajectory_repeat2.npz"
    form = UNIFORM | {"time_per_frame": "2.0"}
    single = client.post(SEGMENT_URL, files=files_upload(path), data=form).json()
    combined = client.post(SEGMENT_URL, files=files_upload(path, repeat2), data=form).json()
    assert combined["meta"]["n_files"] == 2
    assert combined["download_stem"] == "sample_trajectory_avg2_segment_acf"
    # C(0) is normalized to 1.0 by every input, so the average is still exactly 1.0 at lag 0.
    assert combined["rows"][0] == [0.0, 1.0]
    # A different, independently-seeded second file changes the averaged curve from the
    # single-file curve past lag 0 (statistically certain not to coincide by chance).
    assert combined["rows"][1] != single["rows"][1]


def test_segment_acf_rejects_files_with_different_atom_counts(client, data_dir, tmp_path):
    mismatched = write_npz(tmp_path / "mismatched.npz", n_frames=32, n_atoms=16)
    form = UNIFORM | {"time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=files_upload(trajectory(data_dir), mismatched), data=form)
    assert r.status_code == 422
    assert "same number of atoms" in r.json()["detail"]


def test_segment_acf_needs_a_positive_frame_time(client, data_dir):
    form = UNIFORM | {"time_per_frame": "0"}
    r = client.post(SEGMENT_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422


# --- Rouse mode route ------------------------------------------------------


def test_rouse_mode_acf_starts_at_one(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "1"}
    r = client.post(ROUSE_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "acf"]
    assert body["n_rows_total"] == 32
    assert body["rows"][0] == pytest.approx([0.0, 1.0])
    assert body["download_stem"] == "sample_trajectory_rouse_p1"


def test_rouse_mode_forwards_the_library_message_for_a_mode_that_is_too_high(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "8"}
    r = client.post(ROUSE_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "p=8" in detail and "N=8" in detail


def test_rouse_mode_pgfplots_download_names_the_mode(client, data_dir):
    form = UNIFORM | {"time_per_frame": "1.0", "p": "2"}
    r = client.post(
        ROUSE_URL + "/download?fmt=pgfplots", files=files_upload(trajectory(data_dir)), data=form
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
    r = client.post(ISF_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["lag_time", "isf"]
    assert body["n_rows_total"] == 32
    assert body["rows"][0] == pytest.approx([0.0, 1.0])
    assert body["download_stem"] == "sample_trajectory_isf"
    assert body["meta"]["frames"] == 32 and body["meta"]["atoms"] == 32


def test_isf_needs_a_positive_q(client, data_dir):
    form = {"time_per_frame": "1.0", "q_magnitude": "0", "n_vectors": "5"}
    r = client.post(ISF_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422


def test_isf_rejects_a_single_q_direction(client, data_dir):
    # lada spreads the directions over n_vectors - 1 steps, so a single one has no meaning.
    form = {"time_per_frame": "1.0", "q_magnitude": "1.0", "n_vectors": "1"}
    r = client.post(ISF_URL, files=files_upload(trajectory(data_dir)), data=form)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("n_vectors" in error["loc"] for error in detail)


def test_isf_csv_download(client, data_dir):
    form = {"time_per_frame": "1.0", "q_magnitude": "1.0", "n_vectors": "5"}
    r = client.post(
        ISF_URL + "/download?fmt=csv", files=files_upload(trajectory(data_dir)), data=form
    )
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_trajectory_isf.csv"'
    df = pd.read_csv(io.BytesIO(r.content), float_precision="round_trip")
    assert list(df.columns) == ["lag_time", "isf"] and len(df) == 32


# --- Guardrails ------------------------------------------------------------


def test_too_many_frames_is_413(client, tmp_path):
    path = write_npz(tmp_path / "long.npz", n_frames=101, n_atoms=32)
    form = UNIFORM | {"time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=files_upload(path), data=form)
    assert r.status_code == 413
    assert "at most 100" in r.json()["detail"]


def test_oversized_trajectory_is_413(client, tmp_path, temp_root):
    path = write_npz(tmp_path / "big.npz", n_frames=50, n_atoms=1050)
    assert path.stat().st_size > 1024 * 1024
    form = UNIFORM | {"time_per_frame": "1.0"}
    r = client.post(SEGMENT_URL, files=files_upload(path), data=form)
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
    """max follows the active mode, and is dropped when no bead count is known yet.

    Binding it to a null bead count would render max="-1", making p permanently invalid and
    blocking the download buttons, which submit this same form.
    """
    r = client.get("/dynamics/rouse-mode")
    assert ':max="maxP"' in r.text
    assert "this.activeBeads ? this.activeBeads - 1 : false" in r.text
    assert 'name="p"' in r.text


@pytest.mark.parametrize("slug", ["segment-acf", "rouse-mode"])
def test_topology_pages_offer_the_data_file_mode(client, slug):
    r = client.get(f"/dynamics/{slug}")

    assert 'value="data_file"' in r.text
    assert 'name="topology_file"' in r.text
    assert "x-show=\"mode === 'data_file'\" x-cloak" in r.text
    assert "window.ladaReadTopology($event, $data)" in r.text


@pytest.mark.parametrize("slug", ["segment-acf", "rouse-mode"])
def test_every_required_topology_input_is_disabled_outside_its_own_mode(client, slug):
    """The download buttons submit this form, so a hidden `required` input would silently
    block them: the browser refuses to submit and cannot focus the field to say why."""
    r = client.get(f"/dynamics/{slug}")

    for name, mode in [
        ("n_chains", "uniform"),
        ("beads_per_chain", "uniform"),
        ("first_index", "uniform"),
        ("topology_file", "data_file"),
    ]:
        field = r.text.split(f'name="{name}"', 1)[1].split(">", 1)[0]
        assert "required" in field, name
        assert f":disabled=\"mode !== '{mode}'\"" in field, name


@pytest.mark.parametrize("slug", ["segment-acf", "rouse-mode"])
def test_the_data_file_travels_with_every_regime(client, slug):
    """Topology is shared across regimes, so each per-regime request has to carry the file."""
    r = client.get(f"/dynamics/{slug}")
    shared = r.text.split('data-shared-fields="', 1)[1].split('"', 1)[0]

    assert "topology_file" in shared.split(",")


def test_segment_acf_page_has_no_mode_number_field(client):
    """The segment ACF route takes no `p`; a required input for it would block the form."""
    r = client.get("/dynamics/segment-acf")
    assert r.status_code == 200
    assert 'name="p"' not in r.text


# --- Topology from an uploaded data file -----------------------------------


def test_data_file_topology_matches_the_hand_typed_uniform_form(client, data_dir):
    """The anchor: reading the topology must change nothing but how it was entered.

    The fixture data file describes exactly the layout UNIFORM spells out, so any difference
    here is the reader mis-numbering beads rather than a difference of opinion about physics.
    """
    typed = client.post(
        SEGMENT_URL,
        files=files_upload(trajectory(data_dir)),
        data=UNIFORM | {"time_per_frame": "1.0"},
    )
    read = data_file_post(client, data_dir, SEGMENT_URL)

    assert read.status_code == 200, read.text
    assert read.json()["rows"] == typed.json()["rows"]
    assert read.json()["meta"]["chains"] == 4


def test_data_file_topology_says_where_the_chains_came_from(client, data_dir):
    meta = data_file_post(client, data_dir, SEGMENT_URL).json()["meta"]

    assert meta["topology"] == "4 chains of 8 beads, from molecule ids"
    assert "topology notes" not in meta


def test_rouse_mode_accepts_a_data_file_topology(client, data_dir):
    typed = client.post(
        ROUSE_URL,
        files=files_upload(trajectory(data_dir)),
        data=UNIFORM | {"time_per_frame": "1.0", "p": "1"},
    )
    read = data_file_post(client, data_dir, ROUSE_URL, p="1")

    assert read.status_code == 200, read.text
    assert read.json()["rows"] == typed.json()["rows"]


def test_chains_can_come_from_the_bond_graph_alone(client, data_dir):
    """The bonds-only fixture has one chain whose ids run out of backbone order."""
    r = data_file_post(client, data_dir, SEGMENT_URL, name="sample_chains_bonds_only.data")

    assert r.status_code == 200, r.text
    assert r.json()["meta"]["topology"] == "4 chains of 8 beads, from the bond graph"


def test_a_walked_chain_is_not_the_same_as_an_id_sorted_one(client, data_dir):
    """Proof the walk did the work: its answer differs from the naive one, on real routes."""
    walked = data_file_post(client, data_dir, SEGMENT_URL, name="sample_chains_bonds_only.data")
    sorted_ids = data_file_post(client, data_dir, SEGMENT_URL)

    assert walked.json()["rows"] != sorted_ids.json()["rows"]


def test_a_data_file_for_a_different_system_is_refused(client, data_dir):
    r = data_file_post(client, data_dir, SEGMENT_URL, name="sample_molecular.data")

    assert r.status_code == 422
    assert "12 atoms" in r.json()["detail"]
    assert "32" in r.json()["detail"]


def test_data_file_mode_without_a_file_asks_for_one(client, data_dir):
    """An empty file input arrives as no field at all, which must not read as a broken file."""
    r = client.post(
        SEGMENT_URL,
        files=files_upload(trajectory(data_dir)),
        data=DATA_FILE | {"time_per_frame": "1.0"},
    )

    assert r.status_code == 422
    assert "Choose a LAMMPS data file" in r.json()["detail"]


def test_a_file_with_neither_molecule_ids_nor_bonds_is_refused(client, data_dir, tmp_path):
    plain = tmp_path / "plain.data"
    plain.write_text(
        "no topology here\n\n2 atoms\n\n1 atom types\n\n0.0 1.0 xlo xhi\n0.0 1.0 ylo yhi\n"
        "0.0 1.0 zlo zhi\n\nAtoms # atomic\n\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n"
    )
    files = files_upload(trajectory(data_dir)) + [
        ("topology_file", ("plain.data", plain.read_bytes()))
    ]
    r = client.post(SEGMENT_URL, files=files, data=DATA_FILE | {"time_per_frame": "1.0"})

    assert r.status_code == 422
    assert "no molecule id" in r.json()["detail"]


def test_ragged_chains_run_a_segment_acf_but_not_a_rouse_mode(client, data_dir, tmp_path):
    """Polydisperse melts are ordinary; only the Rouse analysis genuinely cannot take them."""
    rows = [f"{i} 1 1 {i}.0 0.0 0.0" for i in range(1, 21)]
    rows += [f"{i} 2 1 {i}.0 0.0 0.0" for i in range(21, 33)]
    ragged = tmp_path / "ragged.data"
    ragged.write_text(
        "two chains of different length\n\n32 atoms\n\n1 atom types\n\n"
        "0.0 1.0 xlo xhi\n0.0 1.0 ylo yhi\n0.0 1.0 zlo zhi\n\n"
        "Atoms # molecular\n\n" + "\n".join(rows) + "\n"
    )
    upload = ("topology_file", ("ragged.data", ragged.read_bytes()))
    files = files_upload(trajectory(data_dir)) + [upload]

    segment = client.post(SEGMENT_URL, files=files, data=DATA_FILE | {"time_per_frame": "1.0"})
    assert segment.status_code == 200, segment.text
    assert segment.json()["meta"]["topology"] == "2 chains of varying lengths, from molecule ids"

    files = files_upload(trajectory(data_dir)) + [upload]
    rouse = client.post(
        ROUSE_URL, files=files, data=DATA_FILE | {"time_per_frame": "1.0", "p": "1"}
    )
    assert rouse.status_code == 422
    assert "same number of beads" in rouse.json()["detail"]


def test_a_guessed_bead_order_is_reported_in_the_result(client, data_dir, tmp_path):
    """A file without bonds cannot confirm the order, so the result has to say so."""
    rows = [f"{i} {1 + (i - 1) // 8} 1 {i}.0 0.0 0.0" for i in range(1, 33)]
    no_bonds = tmp_path / "nobonds.data"
    no_bonds.write_text(
        "molecule ids but no bonds\n\n32 atoms\n\n1 atom types\n\n"
        "0.0 1.0 xlo xhi\n0.0 1.0 ylo yhi\n0.0 1.0 zlo zhi\n\n"
        "Atoms # molecular\n\n" + "\n".join(rows) + "\n"
    )
    files = files_upload(trajectory(data_dir)) + [
        ("topology_file", ("nobonds.data", no_bonds.read_bytes()))
    ]
    r = client.post(SEGMENT_URL, files=files, data=DATA_FILE | {"time_per_frame": "1.0"})

    assert r.status_code == 200, r.text
    assert "atom-id order" in r.json()["meta"]["topology notes"]


def test_data_file_topology_downloads(client, data_dir):
    files = files_upload(trajectory(data_dir)) + [
        topology_upload(data_dir, "sample_chains_molecular.data")
    ]
    r = client.post(
        f"{SEGMENT_URL}/download?fmt=csv", files=files, data=DATA_FILE | {"time_per_frame": "1.0"}
    )

    assert r.status_code == 200, r.text
    frame = pd.read_csv(io.StringIO(r.text))
    assert list(frame.columns) == ["lag_time", "acf"]
    assert len(frame) == 32


def test_data_file_topology_leaves_no_workspace_behind(client, data_dir, temp_root):
    r = data_file_post(client, data_dir, SEGMENT_URL)

    assert r.status_code == 200
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_a_bad_data_file_leaves_no_workspace_behind(client, data_dir, temp_root):
    r = data_file_post(client, data_dir, SEGMENT_URL, name="sample_molecular.data")

    assert r.status_code == 422
    assert not temp_root.exists() or list(temp_root.iterdir()) == []
