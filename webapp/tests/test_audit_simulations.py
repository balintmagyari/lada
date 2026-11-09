"""Regression tests for the simulation-page audit findings (D1, D9, D11)."""

import json

import pytest

from lada_web.services.dynamics_service import parse_custom_lines, resolve_chain_indices

from .conftest import files_upload
from .test_frontend_helpers import run_js

ROUSE_URL = "/api/dynamics/rouse-mode"
SEGMENT_URL = "/api/dynamics/segment-acf"
FULL_CHAINS = "\n".join(" ".join(str(8 * c + b) for b in range(8)) for c in range(4))
HEAD_TAIL = "0,7\n8,15\n16,23\n24,31"


# --- D1: Rouse modes from a custom topology need every bead ---------------------------


def test_d1_rouse_mode_refuses_head_tail_pairs(client, data_dir):
    r = client.post(
        ROUSE_URL,
        files=files_upload(data_dir / "sample_trajectory.npz"),
        data={"mode": "custom", "custom": HEAD_TAIL, "time_per_frame": "0.5", "p": "1"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "every bead" in detail
    assert "0 1 2 ... 7" in detail


def test_d1_rouse_mode_accepts_full_bead_lists(client, data_dir):
    r = client.post(
        ROUSE_URL,
        files=files_upload(data_dir / "sample_trajectory.npz"),
        data={"mode": "custom", "custom": FULL_CHAINS, "time_per_frame": "0.5", "p": "1"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["chains"] == 4


def test_d1_a_genuine_two_bead_dumbbell_is_still_a_chain():
    chains = resolve_chain_indices(
        "custom",
        n_chains=None,
        beads_per_chain=None,
        first_index=0,
        custom="0 1\n2 3",
        n_atoms=4,
    )
    assert chains.tolist() == [[0, 1], [2, 3]]


def test_d1_segment_acf_still_takes_head_tail_pairs(client, data_dir):
    r = client.post(
        SEGMENT_URL,
        files=files_upload(data_dir / "sample_trajectory.npz"),
        data={"mode": "custom", "custom": HEAD_TAIL, "time_per_frame": "0.5"},
    )
    assert r.status_code == 200, r.text


def test_d1_rouse_page_help_says_every_bead_is_needed(client):
    rouse = client.get("/dynamics/rouse-mode").text
    assert "every bead" in rouse
    assert "head,tail</code> pairs are refused" in rouse
    segment = client.get("/dynamics/segment-acf").text
    assert "only the first and" in segment
    assert "pairs are refused" not in segment


# --- D11: "0,0" is a degenerate segment, and says so ----------------------------------


def test_d11_repeated_bead_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="Line 2: bead 3 is listed more than once"):
        parse_custom_lines("0,7\n3,3")


def test_d11_zero_zero_segment_is_422_not_a_table_of_blanks(client, data_dir):
    r = client.post(
        SEGMENT_URL,
        files=files_upload(data_dir / "sample_trajectory.npz"),
        data={"mode": "custom", "custom": "0,0", "time_per_frame": "1"},
    )
    assert r.status_code == 422
    assert "listed more than once" in r.json()["detail"]


# --- D9: removing a middle regime ------------------------------------------------------

# A form whose inputs are looked up by name, and a fieldset holding the blocks that survive
# after the user added regimes 1, 2 and 3 and removed regime 2.
FORM_STUB = """(function () {
  // The shared harness's FormData keeps one value per name and has no getAll.
  global.FormData = class {
    constructor() { this.entries = []; }
    append(k, v) { this.entries.push([k, v]); }
    getAll(k) { return this.entries.filter((e) => e[0] === k).map((e) => e[1]); }
    get parts() { return Object.fromEntries(this.entries); }
  };
  const inputs = {
    files: [{ type: 'text', value: 'base' }],
    regime_1_files: [{ type: 'text', value: 'one' }],
    regime_3_files: [{ type: 'text', value: 'three' }],
    shared: [{ type: 'text', value: 's' }],
  };
  const form = {
    querySelectorAll: (sel) => inputs[sel.match(/name="([^"]+)"/)[1]] || [],
  };
  const blocks = [{ dataset: { regimeKey: '1' } }, { dataset: { regimeKey: '3' } }];
  const fieldset = { querySelectorAll: () => blocks };
  const button = { closest: (sel) => (sel === 'fieldset.regimes' ? fieldset : null) };
  const keys = hooks.regimeKeys(button);
  return {
    keys: keys,
    parts: keys.map((k) => hooks.regimeFormData(form, k, ['files'], ['shared']).parts),
  };
})()"""


def test_d9_regimes_are_found_by_key_so_a_middle_one_can_be_removed():
    out = run_js(FORM_STUB)
    assert out["keys"] == [None, "1", "3"]
    assert [p["files"] for p in out["parts"]] == ["base", "one", "three"]
    assert all(p["shared"] == "s" for p in out["parts"])


def test_d9_regime_blocks_carry_their_key_for_the_script(client):
    page = client.get("/dynamics/rouse-mode").text
    assert ':data-regime-key="regime"' in page
    assert "'regime_' + regime + '_files'" in page
    assert json.dumps("regime-count") not in page and "data-regime-count" not in page
