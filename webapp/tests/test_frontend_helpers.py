"""The page's own JavaScript, exercised in Node.

Every other test here stops at the HTTP response. Both frontend bugs that reached a user lived
past that line — a result the server sent correctly but the page could not display, and a
validation message the server worded well that the page rendered as "[object Object]" — so the
two pure helpers behind them are run for real rather than reviewed by eye. Skipped when Node is
absent, which keeps `pip install -e ./webapp[dev]` free of a Node dependency.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "src" / "lada_web" / "static" / "js" / "app.js"

# app.js wires up navigation and htmx hooks as it loads; these stubs are the least that lets it
# run headless. They deliberately do nothing: only the pure string helpers are under test.
HARNESS = """
const fs = require('fs');
const noop = () => {};
const element = () => ({
  addEventListener: noop, querySelector: () => null, querySelectorAll: () => [],
  classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
  dataset: {}, style: {}, setAttribute: noop, removeAttribute: noop, appendChild: noop,
  getAttribute: () => null, focus: noop, closest: () => null, contains: () => false,
});
global.document = {
  addEventListener: noop, createElement: element, getElementById: () => null,
  querySelector: () => null, querySelectorAll: () => [], body: element(),
  documentElement: element(), readyState: 'complete',
};
global.window = {
  addEventListener: noop,
  matchMedia: () => ({ matches: false, addEventListener: noop }),
};
global.fetch = () => Promise.reject(new Error('not used'));
global.FormData = class { constructor() { this.parts = {}; } append(k, v) { this.parts[k] = v; } };
eval(fs.readFileSync(%s, 'utf8'));
const hooks = global.window.ladaTestHooks;
Promise.resolve(%s).then((v) => console.log(JSON.stringify(v)));
"""

# A file input holding one file, plus a fetch that answers with the given ok flag and body.
# One expression, because it is substituted into Promise.resolve(...) in HARNESS.
PREVIEW_HARNESS = """(function () {
  const state = { preview: null, previewError: '', previewing: false };
  const event = { target: { files: [{ name: 'system.data' }] } };
  const fetcher = () => %s;
  return hooks.readTopology(event, state, fetcher).then(function () { return state; });
})()"""

ANSWER = "Promise.resolve({ ok: %s, json: () => Promise.resolve(%s) })"


def run_js(expression: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = HARNESS % (json.dumps(str(APP_JS)), expression)
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_escape_html_escapes_quotes_so_result_json_survives_an_attribute():
    """The bug: innerHTML serialization leaves `"` alone, so the JSON ended at its first quote.

    A job's result reaches the page inside data-result="...", and JSON is mostly quotes.
    """
    escaped = run_js('hooks.escapeHtml(JSON.stringify({ title: "a", rows: [1] }))')
    assert '"' not in escaped
    assert "&quot;" in escaped


def test_escape_html_still_escapes_markup():
    assert run_js('hooks.escapeHtml("<script>x&y</script>")') == (
        "&lt;script&gt;x&amp;y&lt;/script&gt;"
    )


def test_detail_message_unpacks_a_pydantic_error_list():
    """The bug: FastAPI sends `detail` as a list of objects; String(list) is "[object Object]"."""
    body = {
        "detail": [
            {"type": "greater_than", "loc": ["body", "volume"], "msg": "Input should be > 0"},
            {"type": "greater_than", "loc": ["body", "temperature"], "msg": "Input should be > 0"},
        ]
    }
    message = run_js(f"hooks.detailMessage({json.dumps(body)}, 'fallback')")
    assert message == "volume: Input should be > 0; temperature: Input should be > 0"
    assert "object Object" not in message


def test_detail_message_passes_a_plain_string_through():
    body = {"detail": "The trajectory has 201 frames; this server accepts at most 50."}
    assert run_js(f"hooks.detailMessage({json.dumps(body)}, 'fallback')") == body["detail"]


def test_detail_message_falls_back_when_there_is_no_detail():
    assert run_js("hooks.detailMessage({}, 'Could not start the calculation.')") == (
        "Could not start the calculation."
    )


def preview_state(ok: bool, body: object) -> dict:
    answer = ANSWER % ("true" if ok else "false", json.dumps(body))
    return run_js(PREVIEW_HARNESS % answer)


def test_reading_a_topology_file_fills_the_preview():
    body = {
        "n_atoms": 32,
        "n_chains": 4,
        "beads_per_chain": 8,
        "summary": "4 chains of 8 beads, from molecule ids",
        "notes": [],
    }
    state = preview_state(True, body)

    assert state["preview"] == body
    assert state["previewError"] == ""
    assert state["previewing"] is False


def test_a_refused_topology_file_shows_the_servers_reason():
    """The refusals name what the file is missing; showing a generic message throws that away."""
    detail = "This file cannot say which atoms form a chain: no molecule id, and no Bonds section."
    state = preview_state(False, {"detail": detail})

    assert state["previewError"] == detail
    assert state["preview"] is None
    assert state["previewing"] is False


def test_the_spinner_is_cleared_even_when_the_request_fails_outright():
    """Without the final `.then`, a dropped connection would leave "Reading…" on screen forever."""
    state = run_js(PREVIEW_HARNESS % "Promise.reject(new Error('offline'))")

    assert state["previewing"] is False
    assert state["previewError"] == "This data file could not be read."


def test_clearing_the_file_input_clears_the_stale_preview():
    """Otherwise an earlier file's chain count stays on screen next to an empty input."""
    state = run_js(
        "(function () {"
        "  var s = { preview: { n_atoms: 1 }, previewError: 'old', previewing: true };"
        "  return hooks.readTopology({ target: { files: [] } }, s).then(function () { return s; });"
        "})()"
    )

    assert state["preview"] is None
    assert state["previewError"] == ""
