# lada-web

Browser front-end for the `lada` package: a calculation tool, not a 3D trajectory/molecule
viewer. Upload a LAMMPS output file, pick a calculation, see the result as a table (and a chart
where one applies), download it as CSV or a pgfplots-ready `.dat` file. Every calculation is a
thin web layer over the functions already documented in the repository root `README.md` — this
document only covers the parts specific to running that functionality as a service.

> **PyPI pin caveat.** `webapp/pyproject.toml` depends on `lada>=1.2.0,<2.0.0`, matching the
> functions this app actually calls (`calculate_segment_acf_from_trajectory` and friends, added in
> 1.2.0). As of this writing PyPI's latest published `lada` release is **1.0.6** — 1.2.0 exists only
> as the unreleased local package in this repository. `pip install .` and the Docker build below
> will fail until `lada` 1.2.0 is published to PyPI. Local development works today because
> `pip install -e . -e "./webapp[dev]"` from the repo root installs `lada` from `src/` in editable
> mode, bypassing the PyPI resolution entirely.

## Quick start

From the repository root, with the same virtual environment used for `lada` itself:

```bash
pip install -e . -e "./webapp[dev]"
uvicorn lada_web.main:app --reload
```

Open `http://127.0.0.1:8000`. Every page lists its own required file and fields; there is no
separate configuration step for local use — the defaults in [Configuration](#configuration) apply
automatically.

## Pages

Thirteen calculations across five sections, each a single page with one upload form. The file
column names the multipart field the page's form (and the matching `/api/...` route) expects.

### Read files (`/parsers/...`)

| Page | Route | File it needs |
|---|---|---|
| Dump trajectory | `/parsers/dump` | a LAMMPS dump trajectory file |
| Thermo log | `/parsers/log` | a LAMMPS log file |
| Data file | `/parsers/data` | a LAMMPS data file (`write_data` output) |
| Stress ACF file | `/parsers/acf` | a `fix ave/correlate/long` stress-ACF output file |

The dump page offers a fast per-frame **preview** (`POST /api/parsers/dump/preview`, streamed,
never materializes the full trajectory — no download route) alongside the full atom table
(`POST /api/parsers/dump/table`, downloadable). The data-file page takes a `section` field
(`Atoms`, `Bonds`, `Masses`, `Velocities`, `Angles`, `Dihedrals`, `Impropers`, or one of the
`* Coeffs` sections; default `Atoms`).

### Chain conformations (`/conformations/...`)

| Page | Route | File it needs |
|---|---|---|
| Radius of gyration | `/conformations/rg` | a LAMMPS dump trajectory file |
| End-to-end distance | `/conformations/ree` | a LAMMPS dump trajectory file |
| End-to-end vectors | `/conformations/ree-vectors` | a LAMMPS dump trajectory file |

All three take `coord_cols` (default `xu,yu,zu`, three comma-separated column names) and
`molecule_col` (default `mol`); Rg² additionally takes an optional `mass_col`, and the end-to-end
pages take `atom_id_col` (default `id`) to identify each chain's terminal atoms.

### Dynamics (`/dynamics/...`)

| Page | Route | File it needs |
|---|---|---|
| End-to-end vector autocorrelation | `/dynamics/segment-acf` | a NumPy `.npz` trajectory file |
| Rouse mode autocorrelation | `/dynamics/rouse-mode` | a NumPy `.npz` trajectory file |
| Intermediate scattering function | `/dynamics/isf` | a NumPy `.npz` trajectory file |

The `.npz` file must contain a `coords` array of shape `(n_frames, n_atoms, 3)`. These pages need a
`time_per_frame` and a way to group atoms into chains — either **uniform** mode (`n_chains`,
`beads_per_chain`, `first_index`, contiguous ascending atom-ID blocks) or **custom** mode (one chain
per line in a `custom` textarea). The Rouse mode page additionally takes `p` (which mode; `p=0` is
the non-decaying center-of-mass mode, `p=1` the whole-chain relaxation mode); the ISF page takes
`q_magnitude` and `n_vectors` (isotropic averaging directions, default 50). See
[Limits and timeouts](#limits-and-timeouts) — these are the CPU-heavy pages.

### Rheology (`/rheology/...`)

| Page | Route | File it needs |
|---|---|---|
| Stress relaxation modulus | `/rheology/stress-relaxation` | a stress-ACF file |
| Dynamic moduli | `/rheology/dynamic-moduli` | a stress-ACF file |

Both take `volume` and `temperature` (both required, strictly positive, LJ reduced units unless
you supply your own `kB`-equivalent data). Dynamic moduli additionally takes the Prony-fit
parameters `method` (`GK`, `FSR`, or `both`; default `GK`), `t_min` (default 1.0), optional
`t_cutoff`, `n_modes` (default 50), `n_omega` (default 200), and optional `omega_min`/`omega_max`.

### Topology tools (`/modifiers/...`)

| Page | Route | File it needs |
|---|---|---|
| Rewrite end beads | `/modifiers/rewrite-end-beads` | a molecular-style LAMMPS data file |

Takes `new_end_type` and `base_type` (default 1). This page is download-only — there is no
on-screen "view" step, since the result is a modified data file, not a table.

## JSON API

Every calculation is also a plain HTTP endpoint under `/api/<section>/...`. Interactive docs
(Swagger UI, generated from the same route definitions) are at `/api/docs`; the raw OpenAPI schema
is at `/api/openapi.json`.

A request without the `HX-Request: true` header (i.e. anything that isn't the app's own htmx
frontend) gets JSON back — a plain `curl` call needs no special `Accept` header:

```bash
curl -F file=@tests/data/sample_stress_acf.txt \
     -F volume=1000 \
     -F temperature=1 \
     http://127.0.0.1:8000/api/rheology/stress-relaxation
```

returns

```json
{
  "title": "Stress relaxation modulus",
  "columns": ["lag_time", "G_GK", "G_FSR"],
  "rows": [[0.0, 12.34, 12.9], ...],
  "n_rows_total": 500,
  "truncated": false,
  "meta": {"file": "sample_stress_acf.txt"},
  "plot": {"x": "lag_time", "y": ["G_GK", "G_FSR"], ...},
  "download_stem": "sample_stress_acf_G_t"
}
```

`rows` is capped at `LADA_WEB_MAX_TABLE_ROWS` (`truncated` says so when it is); `plot.series`
always carries every row.

Every view route except `POST /api/parsers/dump/preview` has a `.../download` sibling that takes
the same fields plus a `fmt` query parameter (`csv` or `pgfplots`; default `csv`) and streams the
file instead of returning JSON — it recomputes from the same upload rather than reusing a cached
result (see [Design notes](#design-notes)):

```bash
curl -F file=@tests/data/sample_stress_acf.txt \
     -F volume=1000 \
     -F temperature=1 \
     "http://127.0.0.1:8000/api/rheology/stress-relaxation/download?fmt=pgfplots" \
     -o sample_stress_acf_G_t.dat
```

`lada` errors (bad columns, wrong ACF schema, `p >= beads_per_chain`, and similar) come back as
HTTP 422 with the original message unchanged. Oversized uploads come back as HTTP 413.

## Configuration

Every setting is a `LADA_WEB_`-prefixed environment variable (`lada_web/config.py:Settings`):

| Variable | Default | Meaning |
|---|---|---|
| `LADA_WEB_MAX_UPLOAD_MB` | `50` | Size cap for text uploads (dump/log/data/ACF files) |
| `LADA_WEB_MAX_TRAJECTORY_MB` | `500` | Size cap for `.npz` trajectory uploads |
| `LADA_WEB_MAX_TRAJECTORY_FRAMES` | `3000` | Rejects a trajectory with more frames than this, checked right after loading the array's shape, before any calculation runs |
| `LADA_WEB_MAX_TABLE_ROWS` | `500` | Rows shown on screen / returned in JSON `rows`; downloads and plots are never truncated by this |
| `LADA_WEB_TEMP_ROOT` | unset (system temp directory) | Where per-request upload workspaces are created; set this to point uploads at a specific volume |

## Limits and timeouts

The three dynamics calculations (segment ACF, Rouse mode ACF, ISF) are CPU-bound and scale as
**O(n_frames²)** per chain (worse for the ISF, which additionally averages over `n_vectors`
scattering directions) — a large trajectory can take minutes to process on one request.

- `LADA_WEB_MAX_TRAJECTORY_FRAMES` (default 3000) is the guardrail: it rejects oversized
  trajectories with HTTP 413 before the sliding-window loop starts, rather than running a job queue.
  Lower it if you're running on modest hardware; raise it only if you also raise proxy timeouts.
- If you put this app behind a reverse proxy, raise its read timeout well past the default, or
  large dynamics requests will be cut off mid-calculation. For nginx:

  ```nginx
  proxy_read_timeout 600s;
  ```

  Uvicorn itself has no request timeout by default, so the proxy is almost always the first thing
  that gives up.

## Docker

```bash
cd webapp
docker compose up
```

serves the app on `http://localhost:8000` using the five `LADA_WEB_*` variables at their defaults
(edit `docker-compose.yml` to change them). The image is built from `webapp/` only (`build: .` with
`webapp/` as the build context) and runs as a non-root user.

**This will not build successfully today** — see the PyPI pin caveat at the top of this document:
the Dockerfile installs `lada>=1.2.0,<2.0.0` from PyPI, and PyPI does not yet have a 1.2.0 release.
Docker was not available on the machine this was written on either, so the Dockerfile, image build,
and `docker compose up` have not been run or verified end to end — review it carefully before
relying on it, and treat "builds cleanly" as unconfirmed until both are true.

## Testing

```bash
# from webapp/
../.venv/bin/python -m pytest -q -p no:cacheprovider   # or: python -m pytest, once installed

# from the repository root
.venv/bin/ruff check webapp/src webapp/tests
.venv/bin/ruff format --check webapp/src webapp/tests
```

Tests use `starlette.testclient.TestClient` against `create_app()` with lowered limits (so
oversized-upload and frame-count rejection paths are reachable without huge fixtures), and reuse
the fixtures in the root `tests/data/` rather than duplicating any under `webapp/`.

## Design notes

- **Stateless, recompute-on-download.** There is no server-side result cache and no database. Every
  "Download CSV" / "Download pgfplots" button resubmits the full form (file included) to a
  `.../download` route, which recomputes from scratch and streams the file. Simpler than caching a
  result keyed by upload; the cost is recomputing once per download, which is fine at this scale.
- **No background job queue.** Long-running dynamics requests block the worker thread for the
  duration of the request (see [Limits and timeouts](#limits-and-timeouts)) rather than being
  handed off to Celery/RQ/etc. The `services/` layer is pure functions with no FastAPI imports
  specifically so that move stays cheap later if usage ever demands it — it isn't built now.
- **`/api/` is reserved, not yet a public API.** There is no auth, no rate limiting, and no API
  key — every route is reachable by anyone who can reach the server. That's fine for a single-user
  or trusted-network deployment; don't expose an unauthenticated instance to the open internet.
- **Error responses can be verbose.** The catch-all 500 handler deliberately echoes the exception's
  type and message back to the client (e.g. `"Unexpected error while processing the request:
  ZeroDivisionError: ..."`), rather than hiding it behind a generic message — useful for debugging a
  locally-run instrument, but it does mean an unhandled server error can leak implementation
  details (file paths, internal function names) to whoever sent the request. Keep that in mind
  before exposing an instance beyond localhost or a trusted network; a reverse proxy that rewrites
  5xx bodies is one way to close that off without touching the app.
- **No 3D viewer.** This app produces tables, downloads, and 2D plots (Plotly.js) of scalar/vector
  results. Trajectory/molecule visualization is out of scope by design — OVITO/VMD already cover it.
