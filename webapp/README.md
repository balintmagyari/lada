# lada-web

Browser front-end for the `lada` package: a calculation tool, not a 3D trajectory/molecule
viewer. Upload a LAMMPS output file, pick a calculation, see the result as a table (and a chart
where one applies), download it as CSV or a pgfplots-ready `.dat` file. Every calculation is a
thin web layer over the functions already documented in the repository root `README.md` — this
document only covers the parts specific to running that functionality as a service.

> **Install `lada` from this repository.** `webapp/pyproject.toml` depends on `lada>=1.2.0,<2.0.0`,
> and the app relies on functions only this repository's `lada` has (`*_from_trajectory`, and the
> `progress_callback` argument behind the progress bars). PyPI's latest `lada` is older, so a bare
> `pip install ./webapp` will fail to resolve. Install `lada` from `src/` first — the Quick start
> below does this in editable mode, and the [Docker](#docker) image does it from the repository root.

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

The conformations, dynamics and rheology pages take **one or more** files in a repeated `files`
field (up to `LADA_WEB_MAX_FILES_PER_REQUEST`), treated as repeat runs with different random seeds
and averaged into one result: Rg²/Ree² per timestep (the files must share their timesteps),
dynamics and rheology curves point by point after truncating to the shortest file. End-to-end
vectors are the exception — vectors of different chains are not comparable, so files are stacked
with a `repeat` column instead of averaged. The dynamics and rheology pages can also split a
calculation into **regimes** (e.g. fast/mid/slow sampling, each with its own files and frame
spacing): each regime is computed separately and the curves are joined by
`POST /api/dynamics/merge-regimes` or `POST /api/rheology/merge-regimes`, which reject regimes whose
lag-time ranges overlap. For dynamic moduli, the regimes' G(t) curves are merged first and fitted
once (`POST /api/rheology/dynamic-moduli/from-regimes`).

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
is at `/api/openapi.json`. The thirteen view routes document a `TableResult` response body there;
the download and modifier routes document a file attachment instead.

A request without the `HX-Request: true` header (i.e. anything that isn't the app's own htmx
frontend) gets JSON back — a plain `curl` call needs no special `Accept` header:

```bash
curl -F files=@tests/data/sample_stress_acf.txt \
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

The eight conformations, dynamics and rheology calculations also have a `.../start` sibling that
takes the same fields, runs the calculation on a background thread and answers HTTP 202 with
`{"job_id": "..."}`. `GET /api/jobs/{job_id}` then reports `status` (`pending`, `running`, `done`
or `error`),
a `progress` fraction from 0 to 1, and — once done — the same `TableResult` as above in `result`.
This is what the pages' progress bars poll. Finished jobs stay fetchable for
`LADA_WEB_JOB_TTL_SECONDS`, and a start beyond `LADA_WEB_MAX_CONCURRENT_JOBS` running jobs answers
HTTP 429.

Every view route except `POST /api/parsers/dump/preview` has a `.../download` sibling that takes
the same fields plus a `fmt` query parameter (`csv` or `pgfplots`; default `csv`) and streams the
file instead of returning JSON — it recomputes from the same upload rather than reusing a cached
result (see [Design notes](#design-notes)):

```bash
curl -F files=@tests/data/sample_stress_acf.txt \
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
| `LADA_WEB_MAX_TRAJECTORY_FRAMES` | `3000` | Rejects a trajectory with more frames than this, checked from the array's header, before any of it is loaded |
| `LADA_WEB_MAX_TRAJECTORY_MEMORY_MB` | `2048` | Rejects a trajectory whose `coords` array would need more than this once decompressed (`n_frames × n_atoms × 3 × itemsize`), checked from the same header |
| `LADA_WEB_MAX_TABLE_ROWS` | `500` | Rows shown on screen / returned in JSON `rows`; downloads and plots are never truncated by this |
| `LADA_WEB_MAX_FILES_PER_REQUEST` | `10` | Most files one calculation (or one regime) may average |
| `LADA_WEB_MAX_CONCURRENT_JOBS` | `4` | Background calculations running at once; further starts answer HTTP 429 |
| `LADA_WEB_JOB_TTL_SECONDS` | `600` | How long a job's result stays fetchable from `/api/jobs/{job_id}` |
| `LADA_WEB_TEMP_ROOT` | unset (system temp directory) | Where per-request upload workspaces are created; set this to point uploads at a specific volume |

## Limits and timeouts

The three dynamics calculations (segment ACF, Rouse mode ACF, ISF) are CPU-bound and scale as
**O(n_frames²)** per chain (worse for the ISF, which additionally averages over `n_vectors`
scattering directions) — a large trajectory can take minutes to process on one request.

- `LADA_WEB_MAX_TRAJECTORY_FRAMES` (default 3000) and `LADA_WEB_MAX_TRAJECTORY_MEMORY_MB`
  (default 2048) are the guardrails: both are read from the `.npz` array header and answer HTTP
  413 before the sliding-window loop starts. The size of the
  uploaded file bounds neither — a compressed `.npz` of a few megabytes can declare an array of
  hundreds of gigabytes — which is why the decompressed footprint is checked separately. Lower
  them if you're running on modest hardware; raise them only if you also raise proxy timeouts,
  and, under Docker, `mem_limit` in `docker-compose.yml`.
- The pages run calculations as background jobs and poll for progress, so no single browser
  request stays open for the whole calculation. The plain view and `.../download` routes still
  compute inside the request, though — if API clients use those behind a reverse proxy, raise its
  read timeout well past the default, or large dynamics requests will be cut off. For nginx:

  ```nginx
  proxy_read_timeout 600s;
  ```

  Uvicorn itself has no request timeout by default, so the proxy is almost always the first thing
  that gives up.

### Request size limits

A body larger than the addressed route accepts is refused with HTTP 413 by a middleware in
`create_app`, on the request's declared `Content-Length` — the trajectory budget for
`/api/dynamics/...`, the upload budget everywhere else, each plus 1 MiB for the multipart
envelope. This is a cheap first filter, not a guarantee: it can only act on a length the client
declares. A POST to `/api/` that declares none (a chunked upload, say) is refused outright for
that reason, since its size could not be checked before the body was read; a body that
understates its length is caught later by `uploads.save_upload`, after the bytes it already
streamed in have been spooled to disk. Put a limit in your reverse proxy as well
(`client_max_body_size` for nginx) if you need one enforced closer to the wire.

## Docker

```bash
cd webapp
docker compose up --build
```

serves the app on `http://localhost:8000`. The build context is the **repository root** (compose
sets `context: ..`; without compose, run `docker build -f webapp/Dockerfile .` from the root), so the
image can install `lada` from `src/` before `lada-web`. `webapp/Dockerfile.dockerignore` allowlists
the few paths that installs need, keeping `.venv/`, `.git/` and test data out of the build. The
container runs as a non-root user. What the files set beyond that:

- **`HEALTHCHECK`** fetches `/` every 30 s — the cheapest route that proves the app is wired up.
- **`--proxy-headers --forwarded-allow-ips '*'`**, because the README recommends nginx in front
  and uvicorn otherwise logs the proxy's own address and scheme for every request. It trusts
  `X-Forwarded-*` from any peer, so `ports:` publishes to `127.0.0.1` only: put the proxy in
  front of that, and do not expose the container port to the internet.
- **`--workers 1`**, and it must stay one: jobs and their progress live in the process's memory,
  so with two workers a job started on one would answer "Job not found" when polled on the other.
  To scale out, run more containers behind a proxy with sticky sessions, not more workers.
- **`--limit-concurrency 16`**, so a burst of requests is answered with 503 rather than queuing
  without bound. Progress polls count towards it, hence the headroom over the job cap.
- **`LADA_WEB_MAX_CONCURRENT_JOBS=2`**, lowered from 4 so the running jobs' trajectories fit the
  memory limit below.
- **A 1200 MB `tmpfs` on `/tmp`**, with `LADA_WEB_TEMP_ROOT` pointing into it. Uploads land there
  twice — Starlette spools the multipart file part while parsing, then `save_upload` writes the
  request's own copy — so a 500 MB trajectory needs about 1 GB, and this keeps all of it out of
  the container's writable layer. It also caps a multi-file request's total upload — ten 500 MB
  trajectories will not fit — so raise it if users average many large files at once.
- **`mem_limit: 4g`**, which is that tmpfs plus two jobs holding up to
  `LADA_WEB_MAX_TRAJECTORY_MEMORY_MB` (lowered to 1024 in the compose file) each, plus the
  interpreter and NumPy. Those numbers move together: raising any of them means raising this.

**Not yet verified end to end.** Docker was not available on the machine this was written on, so
the image build and `docker compose up` have not been run. Build and exercise it once locally before
relying on it.

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
- **In-process jobs, no external queue.** The pages' calculations run on daemon threads tracked
  by an in-memory `JobStore` (`jobs.py`), with a semaphore capping how many run at once and a sweep
  thread dropping results after the TTL — no Redis, Celery or database. The cost is that jobs belong
  to one process: the server must run a single worker, a restart loses running jobs, and a job keeps
  running if its browser tab closes (it is swept later). The `services/` layer is pure functions
  with no FastAPI imports, so moving to an external queue stays cheap if usage ever demands it.
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
- **One web font, and what happens without it.** `base.html` pulls a single UI typeface — IBM Plex
  Sans, weights 400 and 600, `display=swap` — from Google Fonts, so the app looks the same on every
  machine instead of rendering in whatever the OS calls a UI face. `--font-sans` in `app.css` keeps
  the full fallback stack (`"Source Sans 3", "Segoe UI", system-ui, -apple-system, …`) behind it, so
  a machine with no network paints immediately in a system face with the layout unchanged; nothing
  is hidden while the font loads, and no text shifts position, only its shape. Monospace is a system
  stack only (`ui-monospace, "SF Mono", Menlo, …`) — the `<code>` spans in the page copy are not
  worth a second download. The numeric tables rely on `font-variant-numeric: tabular-nums`, which
  both IBM Plex Sans and every face in the fallback stack support, so columns stay aligned either
  way. The same page also loads Pico, htmx, Alpine and Plotly from CDNs; offline, those are the
  things that actually stop working (no plot, no htmx results swap), not the typeface. The section
  menus in the top bar are deliberately plain DOM in `static/js/app.js` rather than Alpine, so
  navigation survives a blocked CDN, and with scripting off entirely every menu renders open in the
  normal flow and all the links stay reachable.
