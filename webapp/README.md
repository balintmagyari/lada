# lada-web

Browser front-end for the `lada` package, in two workspaces that share one sidebar:

- **Simulations** — upload a LAMMPS output file, pick a calculation, see the result as a table
  (and a chart where one applies), download it as CSV or a pgfplots-ready `.dat` file. Every
  calculation is a thin web layer over the functions documented in the repository root
  `README.md`.
- **Rheometer** — import the files a rheometer exports (or any table of rheology data), draw them
  as the standard rheology plots or any column against any other, build master curves by
  time–temperature superposition, and export publication figures (SVG, PDF, PNG) or pgfplots
  sources for LaTeX.

It is a calculation and plotting tool, not a 3D trajectory/molecule viewer. This document covers
what is specific to running it as a service.

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

`/` is a home page with one door per workspace; `/simulations` and `/rheometer` are their
overviews. The sidebar lists every tool of the current workspace, always in the same order, and
a two-part switch under the search box flips it to the other workspace without leaving the page.
`/` or ⌘K / Ctrl+K opens a search box over the tools of both workspaces.

### Simulations workspace

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

#### Read files (`/parsers/...`)

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

#### Chain conformations (`/conformations/...`)

| Page | Route | File it needs |
|---|---|---|
| Radius of gyration | `/conformations/rg` | a LAMMPS dump trajectory file |
| End-to-end distance | `/conformations/ree` | a LAMMPS dump trajectory file |
| End-to-end vectors | `/conformations/ree-vectors` | a LAMMPS dump trajectory file |

All three take `coord_cols` (default `xu,yu,zu`, three comma-separated column names) and
`molecule_col` (default `mol`); Rg² additionally takes an optional `mass_col`, and the end-to-end
pages take `atom_id_col` (default `id`) to identify each chain's terminal atoms.

#### Dynamics (`/dynamics/...`)

| Page | Route | File it needs |
|---|---|---|
| End-to-end vector autocorrelation | `/dynamics/segment-acf` | a NumPy `.npz` trajectory file |
| Rouse mode autocorrelation | `/dynamics/rouse-mode` | a NumPy `.npz` trajectory file |
| Intermediate scattering function | `/dynamics/isf` | a NumPy `.npz` trajectory file |

The `.npz` file must contain a `coords` array of shape `(n_frames, n_atoms, 3)`. These pages need a
`time_per_frame` and a way to group atoms into chains, in one of three modes:

- **uniform** (`n_chains`, `beads_per_chain`, `first_index`) — contiguous ascending atom-ID blocks.
- **data_file** (`topology_file`) — upload the LAMMPS data file for the same system and the chains
  are read from it: from the molecule ids where the atom style has them, otherwise from the `Bonds`
  section. Because a `.npz` holds no atom ids the two files are lined up by ranking atom ids, so
  the data file has to describe the same number of atoms as the trajectory; a mismatch is a 422.
  Chains of unequal length work for the segment ACF but not for Rouse modes. Choosing a file also
  fires `POST /api/topology/preview`, which reports the chains found — and anything left out, such
  as solvent — before the calculation is started.
- **custom** — one chain per line in a `custom` textarea.

The Rouse mode page additionally takes `p` (which mode; `p=0` is
the non-decaying center-of-mass mode, `p=1` the whole-chain relaxation mode); the ISF page takes
`q_magnitude` and `n_vectors` (isotropic averaging directions, default 50). See
[Limits and timeouts](#limits-and-timeouts) — these are the CPU-heavy pages.

#### Rheology (`/rheology/...`)

| Page | Route | File it needs |
|---|---|---|
| Stress relaxation modulus | `/rheology/stress-relaxation` | a stress-ACF file |
| Dynamic moduli | `/rheology/dynamic-moduli` | a stress-ACF file |

Both take `volume` and `temperature` (both required, strictly positive, LJ reduced units unless
you supply your own `kB`-equivalent data). Dynamic moduli additionally takes the Prony-fit
parameters `method` (`GK`, `FSR`, or `both`; default `GK`), `t_min` (default 1.0), optional
`t_cutoff`, `n_modes` (default 50), `n_omega` (default 200), and optional `omega_min`/`omega_max`.

#### Topology tools (`/modifiers/...`)

| Page | Route | File it needs |
|---|---|---|
| Rewrite end beads | `/modifiers/rewrite-end-beads` | a molecular-style LAMMPS data file |

Takes `new_end_type` and `base_type` (default 1). This page is download-only — there is no
on-screen "view" step, since the result is a modified data file, not a table.

### Rheometer workspace

| Section | Page | What it draws |
|---|---|---|
| My data | `/rheometer/my-data` | Import files; check and correct how every column was read |
| Oscillation | `/rheometer/frequency-sweep` | G′, G″ (and optionally tan δ, \|η*\|, δ) against ω |
| Oscillation | `/rheometer/amplitude-sweep` | G′, G″ against strain or stress amplitude |
| Oscillation | `/rheometer/temperature-sweep` | G′, G″ and tan δ against temperature |
| Oscillation | `/rheometer/time-sweep` | G′, G″ and tan δ against time |
| Flow & transient | `/rheometer/flow-curve` | Viscosity (and optionally shear stress) against shear rate |
| Flow & transient | `/rheometer/creep` | Creep compliance J(t) against time |
| Flow & transient | `/rheometer/stress-relaxation` | Relaxation modulus G(t) against time |
| Superposition | `/rheometer/master-curve` | Master curve by time–temperature superposition, with aT, WLF and Arrhenius fits |
| Custom plot | `/rheometer/custom-plot` | Any column of any file on either axis |

**Import.** `/rheometer/my-data` takes up to `LADA_WEB_MAX_FILES_PER_REQUEST` files at once:
`.csv`, `.txt`, `.dat`, `.tsv` (any delimiter, decimal commas, units in a row under the header or
in the header itself), Excel `.xlsx`/`.xls` (every sheet), and exports from **TA Instruments
TRIOS** (split into its `[step]` sections) and **Anton Paar RheoCompass** (split into its
intervals), and **TA Orchestrator** data files (`.RSD01`, `.RSD02` …, the binary files ARES
rheometers save). The format is recognised from the content, not the extension. Every column is matched
to a rheology quantity by its header (English and common German names), converted to canonical
units (Pa, Pa·s, rad/s, 1/s, s, °C, %), and missing quantities that follow exactly from the
measured ones are added: tan δ, |G*|, |η*|, δ, and ω ↔ f. Each table is labelled with the kind of
test it looks like and the temperature it was measured at. A column that was not recognised, or
recognised wrongly, can be corrected on the page (`POST /api/rheometry/rederive`).

> **The TRIOS and RheoCompass readers have not been tested against real exports.** They follow
> those programs' commonly documented layouts, and the fixtures in
> `webapp/tests/data/rheometry/` are synthetic. Because the format-specific code only splits a
> file into blocks and every block then goes through the same tolerant table finder
> (`rheometry/tabular.py`), a layout that differs a little should still import, with less
> metadata at worst. Add real exports as fixtures when you have them.

**TA Orchestrator files** hold the instrument's raw signals rather than the table Orchestrator
shows, so `rheometry/orchestrator.py` recomputes that table (G′, G″, tan δ, stress, strain,
torque, frequency, temperature, plus elapsed time) and the test's metadata (name, operator,
start, plate diameter, gap, transducer range, notes). It was checked against Orchestrator's own
text exports of a frequency, a strain and a time sweep (8 mm parallel plates, both transducer
ranges, `webapp/tests/data/rheometry/orchestrator/`): every column agrees to the float32
precision the file stores. Other geometries (cone-and-plate, Couette, torsion) are read with the
same parallel-plate formulas and are flagged on import: compare one point with Orchestrator
before relying on them, and add an export as a fixture when you have one.

**Where the data lives.** Imported files are kept in the browser (IndexedDB), never on the
server: the server reads each upload into JSON and forgets it. Every rheometer page in that browser
sees the same files, a reload loses nothing, and *Save project* / *Open project* move them between
browsers as one JSON file. Each graph page remembers its settings (not its data) in
`localStorage`.

**Graphs and export.** Each graph page is the same studio with a different preset: it ticks the
tables that match its kind of test, overlays samples with one colour each (G′ filled, G″ open),
offers unit, axis, size and style controls (what goes on each axis, linear or log scale and limits for x, left and right y — either end alone — and a colour per sample plus a marker shape, fill, size and colour per quantity), and has two views — *Explore* (interactive Plotly) and
*Figure*, which shows the exported figure itself, rendered by the server. Export gives SVG, PDF
(fonts embedded) or PNG at 300/600/1200 dpi, at exactly the chosen size (single column 85 mm,
double column 175 mm, slide, square), always in print style whatever the site's theme; or a
pgfplots zip with `figure.tex` and one data table per series, which compiles as is
(`\input{figure}` with `pgfplots`, `compat=1.18`).

**Master curves.** The server finds each sweep's horizontal shift by overlap of log G′ and log G″
with its already-placed neighbour, working outwards from the reference temperature; a shift only
counts when the sweeps then really lie on one curve. Any shift can be typed in by hand. The WLF
and Arrhenius fits follow the shifts actually applied; an optional vertical correction uses
bT = T / T_ref. Tested against a known answer: a 13-mode Maxwell material at five temperatures
with 1 % noise gives aT within 2 % and C1, C2 within 5 %.

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

The Rheometer workspace's routes take and return JSON (except `parse`, which takes the files):

| Route | Does |
|---|---|
| `POST /api/rheometry/parse` | Files (repeated `files` field) → datasets; a file that cannot be read is listed in `errors` by name and does not stop the others |
| `GET /api/rheometry/quantities` | The quantity registry (names, symbols, units with conversion factors), format and test-type labels, accepted file types |
| `POST /api/rheometry/rederive` | A block plus column corrections → the block with values reconverted and derived columns and test type recomputed |
| `POST /api/rheometry/figure?fmt=svg\|pdf\|png&dpi=` | A figure spec → the figure as an attachment (`preview=true` for inline) |
| `POST /api/rheometry/pgfplots` | A figure spec → `figure.tex` and its data tables, zipped |
| `POST /api/rheometry/tts` | Frequency sweeps with temperatures → shift factors, WLF and Arrhenius fits |

## Configuration

Every setting is a `LADA_WEB_`-prefixed environment variable (`lada_web/config.py:Settings`):

| Variable | Default | Meaning |
|---|---|---|
| `LADA_WEB_MAX_UPLOAD_MB` | `50` | Size cap for text uploads (dump/log/data/ACF files) |
| `LADA_WEB_MAX_TRAJECTORY_MB` | `500` | Size cap for `.npz` trajectory uploads |
| `LADA_WEB_MAX_TRAJECTORY_FRAMES` | `3000` | Rejects a trajectory with more frames than this, checked from the array's header, before any of it is loaded |
| `LADA_WEB_MAX_TRAJECTORY_MEMORY_MB` | `2048` | Rejects a trajectory whose `coords` array would need more than this once decompressed (`n_frames × n_atoms × 3 × itemsize`), checked from the same header |
| `LADA_WEB_MAX_TOPOLOGY_MB` | `20` | Size cap for a chain-topology data file. Well under the general upload cap: the parser reads the whole file into memory and parses it there, so peak use is several times the file size, on a request thread |
| `LADA_WEB_MAX_TABLE_ROWS` | `500` | Rows shown on screen / returned in JSON `rows`; downloads and plots are never truncated by this |
| `LADA_WEB_MAX_FILES_PER_REQUEST` | `10` | Most files one calculation (or one regime) may average |
| `LADA_WEB_MAX_CONCURRENT_JOBS` | `4` | Background calculations running at once; further starts answer HTTP 429 |
| `LADA_WEB_JOB_TTL_SECONDS` | `600` | How long a job's result stays fetchable from `/api/jobs/{job_id}` |
| `LADA_WEB_MAX_RHEOMETRY_ROWS` | `200000` | Most data points one imported rheometer file may hold, across all its steps |
| `LADA_WEB_MAX_XLSX_UNCOMPRESSED_MB` | `200` | Refuses an `.xlsx` that would unpack to more than this, checked from the archive's directory before the workbook is opened |
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

- **The Rheometer workspace adds `matplotlib`, `openpyxl` and `xlrd`** to the image, about
  75 MB installed (measured on macOS; a Linux image will be similar). matplotlib builds its font cache while the image is built, as the `lada` user, so the
  first exported figure is as quick as the rest. Figures are drawn one at a time (matplotlib's
  style settings are process-wide), each in well under a second at the default caps.

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
oversized-upload and frame-count rejection paths are reachable without huge fixtures). The
Simulations tests reuse the fixtures in the root `tests/data/`; the Rheometer tests use the
synthetic files in `webapp/tests/data/rheometry/`, written by `generate_fixtures.py` there. The
superposition tests build their own sweeps from a known WLF law. One test compiles an exported
pgfplots figure with `pdflatex`; it is skipped where LaTeX is not installed (as in CI).

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
- **Fonts, and what happens without them.** `base.html` loads one family from Google Fonts —
  Atkinson Hyperlegible Next for text and its Mono for numbers in tables — with `display=swap`.
  `--font-sans` and `--font-mono` in `app.css` keep system stacks behind it, so an offline machine
  paints at once in a system face with the layout unchanged. Pico, htmx, Alpine and Plotly come
  from CDNs; offline, those are what stop working (no plots, no htmx results), not the typeface.
  Navigation is plain DOM in `static/js/app.js`, so it survives a blocked CDN, and with scripting
  off every tool of both workspaces is listed in the sidebar.
- **Rheometer data stays in the browser.** The server parses an upload and forgets it, like every
  other route; the datasets live in the browser's IndexedDB. There is no account, no sharing and
  nothing to back up on the server — and nothing to lose there either. The figure spec the
  browser sends for export already holds the plotted numbers in display units, so the server
  renders exactly what the Explore view shows.
