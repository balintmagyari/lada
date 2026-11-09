# lada-web

A browser tool that turns the files a rheometer exports into clear, publication-ready graphs.
Import exports or spreadsheets, draw them as the standard rheology plots or any column against
any other, build master curves by time–temperature superposition, and export figures (SVG,
PDF, PNG) or pgfplots sources for LaTeX.

This is **version 1**: rheometer graphing only. The simulation tools (LAMMPS parsers,
conformations, dynamics, ACF rheology, topology rewriting) live on the `webapp-simulations`
branch and are planned to return as a second workspace (see `WEBAPP_PLAN.md`).

> **Install `lada` from this repository.** `webapp/pyproject.toml` depends on `lada>=1.2.0`
> for its pgfplots writer, and PyPI's latest `lada` is older, so a bare `pip install ./webapp`
> will fail to resolve. Install `lada` from `src/` first: the Quick start below does this in
> editable mode, and the [Docker](#docker) image does it from the repository root.

## Quick start

From the repository root, with the same virtual environment used for `lada` itself:

```bash
pip install -e . -e "./webapp[dev]"
uvicorn lada_web.main:app --reload
```

Open `http://127.0.0.1:8000`. There is no configuration step for local use; the defaults in
[Configuration](#configuration) apply automatically.

## Pages

`/` (also `/rheometer`) lists every tool. The sidebar lists them in the same order on every
page, and `/` or ⌘K / Ctrl+K opens a search box over them.

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

Interactive docs (Swagger UI, generated from the route definitions) are at `/api/docs`; the raw
OpenAPI schema is at `/api/openapi.json`. Every route takes and returns JSON, except `parse`,
which takes the files:

| Route | Does |
|---|---|
| `POST /api/rheometry/parse` | Files (repeated `files` field) → datasets; a file that cannot be read is listed in `errors` by name and does not stop the others |
| `GET /api/rheometry/quantities` | The quantity registry (names, symbols, units with conversion factors), format and test-type labels, accepted file types |
| `POST /api/rheometry/rederive` | A block plus column corrections → the block with values reconverted and derived columns and test type recomputed |
| `POST /api/rheometry/figure?fmt=svg\|pdf\|png&dpi=` | A figure spec → the figure as an attachment (`preview=true` for inline) |
| `POST /api/rheometry/pgfplots` | A figure spec → `figure.tex` and its data tables, zipped |
| `POST /api/rheometry/tts` | Frequency sweeps with temperatures → shift factors, WLF and Arrhenius fits |

```bash
curl -F files=@webapp/tests/data/rheometry/flow_curve.csv http://127.0.0.1:8000/api/rheometry/parse
```

A file the reader cannot make sense of comes back in `errors` with a message saying why; a
malformed request is HTTP 422, an oversized upload HTTP 413, and a figure requested while
another one is still being drawn HTTP 429.

## Configuration

Every setting is a `LADA_WEB_`-prefixed environment variable (`lada_web/config.py:Settings`):

| Variable | Default | Meaning |
|---|---|---|
| `LADA_WEB_MAX_UPLOAD_MB` | `50` | Size cap for one imported file. Real exports are kilobytes to a few MB |
| `LADA_WEB_MAX_FILES_PER_REQUEST` | `10` | Most files one import may carry |
| `LADA_WEB_MAX_JSON_MB` | `5` | Size cap for the JSON routes (rederive, figure, pgfplots, superposition) |
| `LADA_WEB_MAX_RHEOMETRY_ROWS` | `200000` | Most data points one imported file may hold, across all its steps |
| `LADA_WEB_MAX_XLSX_UNCOMPRESSED_MB` | `200` | Refuses an `.xlsx` that would unpack to more than this, checked from the archive's directory before the workbook is opened |
| `LADA_WEB_TEMP_ROOT` | unset (system temp directory) | Where per-request upload workspaces are created |

`WEB_CONCURRENCY` (read by uvicorn, default 2 in the image) sets the number of worker processes.

### Request size limits

A body larger than the addressed route accepts is refused with HTTP 413 by a middleware in
`create_app`, on the request's declared `Content-Length`: `MAX_UPLOAD_MB × MAX_FILES_PER_REQUEST`
(plus 1 MiB for the multipart envelope) for the import route, `MAX_JSON_MB` everywhere else.
This is a cheap first filter, not a guarantee: it can only act on a length the client declares.
A POST to `/api/` that declares none (a chunked upload, say) is refused outright for that reason;
a body that understates its length is caught later by `uploads.save_upload`, after the bytes it
already streamed in have been spooled to disk. Put a limit in your reverse proxy as well
(`client_max_body_size` for nginx, whose default of 1 MB is too small for workbooks).

## Docker

```bash
cd webapp
docker compose up --build
```

serves the app on `http://localhost:8000`. The build context is the **repository root** (compose
sets `context: ..`; without compose, run `docker build -f webapp/Dockerfile .` from the root), so
the image can install `lada` from `src/` before `lada-web`. `webapp/Dockerfile.dockerignore`
allowlists the few paths that installs need. The container runs as a non-root user. What the
files set beyond that:

- **`HEALTHCHECK`** fetches `/` every 30 s, the cheapest route that proves the app is wired up.
- **`$PORT`** is honoured (default 8000), so platforms that assign the port (Render, Fly.io,
  Railway, Cloudflare Containers) need no change to the image.
- **`WEB_CONCURRENCY=2`** worker processes. Nothing is kept between requests, so any worker can
  answer any request and more workers (or more containers) scale it out with no sticky sessions.
- **`--proxy-headers --forwarded-allow-ips '*'`**, so the logs show the client rather than the
  proxy. It trusts `X-Forwarded-*` from any peer, so `ports:` publishes to `127.0.0.1` only: put
  a proxy in front of that, and do not expose the container port to the internet.
- **`--limit-concurrency 32`**, so a burst of requests is answered with 503 rather than queuing
  without bound.
- **A 1100 MB `tmpfs` on `/tmp`**, with `LADA_WEB_TEMP_ROOT` pointing into it. An upload lands
  there twice (Starlette's spool, then `save_upload`'s copy), so the largest import the defaults
  allow (10 × 50 MB) needs about 1 GB. Real imports use a tiny fraction of that.
- **`mem_limit: 2g`**: that tmpfs plus two workers of roughly 250 MB each.
- **matplotlib, openpyxl and xlrd** add about 75 MB to the image. matplotlib builds its font
  cache while the image is built, as the `lada` user, so the first exported figure is as quick as
  the rest. Each worker draws one figure at a time (matplotlib's style settings are
  process-wide), each in well under a second at the default caps.

**Not yet verified end to end.** Docker was not available on the machine this was written on, so
the image build and `docker compose up` have not been run. Build and exercise it once locally
before relying on it.

## Testing

```bash
# from webapp/
../.venv/bin/python -m pytest -q -p no:cacheprovider   # or: python -m pytest, once installed

# from the repository root
.venv/bin/ruff check webapp/src webapp/tests
.venv/bin/ruff format --check webapp/src webapp/tests
```

Tests use `starlette.testclient.TestClient` against `create_app()` with lowered limits (so the
oversized-upload paths are reachable without huge fixtures). They use the synthetic files in
`webapp/tests/data/rheometry/`, written by `generate_fixtures.py` there, and the TA Orchestrator
files in its `orchestrator/` folder; the superposition tests build their own sweeps from a known
WLF law. One test compiles an exported pgfplots figure with `pdflatex`; it is skipped where LaTeX
is not installed (as in CI).

## Design notes

- **Nothing is stored on the server.** An import is read into JSON inside a temporary directory
  that is removed when the response is sent; the datasets then live in the browser's IndexedDB.
  There is no database, no account and nothing to back up on the server, and nothing to lose
  there either. The figure spec the browser sends for export already holds the plotted numbers in
  display units, so the server renders exactly what the Explore view shows.
- **`/api/` is not yet a public API.** There is no auth, no rate limiting and no API key: every
  route is reachable by anyone who can reach the server. That is fine for a single-user or
  trusted-network deployment; put access control in front (a reverse proxy's basic auth,
  Cloudflare Access, a university VPN) before exposing an instance to the open internet.
- **Unexpected errors are not echoed.** Errors raised on purpose (a file with no table, a unit
  that does not belong to a quantity) carry messages written for the user and pass through; any
  other exception is logged in full and answered with a fixed message, so a client never sees
  server paths.
- **Fonts, and what happens without them.** `base.html` loads one family from Google Fonts,
  Atkinson Hyperlegible Next for text and its Mono for numbers, with `display=swap`;
  `--font-sans` and `--font-mono` in `app.css` keep system stacks behind it, so an offline
  machine paints at once in a system face. Plotly comes from a CDN, so on a network that blocks it
  the graph pages cannot draw; host it alongside `app.js` if the tool must work offline.
  Navigation is plain DOM in `static/js/app.js`, so it survives a blocked CDN, and with scripting
  off every tool is still listed in the sidebar.
