# Web application front-end for `lada`

## Context

`lada` is currently code-only: every capability (parsing LAMMPS files, computing Rg²/Ree²/stress
relaxation/dynamic moduli/ACFs, exporting to pgfplots, rewriting topology files) requires writing
Python. The goal is to let people who know LAMMPS/MD but not Python use the same functionality
through a browser — upload a file, pick a calculation, see results as a table/chart, download.

Two decisions were made explicitly with the user before this plan was written:

1. **Scope**: a *calculation* front-end, not a 3D trajectory/molecule viewer. `DEVELOPMENT_PLAN.md`
   (gitignored, local-only — confirmed via `.gitignore` lines 24/26 — so this reflects a past local
   planning note, not a public commitment) has a "GUI / web dashboards: out of scope, OVITO/VMD
   already cover this" note; the user confirmed that note is about 3D rendering, and doesn't apply
   to a 2D tables-and-charts front-end over the existing functions. That note will be reworded as
   part of this work (see Phase 4) rather than left contradicting the new `webapp/` directory.
2. **Architecture**: FastAPI backend + a server-rendered frontend (not Streamlit/Gradio) — chosen
   specifically because the user wants room to grow this into "a live, working website" later
   (custom design, a public API, possibly accounts/sharing), which a REST API + frontend split
   supports more naturally than a Python-only data-app framework.
3. **Coverage**: full coverage from day one — all four submodules' public functions, including the
   `.npz`-based dynamics functions, whose array-shaped inputs (`segment_pairs`, `chain_indices`)
   need a genuine UI design (not raw array entry).

The core `lada` library's own stated philosophy (README: named after the LADA car, "simple,
reliable, ... without unnecessary luxury features") argues against adding FastAPI/Jinja2/etc. to
`lada`'s own dependency list. So the web app is a **separate installable project living in the same
repo**, depending on `lada` rather than the other way around — keeps `pip install lada` exactly as
lightweight as it is today.

## Repository layout

```
LaDa/
├── src/lada/                          # existing core package — untouched
├── tests/data/
│   ├── generate_sample_trajectory.py  # NEW — seeded generator
│   ├── sample_trajectory.npz          # NEW — committed (~19 KB); closes the "no .npz fixture
│   │                                      exists anywhere" gap from the earlier dynamics-module
│   │                                      verification work
│   ├── generate_sample_stress_acf.py  # NEW — seeded generator
│   ├── sample_stress_acf.txt          # NEW — committed; sample_acf.txt uses acf_x/y/z columns,
│   │                                      not the ACF_Sxy-style schema calc_stress_relaxation needs
│   └── sample_molecular.data          # NEW — molecular-style fixture (sample.data is atomic-style
│                                          and can't exercise rewrite_end_beads correctly)
├── webapp/                            # NEW — separate installable project
│   ├── pyproject.toml
│   ├── README.md
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── src/lada_web/
│   │   ├── main.py                    # create_app() factory
│   │   ├── config.py                  # Settings (pydantic-settings): upload/frame-count caps
│   │   ├── uploads.py                 # streamed UploadFile -> temp-file helpers, size guards
│   │   ├── errors.py                  # lada ValueError/KeyError -> HTTP 422, forwarded verbatim
│   │   ├── routers/{pages,parsers,conformations,dynamics,rheology,modifiers}.py
│   │   ├── schemas/{common,parsers,conformations,dynamics,rheology,modifiers}.py
│   │   ├── services/                  # pure functions, no FastAPI imports — wrap the lada calls
│   │   │   ├── parsers_service.py / conformations_service.py / dynamics_service.py
│   │   │   ├── rheology_service.py / modifiers_service.py / export_service.py
│   │   ├── templates/                 # Jinja2: base + one page per calculation, per submodule
│   │   └── static/{css/app.css, js/app.js}
│   └── tests/                         # TestClient-based; reuses root tests/data/ fixtures
└── DEVELOPMENT_PLAN.md                # scope note reworded (Phase 4, gitignored/local file)
```

## Backend design

- **App factory** (`create_app()`), not a bare module `app` — lets tests build isolated instances
  with overridden settings (lower upload caps, `tmp_path`-backed temp dirs).
- **Routes are plain `def`, not `async def`.** FastAPI already runs sync path operations in its
  worker threadpool automatically — this is what keeps the event loop unblocked during the
  `O(n_frames²)` sliding-window loops in `dynamics.py`, with no extra code needed.
- **Function → endpoint map** (full list; `iter_dump_frames` and the three deprecated `dynamics.py`
  aliases are deliberately never wired to a route):

  | Submodule | Function | Route |
  |---|---|---|
  | parsers | `dump_frames` | `GET /api/parsers/dump/preview` — per-frame summaries, streamed, never materializes the full trajectory |
  | parsers | `read_dump` | `POST /api/parsers/dump/table` |
  | parsers | `read_lammps_log` | `POST /api/parsers/log/table` |
  | parsers | `read_data_file` | `POST /api/parsers/data/table` (`section` param) |
  | parsers | `read_lammps_acf` | `POST /api/parsers/acf/table` |
  | conformations | `calculate_avg_rg_sq` / `calculate_avg_ree_sq` / `calculate_ree_vectors` | `POST /api/conformations/{rg,ree,ree-vectors}` |
  | dynamics | `calculate_segment_acf_from_trajectory` / `calculate_rouse_mode_acf_from_trajectory` / `calculate_isf_from_trajectory` | `POST /api/dynamics/{segment-acf,rouse-mode,isf}` |
  | rheology | `calc_stress_relaxation` / `calc_dynamic_moduli_prony` | `POST /api/rheology/{stress-relaxation,dynamic-moduli}` |
  | modifiers | `rewrite_end_beads` | `POST /api/modifiers/rewrite-end-beads` — download-only, no "view" step |

  Every view route has a `.../download?fmt=csv|pgfplots` sibling (see Downloads below).

- **File uploads**: parsers take filepaths, so uploads stream to a per-request `TemporaryDirectory`
  (`uploads.py::save_upload`, `shutil.copyfileobj` — never buffers a whole `.npz` into memory).
  Two independent size caps via `Settings`: `MAX_UPLOAD_MB` (default 50, text files) and
  `MAX_TRAJECTORY_MB` (default 500, `.npz`). View/compute endpoints clean up the temp dir via a
  yield-dependency's `finally`; download endpoints instead pass
  `FileResponse(..., background=BackgroundTask(shutil.rmtree, workspace))` — a yield-dependency's
  teardown isn't guaranteed to run after a `FileResponse` finishes streaming, so download cleanup
  needs the explicit `BackgroundTask` form.
- **Guardrail instead of a job queue**: `MAX_TRAJECTORY_FRAMES` (e.g. 3000) checked against
  `coords.shape[0]` right after `np.load`, before the sliding-window loop — cost is
  `O(n_frames²·n_chains)` (segment/Rouse ACF) or worse for ISF (`n_vectors=50` default multiplier),
  so frame count matters more than file size in MB. `services/*.py` being pure functions (no
  FastAPI imports) is what keeps a future Celery/RQ move cheap *if* usage ever demands it — not
  built now.
- **Errors**: `lada`'s existing `ValueError`/`KeyError` messages (bad columns, wrong ACF schema,
  `p >= beads_per_chain`, etc.) are already clear and user-facing — forward them verbatim as HTTP
  422 rather than re-wrapping.

## The `segment_pairs` / `chain_indices` problem

Both dynamics inputs are arrays that no non-coder should type by hand. `schemas/dynamics.py`
defines a `mode: "uniform" | "custom"` toggle (flat `Form(...)` fields, since multipart requests
can't auto-bind nested Pydantic models):

- **Uniform** (default): "number of chains" + "beads per chain" + "first atom index" (default 0).
  Server generates contiguous sequential blocks — chain *i* = atoms
  `[first + i·beads_per_chain, first + (i+1)·beads_per_chain)`. This mirrors the same
  min/max-atom-ID-per-molecule convention `calculate_avg_ree_sq`/`rewrite_end_beads` already use
  elsewhere in `lada`, so it's not a new convention invented just for the UI.
- **Custom/advanced**: a textarea, one chain per line (`head,tail` or a bead list), parsed with
  row-level error messages. Implemented in `dynamics_service.py` as pure, independently-testable
  `resolve_segment_pairs`/`resolve_chain_indices` functions.
- `.npz` files carry no atom-ID metadata, so there's no way to auto-verify a Uniform-mode guess
  against ground truth — the page shows an explicit warning that atom order is assumed to match
  ascending atom ID, 0-indexed.

## Frontend

**Jinja2 + htmx + Alpine.js + Plotly.js + Pico.css, all via CDN — no Node build step.**

- htmx drives every form (`hx-post`, `hx-target="#results"`) — server returns an HTML partial.
- Alpine.js (new, ~15 KB) handles the one place htmx is the wrong tool: local-only UI state like
  the Uniform/Custom toggle and clamping the Rouse `p` input's max to the current `beads_per_chain`.
- Plotly.js over Chart.js: rheology's G'(ω)/G''(ω) and ACF/ISF decay curves want log-axis plots
  with precise hover tooltips — Plotly's log-axis support is first-class, Chart.js's is weaker.
- Pico.css: classless, clean-looking by default, no class-annotated markup, small footprint —
  fits the "no unnecessary luxury features" ethos better than pulling in Bootstrap.
- **Downloads reuse `write_pgfplots_table` for both CSV and pgfplots** — confirmed from
  `src/lada/exporters/latex_exporter.py:118-122`: `delimiter=','` with no `comment` produces a
  clean, header-plus-comma-rows file `pd.read_csv` reads directly; `delimiter=' '` (default)
  produces the `.dat` pgfplots wants. One code path, two `fmt` values.
- **Stateless recompute-on-download**: no server-side result cache. Each page's "Download CSV" /
  "Download pgfplots" buttons resubmit the same multipart form to a `.../download` route, which
  recomputes and streams the file. Simpler than caching; the tradeoff (recomputing once per
  download) is fine at this scale and is a documented, deliberate choice, not an oversight.

## Testing

- `webapp/tests/conftest.py`: `TestClient(create_app(settings=test_settings))` with lowered caps
  for rejection tests, plus a fixture resolving the **root** `tests/data/` (not duplicated).
- Per-router tests: happy path, FastAPI validation (422 on missing fields), `lada` domain errors
  forwarded correctly, oversized-upload rejection; dynamics additionally covers both topology modes
  and the frame-count guardrail.
- Download tests assert `Content-Disposition: attachment` and round-trip the bytes back through
  `pd.read_csv`/`np.loadtxt`, checked numerically against the JSON "view" response — regression-
  tests the recompute-on-download design's consistency.
- `test_temp_cleanup.py`: snapshot the temp root before/after a battery of requests, including
  failing ones, to catch leaked files — the one genuinely new lifecycle risk this app introduces.
- New fixtures land in the **root** `tests/data/` (not `webapp/`), so they also close pre-existing
  gaps in the core `lada` suite noted during earlier verification work this session: no `.npz`
  fixture existed anywhere, and `tests/test_modifiers/` doesn't exist at all today.

## Phased build

1. **Backend skeleton + parsers + conformations + rheology** — `webapp/` scaffold, `create_app`,
   base template, the two new root fixtures (`sample_trajectory.npz`, `sample_stress_acf.txt`),
   then the 9 non-`.npz`/non-file-out functions and their `/download` variants. Splittable into two
   PRs (parsers; conformations+rheology) if smaller review chunks are wanted.
2. **Dynamics / `.npz` endpoints** — the three trajectory functions, the Uniform/Custom
   `TopologySpec`, `MAX_TRAJECTORY_FRAMES` guardrail, Alpine.js introduced here.
3. **Modifiers** — `rewrite_end_beads` + `sample_molecular.data`.
4. **Polish & deploy readiness** — consistent error partials, upload-size middleware finalized,
   landing page listing every calculation, Dockerfile hardening (non-root user), `docker-compose.yml`
   for local dev, `webapp/README.md` (including a reverse-proxy timeout note for slow dynamics
   requests), CI job (`webapp-tests.yml`), and the `DEVELOPMENT_PLAN.md` scope reword. Visual
   polish here is a good point to invoke the `frontend-design` skill rather than defaulting to bare
   Pico.css throughout.

**Each phase is its own review checkpoint** — this plan covers the full architecture, but
implementation proceeds phase by phase with a check-in before moving past Phase 1, not as one large
unreviewed build.

## Deployment

`webapp/Dockerfile`: single-stage `python:3.12-slim`, build context = `webapp/` only, `lada`
installed from PyPI (`lada>=1.2.0,<2.0.0`) rather than the local `src/` tree — keeps the image
simple and makes picking up a new `lada` release an explicit pin bump. Non-root user, plain
multi-worker Uvicorn (Gunicorn+UvicornWorker noted as a later upgrade if this graduates to heavier
traffic, not needed now). `docker-compose.yml` for local dev only, no DB/Redis service (v1 is
stateless). Nothing here blocks hosting on Fly.io/Render/a university server later — a
Dockerfile-in-subdirectory is a supported pattern everywhere, and the `/api/` prefix plus the
pure-function `services/` layer are what keep "public API" and "background job queue" additive
later rather than architecture changes.

## Dependencies (`webapp/pyproject.toml`)

```
dependencies = ["lada>=1.2.0,<2.0.0", "fastapi", "uvicorn[standard]", "python-multipart", "jinja2", "pydantic-settings"]
dev = ["pytest", "httpx", "ruff"]
```

Deliberately excluded: any DB driver, `celery`/`redis`/`arq`, auth library, Node/npm toolchain,
`numba` (core `lada`'s own unrelated concern) — none needed for v1, nothing here blocks adding them
later.

## Explicitly out of scope

3D trajectory/molecule visualization; accounts/auth/sharing/persistent storage; a background job
queue; a public rate-limited/API-keyed surface (the `/api/` prefix is reserved for it, not built);
real-time progress via WebSockets/SSE; the `lada` CLI from the (gitignored) `DEVELOPMENT_PLAN.md`
§5.5 (separate effort, philosophically compatible); promoting the topology-inference helper into
core `lada`; any Node/JS-framework build tooling; `iter_dump_frames` and the deprecated dynamics
aliases; functions that don't exist in core `lada` yet (MSD, RDF, bond/dihedral distributions,
xyz/hdf5 exporters, restart-file parsing).

## Verification

1. `python3 -m pytest` from `webapp/` (own venv/install: `pip install -e . -e ./webapp[dev]` from
   repo root) — all router tests green, including the temp-cleanup regression tests.
2. `ruff check webapp/src webapp/tests` clean, mirroring the core package's ruleset.
3. Manual smoke test per phase: `uvicorn lada_web.main:app --reload`, exercise each new page in a
   browser against the real `tests/data/` fixtures (including the new `.npz`/stress-ACF/molecular
   fixtures) — upload → view results table/chart → download CSV and pgfplots → confirm the
   downloaded file round-trips to the same numbers shown on screen.
4. `docker compose up` (Phase 4) — confirm the container serves the same behavior as local dev.
5. Confirm `python3 -m pytest` and `ruff check src/ tests/` in the **core** repo are still green
   after each phase (the core package itself is never touched, only its fixtures grow).

## Status

- [ ] Phase 1 — Backend skeleton + parsers + conformations + rheology
- [ ] Phase 2 — Dynamics / `.npz` endpoints
- [ ] Phase 3 — Modifiers
- [ ] Phase 4 — Polish & deploy readiness
