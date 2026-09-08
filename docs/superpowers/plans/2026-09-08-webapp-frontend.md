# LaDa Web Front-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser front-end (`webapp/`, package `lada-web`) that exposes every public `lada` function as upload → compute → table/chart → download pages, plus a JSON API, without changing the core `lada` package.

**Architecture:** FastAPI app factory with plain `def` routes under `/api/<section>/…`; pure-function `services/` wrap `lada` calls; Jinja2 templates driven by htmx return HTML partials when the request carries `HX-Request: true` and JSON otherwise; every view route has a `…/download?fmt=csv|pgfplots` sibling that recomputes and streams a file written by `lada.exporters.write_pgfplots_table`. Uploads stream into a per-request temp directory that is always cleaned up.

**Tech Stack:** Python ≥3.12, FastAPI, Uvicorn, python-multipart, Jinja2, pydantic-settings; htmx 2.0.4, Alpine.js 3.14.8, Plotly basic 3.0.1, Pico.css 2.1.1 (all CDN, no Node); pytest + Starlette `TestClient` (httpx2).

**Spec:** `WEBAPP_PLAN.md` at the repository root. The spec is the binding authority; this plan argues from it.

## Global Constraints

Every task's requirements implicitly include this section.

**Scope and packaging**
- `src/lada/` is never modified. Only `tests/data/` fixtures, new core tests, `.github/workflows/`, `README.md`, and `WEBAPP_PLAN.md` may change outside `webapp/`.
- `webapp/` is a separate installable project named `lada-web`, version `0.1.0`, `requires-python = ">=3.12"`, depending on `lada>=1.2.0,<2.0.0`. Runtime dependencies are exactly: `fastapi`, `uvicorn[standard]`, `python-multipart`, `jinja2`, `pydantic-settings`. Dev extra is exactly: `pytest`, `httpx2`, `ruff`. (Ruling: `httpx2` replaces the spec's `httpx` because Starlette ≥1.6 deprecates `httpx` for `TestClient`.) Nothing else — no DB driver, no queue, no auth library.
- Ruff config mirrors the core package: `select = ["E","F","I","UP","B","SIM"]`, `ignore = ["SIM105","SIM108","SIM114"]`, `line-length = 100`, `target-version = "py312"`. `ruff check` and `ruff format --check` must be clean on `webapp/src webapp/tests` and on whatever core `tests/` files a task adds.

**Backend rules**
- App factory `create_app(settings: Settings | None = None) -> FastAPI` in `lada_web/main.py`; a module-level `app = create_app()` exists for `uvicorn lada_web.main:app`.
- Route handlers are plain `def` (never `async def`). Exception handlers may be `async def`.
- Routes live under `/api/<section>/...`. HTML pages are served by `routers/pages.py` from the declarative catalog in `catalog.py`.
- A **view route** returns `TableResult` JSON, or the HTML partial `partials/table_result.html` when the request has header `HX-Request: true`. Use `rendering.table_response(request, result, download_url)` — never branch on the header in a router.
- Every view route (except `POST /api/parsers/dump/preview`) has a sibling `<same path>/download` taking the same multipart fields plus query `fmt: Literal["csv","pgfplots"] = "csv"`. It recomputes with the same service call, writes the file with `export_service.write_table`, and returns it via `uploads.file_download(path, workspace, media_type)` (a `FileResponse` with `Content-Disposition: attachment` and `BackgroundTask` cleanup). Download routes wrap their body in `with download_workspace(settings) as ws:`; view routes take `ws: Annotated[Path, Depends(get_workspace)]`.
- Shared parameters of a view/download pair are declared once as a class dependency in `schemas/<section>.py` (constructor parameters annotated with `Form()`), injected with `Annotated[ThatClass, Depends()]`. The uploaded file is always the multipart field named `file`, declared `file: Annotated[UploadFile, File()]`.
- Uploads go through `uploads.save_upload(upload, workspace, max_bytes)` with `settings.max_upload_bytes` for text files and `settings.max_trajectory_bytes` for `.npz` files.
- Errors: `ValueError`/`KeyError` raised by `lada` (or by services/schemas) become HTTP 422 with the message forwarded verbatim (`exc.args[0]`). `UploadTooLarge` and `TrajectoryTooLarge` become HTTP 413. Do not re-wrap `lada` messages.
- `Settings` fields and defaults: `max_upload_mb: float = 50`, `max_trajectory_mb: float = 500`, `max_trajectory_frames: int = 3000`, `max_table_rows: int = 500`, `temp_root: Path | None = None`. Environment variable prefix `LADA_WEB_` (e.g. `LADA_WEB_MAX_UPLOAD_MB`).
- Result DataFrame column names are exact and tested: see each task's Interfaces block.
- Never wire `lada.parsers.iter_dump_frames` or the deprecated aliases `calculate_segment_acf`, `calculate_rouse_mode_acf`, `calculate_isf` to any route; use the `_from_trajectory` names.

**Frontend rules (every page, no exceptions)**
- Templates extend `base.html` and use the macros in `_macros.html`. Pages do not add their own CSS or inline styles; visual identity lives in `static/css/app.css` (Task 1) only.
- Exactly one form per page: `<form id="calc-form" method="post" enctype="multipart/form-data" hx-encoding="multipart/form-data">` with **no** `action` attribute and **no** `hx-*` attributes on the form itself. Form field names equal the `Form()` parameter names. The file input is named `file`.
- The action button is rendered by `f.calculate_button(label, url)`: `<button type="submit" hx-post="{url}" hx-target="#results" hx-swap="innerHTML" hx-indicator="#results-indicator" hx-disabled-elt="this">`. It sits inside the form, before any other submit button, so pressing Enter triggers it. htmx cancels the native submit for it.
- Results land in `<div id="results">` rendered by `f.results_area()` immediately after the form.
- Download buttons are rendered only inside `partials/table_result.html` and are native submits bound to the page form: `<button type="submit" form="calc-form" formaction="{download_url}?fmt=csv" formmethod="post" formenctype="multipart/form-data">Download CSV</button>` and the same with `fmt=pgfplots` labelled `Download pgfplots table`. They re-upload the form (stateless recompute-on-download, per spec).
- Copy: sentence case; plain verbs; buttons say what happens ("Read thermo table", "Calculate Rg²", "Download CSV"); errors say what went wrong and what to change; no all-caps labels, no "·" separators, no trailing arrows.
- CDN assets (exact URLs, already verified to resolve):
  - `https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css`
  - `https://cdn.jsdelivr.net/npm/htmx.org@2.0.4/dist/htmx.min.js`
  - `https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js` (with `defer`)
  - `https://cdn.plot.ly/plotly-basic-3.0.1.min.js` (with `defer`)

**Environment and tooling (read before running anything)**
- Work only inside the worktree you were given. Use its venv directly, never activate another one: from the repo root the interpreter is `.venv/bin/python`; from `webapp/` it is `../.venv/bin/python`.
- Web app tests: `cd webapp && ../.venv/bin/python -m pytest -q -p no:cacheprovider`. Core tests: from the repo root `.venv/bin/python -m pytest -q -p no:cacheprovider`. Lint: `.venv/bin/ruff check <paths>` and `.venv/bin/ruff format --check <paths>` (run `ruff format <paths>` to fix).
- Fixtures are read from the **root** `tests/data/` via the `data_dir` fixture in `webapp/tests/conftest.py`. Never copy fixtures into `webapp/`.
- Git discipline (several agents share this worktree): stage with `git add <explicit paths>` only — never `git add -A`, `git add .`, `git add -u`. Never stash, rebase, amend, reset, or checkout other branches. If `git commit` fails with `index.lock`, wait two seconds and retry (up to five times). Only touch files listed in your task's **Files** block; if you need a change elsewhere, describe it in your report instead of making it.
- Test output must be pristine: no warnings in the pytest summary (the one third-party `anyio.abc.BlockingPortal` deprecation is filtered in `webapp/pyproject.toml`).

---

## File structure

```
.gitignore                                  # + .venv/  (already done by the controller)
tests/data/generate_sample_trajectory.py    # Task 2 — seeded .npz generator
tests/data/sample_trajectory.npz            # Task 2 — 32 frames × (4 chains × 8 beads) × 3
tests/data/generate_sample_stress_acf.py    # Task 2 — seeded stress-ACF generator
tests/data/sample_stress_acf.txt            # Task 2 — ACF_Sxy… schema, 2 timestep blocks × 100 lags
tests/data/sample_molecular.data            # Task 2 — molecular-style data file (3 chains × 4 beads)
tests/data/README.md                        # Task 2 — document the new fixtures
tests/test_modifiers/test_data_modifier.py  # Task 2 — first tests for rewrite_end_beads
tests/test_fixtures.py                      # Task 2 — generators reproduce committed fixtures
webapp/pyproject.toml                       # Task 1
webapp/README.md                            # Task 1 (stub) → Task 9 (full)
webapp/src/lada_web/__init__.py             # Task 1 — __version__
webapp/src/lada_web/config.py               # Task 1 — Settings, get_settings
webapp/src/lada_web/catalog.py              # Task 1 — Section/Calculation registry (all 13 pages)
webapp/src/lada_web/rendering.py            # Task 1 — templates, is_htmx, wants_html, table_response
webapp/src/lada_web/errors.py               # Task 1 — exception classes + handlers
webapp/src/lada_web/uploads.py              # Task 1 — workspaces, save_upload, file_download
webapp/src/lada_web/main.py                 # Task 1 — create_app; Task 8 adds body-size middleware
webapp/src/lada_web/schemas/__init__.py     # Task 1
webapp/src/lada_web/schemas/common.py       # Task 1 — PlotSpec, TableResult
webapp/src/lada_web/schemas/parsers.py      # Task 3
webapp/src/lada_web/schemas/conformations.py# Task 4
webapp/src/lada_web/schemas/rheology.py     # Task 5
webapp/src/lada_web/schemas/dynamics.py     # Task 6
webapp/src/lada_web/schemas/modifiers.py    # Task 7
webapp/src/lada_web/services/__init__.py    # Task 1
webapp/src/lada_web/services/export_service.py        # Task 1
webapp/src/lada_web/services/parsers_service.py       # Task 1 (log) → Task 3 (rest)
webapp/src/lada_web/services/conformations_service.py # Task 4
webapp/src/lada_web/services/rheology_service.py      # Task 5
webapp/src/lada_web/services/dynamics_service.py      # Task 6
webapp/src/lada_web/services/modifiers_service.py     # Task 7
webapp/src/lada_web/routers/__init__.py     # Task 1
webapp/src/lada_web/routers/pages.py        # Task 1
webapp/src/lada_web/routers/parsers.py      # Task 1 (log) → Task 3 (rest)
webapp/src/lada_web/routers/conformations.py# Task 1 (stub) → Task 4
webapp/src/lada_web/routers/rheology.py     # Task 1 (stub) → Task 5
webapp/src/lada_web/routers/dynamics.py     # Task 1 (stub) → Task 6
webapp/src/lada_web/routers/modifiers.py    # Task 1 (stub) → Task 7
webapp/src/lada_web/templates/base.html     # Task 1
webapp/src/lada_web/templates/index.html    # Task 1 → Task 8 (copy polish)
webapp/src/lada_web/templates/_macros.html  # Task 1
webapp/src/lada_web/templates/error_page.html         # Task 1
webapp/src/lada_web/templates/partials/table_result.html  # Task 1
webapp/src/lada_web/templates/partials/error.html         # Task 1
webapp/src/lada_web/templates/parsers/log.html            # Task 1
webapp/src/lada_web/templates/parsers/{dump,data,acf}.html            # Task 3
webapp/src/lada_web/templates/conformations/{rg,ree,ree-vectors}.html # Task 4
webapp/src/lada_web/templates/rheology/{stress-relaxation,dynamic-moduli}.html # Task 5
webapp/src/lada_web/templates/dynamics/_topology.html                  # Task 6
webapp/src/lada_web/templates/dynamics/{segment-acf,rouse-mode,isf}.html # Task 6
webapp/src/lada_web/templates/modifiers/rewrite-end-beads.html        # Task 7
webapp/src/lada_web/static/css/app.css      # Task 1 → Task 8 (polish)
webapp/src/lada_web/static/js/app.js        # Task 1
webapp/tests/conftest.py                    # Task 1
webapp/tests/test_app.py                    # Task 1 → Task 8 (all pages render)
webapp/tests/test_uploads.py                # Task 1
webapp/tests/test_export_service.py         # Task 1
webapp/tests/test_parsers.py                # Task 1 (log) → Task 3 (rest)
webapp/tests/test_conformations.py          # Task 4
webapp/tests/test_rheology.py               # Task 5
webapp/tests/test_dynamics.py               # Task 6
webapp/tests/test_modifiers.py              # Task 7
webapp/tests/test_temp_cleanup.py           # Task 8
webapp/tests/test_body_limit.py             # Task 8
webapp/Dockerfile, webapp/.dockerignore, webapp/docker-compose.yml   # Task 9
.github/workflows/webapp-tests.yml          # Task 9
README.md (root, short pointer section), WEBAPP_PLAN.md (status ticks) # Task 9
```

Execution waves (the controller runs tasks within a wave in parallel; waves are sequential):
- Wave A: Task 1 ∥ Task 2
- Wave B: Task 3 ∥ Task 4 ∥ Task 5 ∥ Task 6 ∥ Task 7
- Wave C: Task 8 ∥ Task 9

---

### Task 1: Web app scaffold and the reference page (thermo log)

**Files:**
- Create: `webapp/pyproject.toml`, `webapp/README.md` (stub)
- Create: `webapp/src/lada_web/__init__.py`, `config.py`, `catalog.py`, `rendering.py`, `errors.py`, `uploads.py`, `main.py`
- Create: `webapp/src/lada_web/schemas/__init__.py`, `schemas/common.py`
- Create: `webapp/src/lada_web/services/__init__.py`, `services/export_service.py`, `services/parsers_service.py` (log only)
- Create: `webapp/src/lada_web/routers/__init__.py`, `routers/pages.py`, `routers/parsers.py` (log routes), and stub routers `routers/conformations.py`, `routers/rheology.py`, `routers/dynamics.py`, `routers/modifiers.py`
- Create: `webapp/src/lada_web/templates/base.html`, `index.html`, `_macros.html`, `error_page.html`, `partials/table_result.html`, `partials/error.html`, `parsers/log.html`
- Create: `webapp/src/lada_web/static/css/app.css`, `static/js/app.js`
- Test: `webapp/tests/conftest.py`, `test_app.py`, `test_uploads.py`, `test_export_service.py`, `test_parsers.py`

**Interfaces:**
- Consumes: `lada.parsers.read_lammps_log(path) -> ThermoData` with `.to_pandas() -> pd.DataFrame`; `lada.exporters.write_pgfplots_table(df, filepath, delimiter=..., fmt="%.6g")`.
- Produces (used verbatim by Tasks 3–8):
  - `config.Settings` (fields above), `config.get_settings(request) -> Settings`
  - `schemas.common.PlotSpec(x, y: list[str], x_label="", y_label="", x_log=False, y_log=False, series: dict[str, list[float|None]] = {})`
  - `schemas.common.TableResult(title, columns, rows, n_rows_total, truncated=False, meta={}, plot=None, download_stem)`
  - `services.export_service.Format = Literal["csv","pgfplots"]`, `write_table(df, directory, stem, fmt) -> Path`, `media_type(fmt) -> str`, `output_filename(stem, fmt) -> str`, `table_result_from_df(df, *, title, download_stem, max_rows, plot=None, meta=None) -> TableResult`
  - `uploads.save_upload(upload, directory, max_bytes) -> Path`, `uploads.get_workspace` (yield dependency), `uploads.download_workspace(settings)` (context manager), `uploads.file_download(path, workspace, media_type) -> FileResponse`, `uploads.safe_filename(name) -> str`
  - `errors.UploadTooLarge(filename, limit_mb)`, `errors.TrajectoryTooLarge(n_frames, limit)`
  - `rendering.table_response(request, result, download_url: str | None) -> Response`, `rendering.templates`, `rendering.is_htmx`, `rendering.wants_html`
  - `catalog.CALCULATIONS`, `catalog.get_calculation(section, slug)`, `catalog.calculations_by_section()`; each `Calculation` has `.section .slug .title .summary .input_kind .path .template`
  - Jinja macros in `_macros.html`: `file_field(name, label, accept, help)`, `number_field(name, label, value, step=None, min=None, max=None, help='', required=True)`, `text_field(name, label, value='', help='', placeholder='')`, `select_field(name, label, options, value=None, help='')` (options = list of `(value, label)` pairs), `textarea_field(name, label, value='', rows=6, help='', placeholder='')`, `checkbox_field(name, label, checked=False, help='')`, `calculate_button(label, url)`, `results_area()`
  - Template context for calculation pages: `calc` (the `Calculation`). Template globals: `nav` (= `calculations_by_section()`), `app_version`.
  - The routers module layout and the exact route style shown in `routers/parsers.py` below.

**Design brief (read before writing any template or CSS):** Invoke the `frontend-design:frontend-design` skill with the Skill tool and follow its two-pass process (design plan → review against the brief → build). Subject: *LaDa*, a workbench that lets LAMMPS users run polymer-physics post-processing (radius of gyration, end-to-end vectors, autocorrelations, stress relaxation moduli) by uploading simulation output files. Audience: MD researchers who know LAMMPS but not Python. Primary job of every page: upload one file, set a few physical parameters, read a results table and plot, download the numbers. Personality from the package README: "simple, reliable, able to run just about anywhere without unnecessary luxury features" — named after the LADA car. Make deliberate palette/type/layout choices for *that* subject (an instrument, not a marketing site); avoid the generic looks the skill lists (cream + terracotta, black + acid green, broadsheet hairlines, identical rounded cards, all-caps eyebrows, middle-dot metadata strings, monospace labels, arrows in buttons). Use Pico.css for accessible form defaults and override its CSS variables in `app.css` for identity; numeric tables use `font-variant-numeric: tabular-nums`. Layout: persistent navigation listing all calculations grouped by section (collapsible on narrow screens using Alpine), a single content column of at most ~80 characters for prose, tables in an `overflow-x: auto` container. Quality floor: responsive to 360 px, visible focus rings, `prefers-reduced-motion` respected, `<label>` for every input. Write your design plan (palette hexes, typefaces, layout sketch, the one bold element) into your report before coding.

- [ ] **Step 1: Create the project files**

`webapp/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lada-web"
version = "0.1.0"
description = "Browser front-end for the lada LAMMPS analysis package"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
  "lada>=1.2.0,<2.0.0",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "jinja2>=3.1",
  "pydantic-settings>=2.4",
]

[project.optional-dependencies]
dev = ["pytest", "httpx2", "ruff"]

[tool.hatch.build.targets.wheel]
packages = ["src/lada_web"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["SIM105", "SIM108", "SIM114"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = [
  "ignore:The anyio.abc.BlockingPortal alias is deprecated:DeprecationWarning",
]
```

`webapp/README.md` stub (Task 9 replaces it):

```markdown
# lada-web

Browser front-end for the `lada` package. Install from the repository root with
`pip install -e . -e "./webapp[dev]"` and run `uvicorn lada_web.main:app --reload`.
```

`webapp/src/lada_web/__init__.py`:

```python
"""lada-web: a browser front-end for the lada LAMMPS analysis package."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Install the package into the worktree venv**

Run from the repo root: `.venv/bin/python -m pip install -e "./webapp[dev]"`. Expected: `Successfully installed lada-web-0.1.0` (dependencies are already present).

- [ ] **Step 3: Write `config.py`, `schemas/common.py`, `catalog.py`**

`config.py`:

```python
from pathlib import Path

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime limits. Every field can be overridden with a LADA_WEB_* environment variable."""

    model_config = SettingsConfigDict(env_prefix="LADA_WEB_", extra="ignore")

    max_upload_mb: float = 50
    max_trajectory_mb: float = 500
    max_trajectory_frames: int = 3000
    max_table_rows: int = 500
    temp_root: Path | None = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_trajectory_bytes(self) -> int:
        return int(self.max_trajectory_mb * 1024 * 1024)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
```

`schemas/common.py`:

```python
from pydantic import BaseModel, Field

Cell = float | int | str | None


class PlotSpec(BaseModel):
    """Which result columns to plot. `series` holds the full-length data for x and every y."""

    x: str
    y: list[str]
    x_label: str = ""
    y_label: str = ""
    x_log: bool = False
    y_log: bool = False
    series: dict[str, list[float | None]] = Field(default_factory=dict)


class TableResult(BaseModel):
    """A computed table: the rows shown on screen (capped) plus metadata and an optional plot."""

    title: str
    columns: list[str]
    rows: list[list[Cell]]
    n_rows_total: int
    truncated: bool = False
    meta: dict[str, str | int | float] = Field(default_factory=dict)
    plot: PlotSpec | None = None
    download_stem: str
```

`catalog.py` — the complete registry. Titles/summaries are user-facing copy; keep them as written unless the design pass (Task 8) improves them:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    blurb: str


@dataclass(frozen=True)
class Calculation:
    section: str
    slug: str
    title: str
    summary: str
    input_kind: str  # "dump" | "log" | "data" | "acf" | "npz"

    @property
    def path(self) -> str:
        return f"/{self.section}/{self.slug}"

    @property
    def template(self) -> str:
        return f"{self.section}/{self.slug}.html"


SECTIONS: list[Section] = [
    Section("parsers", "Read files", "Turn LAMMPS output files into tables you can inspect and download."),
    Section("conformations", "Chain conformations", "Size and shape of polymer chains from a dump trajectory."),
    Section("dynamics", "Dynamics", "Time correlation functions from a NumPy .npz trajectory."),
    Section("rheology", "Rheology", "Relaxation modulus and dynamic moduli from a stress autocorrelation file."),
    Section("modifiers", "Topology tools", "Rewrite a LAMMPS data file and download the result."),
]

CALCULATIONS: list[Calculation] = [
    Calculation("parsers", "dump", "Dump trajectory", "Preview the frames of a dump file or load the full atom table.", "dump"),
    Calculation("parsers", "log", "Thermo log", "Extract the thermodynamic output table from a LAMMPS log file.", "log"),
    Calculation("parsers", "data", "Data file", "Read one section (Atoms, Bonds, Masses, ...) of a LAMMPS data file.", "data"),
    Calculation("parsers", "acf", "Stress ACF file", "Read the last timestep block of a fix ave/correlate output file.", "acf"),
    Calculation("conformations", "rg", "Radius of gyration", "Ensemble-average Rg² of the chains at every timestep.", "dump"),
    Calculation("conformations", "ree", "End-to-end distance", "Ensemble-average Ree² of the chains at every timestep.", "dump"),
    Calculation("conformations", "ree-vectors", "End-to-end vectors", "The end-to-end vector of every chain at every timestep.", "dump"),
    Calculation("dynamics", "segment-acf", "End-to-end vector autocorrelation", "Normalized autocorrelation of the chain end-to-end vectors.", "npz"),
    Calculation("dynamics", "rouse-mode", "Rouse mode autocorrelation", "Normalized autocorrelation of one Rouse mode p.", "npz"),
    Calculation("dynamics", "isf", "Intermediate scattering function", "Coherent F(q,t)/F(q,0) at one scattering vector magnitude.", "npz"),
    Calculation("rheology", "stress-relaxation", "Stress relaxation modulus", "G(t) by the Green-Kubo and full stress relaxation methods.", "acf"),
    Calculation("rheology", "dynamic-moduli", "Dynamic moduli", "G'(ω) and G''(ω) from a Prony-series fit of G(t).", "acf"),
    Calculation("modifiers", "rewrite-end-beads", "Rewrite end beads", "Give every chain end a new atom type and download the modified data file.", "data"),
]


def get_calculation(section: str, slug: str) -> Calculation | None:
    for calc in CALCULATIONS:
        if calc.section == section and calc.slug == slug:
            return calc
    return None


def calculations_by_section() -> list[tuple[Section, list[Calculation]]]:
    return [(s, [c for c in CALCULATIONS if c.section == s.key]) for s in SECTIONS]
```

- [ ] **Step 4: Write `rendering.py`, `errors.py`, `uploads.py`**

`rendering.py`:

```python
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates

from . import __version__
from .catalog import calculations_by_section
from .schemas.common import TableResult

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals.update(nav=calculations_by_section(), app_version=__version__)


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def table_response(request: Request, result: TableResult, download_url: str | None) -> Response:
    """HTML partial for htmx requests, JSON for everything else."""
    if is_htmx(request):
        context = {
            "result": result,
            "result_json": result.model_dump_json(),
            "download_url": download_url,
        }
        return templates.TemplateResponse(request, "partials/table_result.html", context)
    return JSONResponse(result.model_dump(mode="json"))
```

`errors.py`:

```python
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .rendering import is_htmx, templates, wants_html


class UploadTooLarge(Exception):
    def __init__(self, filename: str, limit_mb: float) -> None:
        super().__init__(f"'{filename}' is larger than the {limit_mb:g} MB upload limit.")


class TrajectoryTooLarge(Exception):
    def __init__(self, n_frames: int, limit: int) -> None:
        super().__init__(
            f"The trajectory has {n_frames} frames; this server accepts at most {limit}. "
            "Save fewer frames or increase LADA_WEB_MAX_TRAJECTORY_FRAMES."
        )


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


def error_response(
    request: Request, status_code: int, message: str, *, title: str = "That didn't work"
) -> Response:
    context = {"title": title, "message": message}
    if is_htmx(request):
        return templates.TemplateResponse(
            request, "partials/error.html", context, status_code=status_code
        )
    if wants_html(request):
        return templates.TemplateResponse(request, "error_page.html", context, status_code=status_code)
    return JSONResponse({"detail": message}, status_code=status_code)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    async def domain_error(request: Request, exc: Exception) -> Response:
        return error_response(
            request, 422, _message(exc), title="The file or settings could not be processed"
        )

    @app.exception_handler(UploadTooLarge)
    @app.exception_handler(TrajectoryTooLarge)
    async def too_large(request: Request, exc: Exception) -> Response:
        return error_response(request, 413, _message(exc), title="Too large")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        if is_htmx(request) or wants_html(request):
            problems = [
                f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
                for e in exc.errors()
            ]
            return error_response(request, 422, "; ".join(problems), title="Check the form")
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return error_response(request, exc.status_code, str(exc.detail), title="Not available")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        return error_response(
            request,
            500,
            f"Unexpected error while processing the request: {type(exc).__name__}: {exc}",
            title="Something went wrong",
        )
```

`uploads.py`:

```python
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .config import Settings, get_settings
from .errors import UploadTooLarge

CHUNK_SIZE = 1024 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str | None, default: str = "upload") -> str:
    """Basename only, restricted to [A-Za-z0-9._-]; never empty."""
    cleaned = _UNSAFE.sub("_", Path(name or "").name).strip("._")
    return cleaned or default


def new_workspace(settings: Settings) -> Path:
    if settings.temp_root is not None:
        settings.temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="lada-web-", dir=settings.temp_root))


def cleanup_workspace(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def save_upload(upload: UploadFile, directory: Path, max_bytes: int) -> Path:
    """Stream an upload into `directory` in 1 MiB chunks; reject it once it exceeds max_bytes."""
    name = safe_filename(upload.filename)
    limit_mb = max_bytes / (1024 * 1024)
    if upload.size is not None and upload.size > max_bytes:
        raise UploadTooLarge(name, limit_mb)
    dest = directory / name
    written = 0
    try:
        with dest.open("wb") as out:
            for chunk in iter(lambda: upload.file.read(CHUNK_SIZE), b""):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLarge(name, limit_mb)
                out.write(chunk)
    except UploadTooLarge:
        dest.unlink(missing_ok=True)
        raise
    return dest


def get_workspace(settings: Annotated[Settings, Depends(get_settings)]) -> Iterator[Path]:
    """Per-request temp directory for view routes; removed when the request finishes."""
    ws = new_workspace(settings)
    try:
        yield ws
    finally:
        cleanup_workspace(ws)


@contextmanager
def download_workspace(settings: Settings) -> Iterator[Path]:
    """Temp directory for download routes: removed on error here, otherwise by file_download."""
    ws = new_workspace(settings)
    try:
        yield ws
    except BaseException:
        cleanup_workspace(ws)
        raise


def file_download(path: Path, workspace: Path, media_type: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        background=BackgroundTask(cleanup_workspace, workspace),
    )
```

- [ ] **Step 5: Write `services/export_service.py` and the log service**

`services/export_service.py`:

```python
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from lada.exporters import write_pgfplots_table

from ..schemas.common import Cell, PlotSpec, TableResult

Format = Literal["csv", "pgfplots"]

# fmt -> (delimiter for write_pgfplots_table, file suffix, media type)
_FORMATS: dict[str, tuple[str, str, str]] = {
    "csv": (",", ".csv", "text/csv"),
    "pgfplots": (" ", ".dat", "text/plain"),
}


def media_type(fmt: Format) -> str:
    return _FORMATS[fmt][2]


def output_filename(stem: str, fmt: Format) -> str:
    return f"{stem}{_FORMATS[fmt][1]}"


def write_table(df: pd.DataFrame, directory: Path, stem: str, fmt: Format) -> Path:
    """One code path for both formats: write_pgfplots_table with ',' (csv) or ' ' (pgfplots)."""
    path = directory / output_filename(stem, fmt)
    write_pgfplots_table(df, str(path), delimiter=_FORMATS[fmt][0])
    return path


def _cell(value: object) -> Cell:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _series(df: pd.DataFrame, column: str) -> list[float | None]:
    return [_cell(v) for v in df[column].to_numpy()]  # type: ignore[misc]


def table_result_from_df(
    df: pd.DataFrame,
    *,
    title: str,
    download_stem: str,
    max_rows: int,
    plot: PlotSpec | None = None,
    meta: dict[str, str | int | float] | None = None,
) -> TableResult:
    """Cap the on-screen rows at max_rows; plot series always carry every row."""
    shown = df.iloc[:max_rows]
    rows = [[_cell(v) for v in row] for row in shown.itertuples(index=False, name=None)]
    if plot is not None:
        plot = plot.model_copy(
            update={"series": {name: _series(df, name) for name in [plot.x, *plot.y]}}
        )
    return TableResult(
        title=title,
        columns=[str(c) for c in df.columns],
        rows=rows,
        n_rows_total=len(df),
        truncated=len(df) > max_rows,
        meta=dict(meta or {}),
        plot=plot,
        download_stem=download_stem,
    )
```

`services/parsers_service.py` (Task 3 extends this file):

```python
from pathlib import Path

import pandas as pd
from lada.parsers import read_lammps_log


def log_table(path: Path) -> pd.DataFrame:
    df = read_lammps_log(str(path)).to_pandas()
    if df.empty:
        raise ValueError(
            "No thermodynamic table was found in this file. Upload a LAMMPS log that "
            "contains a thermo output block."
        )
    return df
```

- [ ] **Step 6: Write the routers and `main.py`**

`routers/pages.py`:

```python
from fastapi import APIRouter, HTTPException, Request

from ..catalog import calculations_by_section, get_calculation
from ..rendering import templates

router = APIRouter(include_in_schema=False)


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"groups": calculations_by_section()})


@router.get("/{section}/{slug}")
def calculation_page(request: Request, section: str, slug: str):
    calc = get_calculation(section, slug)
    if calc is None:
        raise HTTPException(status_code=404, detail="There is no page at this address.")
    return templates.TemplateResponse(request, calc.template, {"calc": calc})
```

`routers/parsers.py` — this is the reference style for every later router:

```python
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from ..config import Settings, get_settings
from ..rendering import table_response
from ..services import parsers_service
from ..services.export_service import Format, media_type, table_result_from_df, write_table
from ..uploads import download_workspace, file_download, get_workspace, save_upload

router = APIRouter(prefix="/api/parsers", tags=["parsers"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
WorkspaceDep = Annotated[Path, Depends(get_workspace)]
FileDep = Annotated[UploadFile, File()]
FormatQuery = Annotated[Format, Query()]


@router.post("/log/table")
def log_table(request: Request, file: FileDep, settings: SettingsDep, ws: WorkspaceDep):
    path = save_upload(file, ws, settings.max_upload_bytes)
    df = parsers_service.log_table(path)
    result = table_result_from_df(
        df,
        title="Thermodynamic output",
        download_stem=f"{path.stem}_thermo",
        max_rows=settings.max_table_rows,
        meta={"rows": len(df), "columns": len(df.columns)},
    )
    return table_response(request, result, download_url="/api/parsers/log/table/download")


@router.post("/log/table/download")
def log_table_download(file: FileDep, settings: SettingsDep, fmt: FormatQuery = "csv"):
    with download_workspace(settings) as ws:
        path = save_upload(file, ws, settings.max_upload_bytes)
        df = parsers_service.log_table(path)
        out = write_table(df, ws, f"{path.stem}_thermo", fmt)
        return file_download(out, ws, media_type(fmt))
```

Stub routers (`conformations.py`, `rheology.py`, `dynamics.py`, `modifiers.py`) each contain only:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/<section>", tags=["<section>"])
```

`main.py`:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .errors import register_error_handlers
from .rendering import STATIC_DIR
from .routers import conformations, dynamics, modifiers, pages, parsers, rheology


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="LaDa web",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings or Settings()
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_error_handlers(app)
    for router in (
        parsers.router,
        conformations.router,
        dynamics.router,
        rheology.router,
        modifiers.router,
        pages.router,  # last: it owns the catch-all /{section}/{slug}
    ):
        app.include_router(router)
    return app


app = create_app()
```

- [ ] **Step 7: Write the templates, CSS and JS**

Required structure (the design is yours; the IDs, attributes and macro signatures are not):

`base.html` — `<!doctype html>`, `lang="en"`, viewport meta, `<title>{% block title %}{% endblock %} – LaDa</title>`, the four CDN assets from Global Constraints plus `/static/css/app.css` and `/static/js/app.js`, a `{% block extra_head %}{% endblock %}`, a skip link, a header naming the app, navigation built from the `nav` global (each link gets `aria-current="page"` when `calc` is defined and `calc.path` matches; on narrow screens the nav collapses behind an Alpine-controlled button), `<main>{% block content %}{% endblock %}</main>`, a footer showing `app_version` and a link to `/api/docs`, and `{% block scripts %}{% endblock %}`.

`_macros.html` — every macro renders a `<label>` wrapping or referencing its control, an optional `<small>` help text, and uses the field `name` as the control `id`. Signatures are in the Interfaces block. `calculate_button` and `results_area` render exactly:

```jinja
{% macro calculate_button(label, url) -%}
<button type="submit" hx-post="{{ url }}" hx-target="#results" hx-swap="innerHTML"
        hx-indicator="#results-indicator" hx-disabled-elt="this">{{ label }}</button>
{%- endmacro %}

{% macro results_area() -%}
<div id="results-indicator" class="htmx-indicator" role="status">Calculating…</div>
<div id="results" aria-live="polite"></div>
{%- endmacro %}
```

`partials/table_result.html` — receives `result`, `result_json`, `download_url`:

```jinja
<section class="result" id="result" data-result="{{ result_json }}">
  <header class="result-header">
    <h2>{{ result.title }}</h2>
    <p class="result-summary">
      {{ result.n_rows_total }} rows, {{ result.columns | length }} columns.
      {%- for key, value in result.meta.items() %} {{ key | replace('_', ' ') | capitalize }}: {{ value }}.{% endfor %}
    </p>
  </header>
  {% if result.plot %}
  <div class="plot-controls">
    <label><input type="checkbox" data-axis="xaxis" {% if result.plot.x_log %}checked{% endif %}> Log x</label>
    <label><input type="checkbox" data-axis="yaxis" {% if result.plot.y_log %}checked{% endif %}> Log y</label>
  </div>
  <div class="plot" id="result-plot"></div>
  {% endif %}
  <div class="table-scroll">
    <table class="numbers">
      <thead><tr>{% for c in result.columns %}<th scope="col">{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
        {% for row in result.rows %}<tr>{% for v in row %}<td>{{ '' if v is none else v }}</td>{% endfor %}</tr>{% endfor %}
      </tbody>
    </table>
  </div>
  {% if result.truncated %}
  <p class="result-note">Showing the first {{ result.rows | length }} of {{ result.n_rows_total }} rows. Downloads contain every row.</p>
  {% endif %}
  {% if download_url %}
  <div class="downloads">
    <button type="submit" form="calc-form" formaction="{{ download_url }}?fmt=csv"
            formmethod="post" formenctype="multipart/form-data">Download CSV</button>
    <button type="submit" form="calc-form" formaction="{{ download_url }}?fmt=pgfplots"
            formmethod="post" formenctype="multipart/form-data">Download pgfplots table</button>
  </div>
  {% endif %}
</section>
<script>window.ladaMountResult(document.getElementById('result'));</script>
```

`partials/error.html` — `<section class="result result-error" id="result" role="alert"><h2>{{ title }}</h2><p>{{ message }}</p></section>`.

`error_page.html` — extends `base.html`; shows the same title/message and a link back to `/`.

`index.html` — extends `base.html`; a short introduction (what the app does, that nothing is stored server-side), then for each `(section, calcs)` in `groups` a heading, the section blurb, and a link per calculation with its summary and input kind ("Needs: LAMMPS dump file" etc.).

`parsers/log.html` — the reference page; every later page copies this shape:

```jinja
{% extends "base.html" %}
{% import "_macros.html" as f %}
{% block title %}{{ calc.title }}{% endblock %}
{% block content %}
<article class="calc">
  <header>
    <h1>{{ calc.title }}</h1>
    <p>Upload a LAMMPS log file and get the thermodynamic output (the columns printed by
       <code>thermo_style</code>) as a table. Multiple runs in one log are concatenated.</p>
  </header>
  <form id="calc-form" method="post" enctype="multipart/form-data" hx-encoding="multipart/form-data">
    {{ f.file_field('file', 'LAMMPS log file', '.lammps,.log,.txt',
                    'The log.lammps file LAMMPS writes while it runs.') }}
    {{ f.calculate_button('Read thermo table', '/api/parsers/log/table') }}
  </form>
  {{ f.results_area() }}
</article>
{% endblock %}
```

`static/js/app.js` — exactly these behaviours, nothing else:
1. `htmx:beforeSwap` listener on `document.body`: when `evt.detail.xhr.status >= 400`, set `evt.detail.shouldSwap = true` and `evt.detail.isError = false` so error partials render into `#results`.
2. `htmx:sendError` listener: put a plain-language "The server could not be reached" message into `#results`.
3. `window.ladaMountResult(section)`: parse `section.dataset.result`; if `plot` is present, build one Plotly scatter (`mode: 'lines+markers'`) trace per `plot.y` from `plot.series`, call `Plotly.newPlot(section.querySelector('#result-plot'), traces, layout, {responsive: true, displaylogo: false})` with axis titles from `x_label`/`y_label` (fall back to the column names) and axis `type` `'log'` where `x_log`/`y_log` is true; wire each `.plot-controls input[data-axis]` change to `Plotly.relayout` toggling that axis between `'log'` and `'linear'`.

`static/css/app.css` — identity, layout, `.htmx-indicator` (hidden until `.htmx-request`), `.table-scroll {overflow-x:auto}`, `.numbers td {font-variant-numeric: tabular-nums; text-align: right}`, `.result-error` styling, `.plot {min-height: 360px}`, responsive nav, focus rings, reduced-motion media query.

- [ ] **Step 8: Write the tests**

`webapp/tests/conftest.py`:

```python
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from lada_web.config import Settings
from lada_web.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data"


@pytest.fixture
def data_dir() -> Path:
    assert DATA_DIR.is_dir(), f"Root fixtures directory missing: {DATA_DIR}"
    return DATA_DIR


@pytest.fixture
def temp_root(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def settings(temp_root: Path) -> Settings:
    return Settings(
        max_upload_mb=1,
        max_trajectory_mb=1,
        max_trajectory_frames=100,
        max_table_rows=20,
        temp_root=temp_root,
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def htmx() -> dict[str, str]:
    return {"HX-Request": "true"}


def upload(path: Path) -> dict:
    """files= argument for TestClient.post."""
    return {"file": (path.name, path.read_bytes())}
```

`test_app.py` — `test_index_lists_every_calculation` (GET `/` is 200 and contains every `calc.path` and `calc.title` from `CALCULATIONS`); `test_reference_page_has_form_contract` (GET `/parsers/log` contains `id="calc-form"`, `hx-post="/api/parsers/log/table"`, `id="results"`, `name="file"`, `hx-encoding="multipart/form-data"`, and the form tag has no `action=`); `test_unknown_page_is_404_json_and_html` (JSON `detail` for default Accept; HTML containing the message when `Accept: text/html`); `test_static_assets_served` (`/static/css/app.css` and `/static/js/app.js` are 200); `test_api_docs_available` (`/api/docs` 200); `test_settings_are_injected` (`app.state.settings.max_upload_mb == 1`).

`test_uploads.py` — build `starlette.datastructures.UploadFile(io.BytesIO(data), filename=..., size=len(data))` and assert: `save_upload` writes the bytes and returns `directory / "sample.txt"`; a 1.5 MB payload with `max_bytes=1 MiB` raises `UploadTooLarge` and leaves no file; the same with `size=None` (streaming path) also raises and leaves no file; `safe_filename("../../etc/passwd") == "passwd"`, `safe_filename("") == "upload"`, `safe_filename("my file (1).npz") == "my_file_1_.npz"`; `download_workspace` removes the directory when the body raises and keeps it otherwise.

`test_export_service.py` — round trips: `write_table(df, tmp, "t", "csv")` → `pd.read_csv` equals `df` (`np.allclose`); `"pgfplots"` → file named `t.dat`, `np.loadtxt(path, skiprows=1)` equals `df.to_numpy()`, first line is the space-joined header; `table_result_from_df` with `max_rows=2` on 5 rows → `len(rows)==2`, `n_rows_total==5`, `truncated is True`; NaN becomes `None`; `np.int64` cells become `int`; with a `PlotSpec(x="t", y=["g"])` the returned `plot.series["t"]` has all 5 values.

`test_parsers.py` (log slice; Task 3 adds the rest):

```python
import io

import numpy as np
import pandas as pd

from .conftest import upload

LOG_URL = "/api/parsers/log/table"
LOG_COLUMNS = ["Step", "Temp", "Press", "Vol", "Lx", "Ly", "Lz", "PotEng", "KinEng", "TotEng"]


def test_log_table_json(client, data_dir):
    r = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == LOG_COLUMNS
    assert body["n_rows_total"] == 5 and len(body["rows"]) == 5
    assert body["rows"][0][0] == 0 and body["rows"][-1][0] == 4000
    assert body["download_stem"] == "sample_log_thermo"


def test_log_table_htmx_partial(client, data_dir, htmx):
    r = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps"), headers=htmx)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<table" in r.text
    assert 'formaction="/api/parsers/log/table/download?fmt=csv"' in r.text
    assert 'formaction="/api/parsers/log/table/download?fmt=pgfplots"' in r.text
    assert 'form="calc-form"' in r.text


def test_log_download_csv_matches_view(client, data_dir):
    view = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps")).json()
    r = client.post(LOG_URL + "/download?fmt=csv", files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="sample_log_thermo.csv"'
    df = pd.read_csv(io.BytesIO(r.content))
    assert list(df.columns) == view["columns"]
    assert np.allclose(df.to_numpy(), np.array(view["rows"], dtype=float))


def test_log_download_pgfplots_matches_view(client, data_dir):
    view = client.post(LOG_URL, files=upload(data_dir / "sample_log.lammps")).json()
    r = client.post(LOG_URL + "/download?fmt=pgfplots", files=upload(data_dir / "sample_log.lammps"))
    assert r.headers["content-disposition"] == 'attachment; filename="sample_log_thermo.dat"'
    assert r.text.splitlines()[0] == " ".join(view["columns"])
    data = np.loadtxt(io.BytesIO(r.content), skiprows=1)
    assert np.allclose(data, np.array(view["rows"], dtype=float))


def test_log_download_rejects_unknown_format(client, data_dir):
    r = client.post(LOG_URL + "/download?fmt=xlsx", files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 422


def test_missing_file_is_422(client, htmx):
    assert client.post(LOG_URL).status_code == 422
    r = client.post(LOG_URL, headers=htmx)
    assert r.status_code == 422 and 'role="alert"' in r.text and "file" in r.text


def test_not_a_log_file_is_422_with_message(client, data_dir, htmx):
    r = client.post(LOG_URL, files=upload(data_dir / "sample.data"))
    assert r.status_code == 422 and r.json()["detail"]
    r = client.post(LOG_URL, files=upload(data_dir / "sample.data"), headers=htmx)
    assert r.status_code == 422 and 'role="alert"' in r.text


def test_oversized_upload_is_413_and_leaves_nothing_behind(client, temp_root):
    blob = b"x" * (1024 * 1024 + 1)
    r = client.post(LOG_URL, files={"file": ("big.lammps", blob)})
    assert r.status_code == 413
    assert "upload limit" in r.json()["detail"]
    assert not temp_root.exists() or list(temp_root.iterdir()) == []
```

If `read_lammps_log` raises something other than `ValueError`/`KeyError` for `sample.data`, catch that specific exception type inside `parsers_service.log_table` and re-raise it as the `ValueError` shown in Step 5 — the message must still be user-readable.

- [ ] **Step 9: Run everything, lint, commit**

Run: `cd webapp && ../.venv/bin/python -m pytest -q -p no:cacheprovider` → all green, no warnings. Then `../.venv/bin/ruff format src tests && ../.venv/bin/ruff check src tests` → clean. From the repo root `.venv/bin/python -m pytest -q -p no:cacheprovider` → 99 passed (unchanged core).

Commit only your files:

```bash
git add webapp/pyproject.toml webapp/README.md webapp/src webapp/tests
git commit -m "feat(webapp): scaffold lada-web with thermo-log reference page"
```

---

### Task 2: Root test fixtures for trajectories, stress ACF and molecular data

**Files:**
- Create: `tests/data/generate_sample_trajectory.py`, `tests/data/sample_trajectory.npz`
- Create: `tests/data/generate_sample_stress_acf.py`, `tests/data/sample_stress_acf.txt`
- Create: `tests/data/sample_molecular.data`
- Create: `tests/test_modifiers/test_data_modifier.py`, `tests/test_fixtures.py`
- Modify: `tests/data/README.md` (append a "Committed fixtures added in September 2026" section)

**Interfaces:**
- Consumes: `lada.parsers.read_lammps_acf`, `lada.parsers.read_data_file`, `lada.modifiers.rewrite_end_beads`, `lada.analysis.calculate_segment_acf_from_trajectory`.
- Produces (relied on by Tasks 5–8): `sample_trajectory.npz` with key `coords`, shape `(32, 32, 3)`, float64, atoms ordered chain-major (4 chains × 8 beads, atom index = chain·8 + bead); `sample_stress_acf.txt` whose last block has timestep 1000 and 100 lag rows at `lag_time = 0, 0.5, …, 49.5`, columns `ACF_Sxy,ACF_Sxz,ACF_Syz,ACF_Nxy,ACF_Nxz,ACF_Nyz`; `sample_molecular.data` with `Atoms # molecular`, 12 atoms in 3 chains (atom ids 1–4, 5–8, 9–12), 2 atom types, `Masses` and `Pair Coeffs` sections.

- [ ] **Step 1: Write the trajectory generator and run it**

`tests/data/generate_sample_trajectory.py`:

```python
"""Generate tests/data/sample_trajectory.npz: a small deterministic synthetic polymer trajectory.

Run from anywhere:  python tests/data/generate_sample_trajectory.py

Layout: N_CHAINS chains of BEADS_PER_CHAIN beads. Atom index = chain * BEADS_PER_CHAIN + bead,
0-indexed and ascending, i.e. the "uniform" topology convention used by lada-web. Frame 0 is a
freely-jointed chain per molecule; every later frame adds an independent Gaussian kick to each
bead, so end-to-end and Rouse-mode autocorrelations decay within the 32 frames.
"""

from pathlib import Path

import numpy as np

N_FRAMES = 32
N_CHAINS = 4
BEADS_PER_CHAIN = 8
BOND_LENGTH = 1.0
KICK_SIGMA = 0.35
SEED = 20260908
OUT = Path(__file__).with_name("sample_trajectory.npz")


def build_coords(rng: np.random.Generator) -> np.ndarray:
    n_atoms = N_CHAINS * BEADS_PER_CHAIN
    coords = np.empty((N_FRAMES, n_atoms, 3))
    origins = rng.uniform(-5.0, 5.0, size=(N_CHAINS, 3))
    steps = rng.normal(size=(N_CHAINS, BEADS_PER_CHAIN - 1, 3))
    steps *= BOND_LENGTH / np.linalg.norm(steps, axis=2, keepdims=True)
    first = np.concatenate(
        [origins[:, None, :], origins[:, None, :] + np.cumsum(steps, axis=1)], axis=1
    )
    coords[0] = first.reshape(n_atoms, 3)
    for t in range(1, N_FRAMES):
        coords[t] = coords[t - 1] + rng.normal(scale=KICK_SIGMA, size=(n_atoms, 3))
    return coords


def main(out: Path = OUT) -> None:
    rng = np.random.default_rng(SEED)
    np.savez_compressed(out, coords=build_coords(rng))


if __name__ == "__main__":
    main()
```

Run `.venv/bin/python tests/data/generate_sample_trajectory.py` and confirm `ls -l tests/data/sample_trajectory.npz` is under 40 KB and `np.load(...)["coords"].shape == (32, 32, 3)`.

- [ ] **Step 2: Write the stress-ACF generator and run it**

`tests/data/generate_sample_stress_acf.py`:

```python
"""Generate tests/data/sample_stress_acf.txt: a synthetic stress-autocorrelation file in the
`fix ave/correlate` layout that lada.parsers.read_lammps_acf reads and with the exact column
names lada.analysis.calc_stress_relaxation requires (sample_acf.txt uses acf_x/y/z instead).

Run from anywhere:  python tests/data/generate_sample_stress_acf.py

Two timestep blocks are written (t=0 reference, then 1000); read_lammps_acf uses the last one.
Each ACF is a two-exponential decay with a small seeded noise term and stays positive.
"""

from pathlib import Path

import numpy as np

COLUMNS = ["ACF_Sxy", "ACF_Sxz", "ACF_Syz", "ACF_Nxy", "ACF_Nxz", "ACF_Nyz"]
AMPLITUDES = {
    "ACF_Sxy": 1.00,
    "ACF_Sxz": 0.95,
    "ACF_Syz": 1.05,
    "ACF_Nxy": 3.60,
    "ACF_Nxz": 3.80,
    "ACF_Nyz": 4.00,
}
N_LAGS = 100
DT = 0.5
TAU_FAST = 2.0
TAU_SLOW = 20.0
SLOW_FRACTION = 0.3
SEED = 20260908
OUT = Path(__file__).with_name("sample_stress_acf.txt")


def acf_values(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    base = amplitude * (
        (1 - SLOW_FRACTION) * np.exp(-t / TAU_FAST) + SLOW_FRACTION * np.exp(-t / TAU_SLOW)
    )
    noise = rng.normal(scale=1e-3 * amplitude, size=t.shape)
    noise[0] = 0.0
    return np.maximum(base + noise, 1e-6)


def main(out: Path = OUT) -> None:
    rng = np.random.default_rng(SEED)
    t = np.arange(N_LAGS) * DT
    lines = [",".join(COLUMNS)]
    for timestep, scale in ((0, 1.0), (1000, 0.98)):
        lines.append(f"# Timestep: {timestep}")
        block = np.column_stack([acf_values(t, AMPLITUDES[c] * scale, rng) for c in COLUMNS])
        for lag, row in zip(t, block, strict=True):
            lines.append(f"{lag:g} " + " ".join(f"{v:.6f}" for v in row))
        lines.append("")
    out.write_text("\n".join(lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
```

Run it; confirm `read_lammps_acf("tests/data/sample_stress_acf.txt")` returns 100 rows, columns `["lag_time", *COLUMNS, "timestep"]`, `timestep == 1000`.

- [ ] **Step 3: Write `sample_molecular.data` exactly**

```
LAMMPS data file: 3 linear chains of 4 beads, molecular atom style (test fixture)

12 atoms
9 bonds
6 angles
0 dihedrals
0 impropers

2 atom types
1 bond types
1 angle types

-10.0 10.0 xlo xhi
-10.0 10.0 ylo yhi
-10.0 10.0 zlo zhi

Masses

1 1.0
2 2.0

Pair Coeffs # lj/cut

1 1.0 1.0
2 0.5 1.2

Bond Coeffs # harmonic

1 30.0 1.5

Angle Coeffs # harmonic

1 5.0 180.0

Atoms # molecular

1 1 1 -6.0 0.0 0.0
2 1 1 -5.0 0.0 0.0
3 1 1 -4.0 0.0 0.0
4 1 1 -3.0 0.0 0.0
5 2 1 -1.5 2.0 0.0
6 2 2 -0.5 2.0 0.0
7 2 1 0.5 2.0 0.0
8 2 1 1.5 2.0 0.0
9 3 1 3.0 -2.0 0.0
10 3 1 4.0 -2.0 0.0
11 3 1 5.0 -2.0 0.0
12 3 1 6.0 -2.0 0.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 5 6
5 1 6 7
6 1 7 8
7 1 9 10
8 1 10 11
9 1 11 12

Angles

1 1 1 2 3
2 1 2 3 4
3 1 5 6 7
4 1 6 7 8
5 1 9 10 11
6 1 10 11 12
```

- [ ] **Step 4: Write the failing tests**

`tests/test_modifiers/test_data_modifier.py`:

```python
"""Tests for lada.modifiers.rewrite_end_beads against the molecular-style fixture."""

from pathlib import Path

import numpy as np
import pytest

from lada.modifiers import rewrite_end_beads
from lada.parsers import read_data_file

DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURE = DATA / "sample_molecular.data"
CHAIN_ENDS = {1, 4, 5, 8, 9, 12}


@pytest.fixture
def rewritten(tmp_path):
    out = tmp_path / "rewritten.data"
    rewrite_end_beads(str(FIXTURE), str(out), new_end_type=3, base_type=1)
    return read_data_file(str(out))


def test_chain_ends_get_the_new_type(rewritten):
    atoms = rewritten.to_pandas("Atoms")
    typed = atoms.loc[atoms["type"] == 3, "id"].astype(int)
    assert set(typed) == CHAIN_ENDS


def test_interior_beads_keep_their_type(rewritten):
    original = read_data_file(str(FIXTURE)).to_pandas("Atoms")
    atoms = rewritten.to_pandas("Atoms")
    interior = ~atoms["id"].isin(CHAIN_ENDS)
    assert (atoms.loc[interior, "type"].to_numpy() == original.loc[interior, "type"].to_numpy()).all()


def test_atom_types_header_and_coefficients_are_extended(rewritten):
    assert rewritten.metadata["atom types"] == 3
    masses = rewritten.to_pandas("Masses")
    assert masses.loc[masses["atom_id"] == 3, "mass"].item() == pytest.approx(1.0)
    assert [3.0, 1.0, 1.0] in rewritten.get("Pair Coeffs").tolist()


def test_bonds_are_untouched(rewritten):
    original = read_data_file(str(FIXTURE))
    assert np.array_equal(rewritten.get("Bonds"), original.get("Bonds"))
```

`tests/test_fixtures.py`:

```python
"""The committed generated fixtures must match what their generator scripts produce."""

import importlib.util
from pathlib import Path

import numpy as np

from lada.analysis import calculate_segment_acf_from_trajectory
from lada.parsers import read_lammps_acf

DATA = Path(__file__).resolve().parent / "data"


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, DATA / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_generator_reproduces_committed_file(tmp_path):
    out = tmp_path / "regenerated.npz"
    _load("generate_sample_trajectory").main(out)
    committed = np.load(DATA / "sample_trajectory.npz")["coords"]
    assert committed.shape == (32, 32, 3)
    assert np.array_equal(np.load(out)["coords"], committed)


def test_stress_acf_generator_reproduces_committed_file(tmp_path):
    out = tmp_path / "regenerated.txt"
    _load("generate_sample_stress_acf").main(out)
    assert out.read_text() == (DATA / "sample_stress_acf.txt").read_text()


def test_stress_acf_fixture_has_the_schema_calc_stress_relaxation_needs():
    df = read_lammps_acf(DATA / "sample_stress_acf.txt")
    assert list(df.columns) == [
        "lag_time", "ACF_Sxy", "ACF_Sxz", "ACF_Syz", "ACF_Nxy", "ACF_Nxz", "ACF_Nyz", "timestep",
    ]
    assert len(df) == 100 and df["timestep"].iloc[0] == 1000
    assert df["lag_time"].iloc[1] == 0.5


def test_trajectory_fixture_works_with_dynamics_functions():
    pairs = np.array([[0, 7], [8, 15], [16, 23], [24, 31]])
    acf = calculate_segment_acf_from_trajectory(
        str(DATA / "sample_trajectory.npz"), pairs, time_per_frame=1.0
    )
    assert acf.shape == (32, 2) and acf[0, 1] == 1.0
```

Run `.venv/bin/python -m pytest tests/test_modifiers tests/test_fixtures.py -q -p no:cacheprovider` and confirm all pass on the generated files (the generators are the implementation; these tests guard them).

- [ ] **Step 5: Document the fixtures**

Append to `tests/data/README.md` a section listing the three new files, what each is for (`.npz` for `lada.analysis.dynamics` and the web app's dynamics pages; stress ACF for `calc_stress_relaxation`; molecular data for `rewrite_end_beads`), and the two regeneration commands.

- [ ] **Step 6: Lint, run the whole core suite, commit**

`.venv/bin/ruff format tests && .venv/bin/ruff check tests` clean; `.venv/bin/python -m pytest -q -p no:cacheprovider` all green.

```bash
git add tests/data/generate_sample_trajectory.py tests/data/sample_trajectory.npz \
        tests/data/generate_sample_stress_acf.py tests/data/sample_stress_acf.txt \
        tests/data/sample_molecular.data tests/data/README.md \
        tests/test_modifiers/test_data_modifier.py tests/test_fixtures.py
git commit -m "test: add trajectory, stress-ACF and molecular fixtures with modifier tests"
```

---

### Task 3: Parsers pages — dump, data file, stress ACF

**Files:**
- Create: `webapp/src/lada_web/schemas/parsers.py`, `templates/parsers/dump.html`, `templates/parsers/data.html`, `templates/parsers/acf.html`
- Modify: `webapp/src/lada_web/services/parsers_service.py`, `routers/parsers.py`, `webapp/tests/test_parsers.py`

**Interfaces:**
- Consumes (Task 1): everything in Task 1's Produces block; the existing log routes are the style guide. `lada.parsers.dump_frames` yields `DumpFrame` with `.metadata["TIMESTEP"]` (int), a key starting with `"BOX BOUNDS"` holding an `np.ndarray` of shape `(3,2)` or `(3,3)`, `.columns` (list[str]) and `.data` (ndarray, rows = atoms). `read_dump(path)` returns the concatenated table with a leading `timestep` column. `read_data_file(path).to_pandas(section)` raises `ValueError` for an unknown section name and `KeyError` for a section absent from the file. `lada.parsers.data_parser.ALLOWED_SECTIONS` is the tuple of valid names.
- Produces: routes `POST /api/parsers/dump/preview` (view only, no download), `POST /api/parsers/dump/table` (+`/download`), `POST /api/parsers/data/table` (+`/download`, form field `section: str = "Atoms"`), `POST /api/parsers/acf/table` (+`/download`). Result columns: preview `[timestep, n_atoms, box_x, box_y, box_z, columns]`; dump table = `read_dump` columns; data table = `to_pandas(section)` columns; acf table = `read_lammps_acf` columns.

- [ ] **Step 1: Write the failing tests** in `test_parsers.py` (append; keep the log tests):

```python
DUMP = "sample_dump.dump"   # 3 frames × 50 atoms, columns id mol type xu yu zu


def test_dump_preview_summarises_each_frame(client, data_dir):
    r = client.post("/api/parsers/dump/preview", files=upload(data_dir / DUMP))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["timestep", "n_atoms", "box_x", "box_y", "box_z", "columns"]
    assert body["n_rows_total"] == 3
    assert body["rows"][0][1] == 50 and body["rows"][0][2] == 20.0
    assert body["rows"][0][5] == "id mol type xu yu zu"
    assert body["plot"] is None


def test_dump_preview_htmx_has_no_download_buttons(client, data_dir, htmx):
    r = client.post("/api/parsers/dump/preview", files=upload(data_dir / DUMP), headers=htmx)
    assert r.status_code == 200 and "formaction=" not in r.text


def test_dump_table_is_capped_on_screen_but_full_on_download(client, data_dir):
    view = client.post("/api/parsers/dump/table", files=upload(data_dir / DUMP)).json()
    assert view["columns"] == ["timestep", "id", "mol", "type", "xu", "yu", "zu"]
    assert view["n_rows_total"] == 150 and len(view["rows"]) == 20 and view["truncated"]
    r = client.post("/api/parsers/dump/table/download?fmt=csv", files=upload(data_dir / DUMP))
    df = pd.read_csv(io.BytesIO(r.content))
    assert len(df) == 150 and list(df.columns) == view["columns"]
    assert np.allclose(df.to_numpy()[:20], np.array(view["rows"], dtype=float))


@pytest.mark.parametrize("section,first_col", [("Atoms", "id"), ("Bonds", "bond_id"), ("Masses", "atom_id")])
def test_data_table_sections(client, data_dir, section, first_col):
    r = client.post("/api/parsers/data/table", files=upload(data_dir / "sample.data"), data={"section": section})
    assert r.status_code == 200, r.text
    assert r.json()["columns"][0] == first_col


def test_data_table_invalid_and_missing_sections_are_422(client, data_dir):
    r = client.post("/api/parsers/data/table", files=upload(data_dir / "sample.data"), data={"section": "Nope"})
    assert r.status_code == 422 and "Invalid section" in r.json()["detail"]
    r = client.post("/api/parsers/data/table", files=upload(data_dir / "sample.data"), data={"section": "Velocities"})
    assert r.status_code == 422 and "not found" in r.json()["detail"]


def test_data_table_download_roundtrip(client, data_dir):
    view = client.post("/api/parsers/data/table", files=upload(data_dir / "sample.data"), data={"section": "Bonds"}).json()
    r = client.post("/api/parsers/data/table/download?fmt=pgfplots", files=upload(data_dir / "sample.data"), data={"section": "Bonds"})
    assert r.headers["content-disposition"] == 'attachment; filename="sample_Bonds.dat"'
    assert np.allclose(np.loadtxt(io.BytesIO(r.content), skiprows=1)[:20], np.array(view["rows"], dtype=float))


def test_acf_table(client, data_dir):
    r = client.post("/api/parsers/acf/table", files=upload(data_dir / "sample_acf.txt"))
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["lag_time", "acf_x", "acf_y", "acf_z", "timestep"]
    assert r.json()["n_rows_total"] == 10


def test_acf_table_rejects_non_acf_file(client, data_dir):
    r = client.post("/api/parsers/acf/table", files=upload(data_dir / "sample_log.lammps"))
    assert r.status_code == 422 and "timestep blocks" in r.json()["detail"]


@pytest.mark.parametrize("page", ["/parsers/dump", "/parsers/data", "/parsers/acf"])
def test_parser_pages_render(client, page):
    r = client.get(page)
    assert r.status_code == 200 and 'id="calc-form"' in r.text and 'name="file"' in r.text
```

Run them: they fail with 404s / TemplateNotFound.

- [ ] **Step 2: Services**

Add to `parsers_service.py`:

```python
def dump_preview(path: Path) -> pd.DataFrame:
    """One row per frame; never concatenates the trajectory."""
    records = []
    for frame in dump_frames(str(path)):
        bounds = next((v for k, v in frame.metadata.items() if k.startswith("BOX BOUNDS")), None)
        sizes = (bounds[:, 1] - bounds[:, 0]) if bounds is not None else np.full(3, np.nan)
        records.append(
            {
                "timestep": frame.metadata["TIMESTEP"],
                "n_atoms": frame.data.shape[0],
                "box_x": float(sizes[0]),
                "box_y": float(sizes[1]),
                "box_z": float(sizes[2]),
                "columns": " ".join(frame.columns),
            }
        )
    if not records:
        raise ValueError("No frames were found. Upload a LAMMPS dump file written by `dump ... custom`.")
    return pd.DataFrame(records)


def dump_table(path: Path) -> pd.DataFrame:
    return read_dump(str(path))


def data_table(path: Path, section: str) -> pd.DataFrame:
    return read_data_file(str(path)).to_pandas(section)


def acf_table(path: Path) -> pd.DataFrame:
    return read_lammps_acf(path)
```

- [ ] **Step 3: Schema and routes**

`schemas/parsers.py` holds `DataSection` (constructor `section: Annotated[str, Form()] = "Atoms"`). Routes follow the log pair exactly; download stems: dump table `f"{path.stem}_atoms"`, data table `f"{path.stem}_{section}"`, acf `f"{path.stem}_acf"`. `dump_preview` passes `download_url=None` and `meta={"frames": n}`. `to_pandas` emits a `warnings.warn` for generic-column sections; that is library behaviour, leave it.

- [ ] **Step 4: Templates**

`parsers/dump.html`: two calculate buttons in the same form — `f.calculate_button('Preview frames', '/api/parsers/dump/preview')` first, then `f.calculate_button('Load full atom table', '/api/parsers/dump/table')`; intro explains preview is cheap and the full table concatenates every frame. `parsers/data.html`: `f.select_field('section', 'Section', options, value='Atoms', help=...)` where `options` is a literal list in the template with one `(name, name)` pair per entry of `ALLOWED_SECTIONS`, in the same order (`pages.py` passes only `calc`, so the template owns this list). `parsers/acf.html`: file field with help naming `fix ave/correlate`.

- [ ] **Step 5: Run, lint, commit**

`cd webapp && ../.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_parsers.py tests/test_app.py`, then the whole webapp suite, then ruff format/check. Commit:

```bash
git add webapp/src/lada_web/schemas/parsers.py webapp/src/lada_web/services/parsers_service.py \
        webapp/src/lada_web/routers/parsers.py webapp/src/lada_web/templates/parsers webapp/tests/test_parsers.py
git commit -m "feat(webapp): dump, data-file and ACF parser pages"
```

---

### Task 4: Conformations pages — Rg², Ree², end-to-end vectors

**Files:**
- Create: `webapp/src/lada_web/schemas/conformations.py`, `services/conformations_service.py`, `templates/conformations/rg.html`, `templates/conformations/ree.html`, `templates/conformations/ree-vectors.html`, `webapp/tests/test_conformations.py`
- Modify: `webapp/src/lada_web/routers/conformations.py` (replace the stub)

**Interfaces:**
- Consumes (Task 1 Produces; log router as style guide). `lada.parsers.read_dump(path)`; `lada.analysis.calculate_avg_rg_sq(df, coord_cols=[...], molecule_col=..., mass_col=None|str) -> dict[timestep, float]` (a `float` only for single-frame input without a timestep column — `read_dump` always adds one); `calculate_avg_ree_sq(df, coord_cols, molecule_col, atom_id_col) -> dict`; `calculate_ree_vectors(df, coord_cols, molecule_col, atom_id_col) -> DataFrame[timestep, mol, dx, dy, dz]`.
- Produces: `POST /api/conformations/rg`, `/ree`, `/ree-vectors`, each with `/download`. Form fields: `coord_cols: str = "xu,yu,zu"`, `molecule_col: str = "mol"`, `atom_id_col: str = "id"` (ree, ree-vectors), `mass_col: str = ""` (rg; empty means unweighted). Result columns: rg `[timestep, rg_sq]`; ree `[timestep, ree_sq]`; ree-vectors `[timestep, mol, dx, dy, dz]`. Plots: rg and ree plot `x="timestep"`, `y=[value column]`, linear axes.

- [ ] **Step 1: Failing tests** (`test_conformations.py`): with `sample_dump.dump` — rg JSON columns and 3 rows whose values equal `calculate_avg_rg_sq(read_dump(...))` values in timestep order (`np.allclose`); the htmx partial contains `id="result-plot"` and both download buttons; `mass_col` absent from the file (`data={"mass_col": "mass"}`) → 422 carrying the library message; `coord_cols="xu,yu"` → 422 with the message "Coordinate columns must be three comma-separated names"; ree JSON equals `calculate_avg_ree_sq`; ree-vectors columns `["timestep","mol","dx","dy","dz"]` and `n_rows_total == 3 * number of molecules`; csv and pgfplots downloads for rg round-trip against the view; all three pages render with `id="calc-form"`.

- [ ] **Step 2: Schema**

```python
from typing import Annotated

from fastapi import Form


def parse_coord_cols(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if len(names) != 3:
        raise ValueError(
            "Coordinate columns must be three comma-separated names, for example xu,yu,zu."
        )
    return names


class ChainColumns:
    """Column names shared by the conformation routes (flat multipart form fields)."""

    def __init__(
        self,
        coord_cols: Annotated[str, Form()] = "xu,yu,zu",
        molecule_col: Annotated[str, Form()] = "mol",
        atom_id_col: Annotated[str, Form()] = "id",
        mass_col: Annotated[str, Form()] = "",
    ) -> None:
        self.coord_cols = parse_coord_cols(coord_cols)
        self.molecule_col = molecule_col.strip()
        self.atom_id_col = atom_id_col.strip()
        self.mass_col = mass_col.strip() or None
```

- [ ] **Step 3: Service** — `rg_sq(path, *, coord_cols, molecule_col, mass_col) -> pd.DataFrame`, `ree_sq(path, *, coord_cols, molecule_col, atom_id_col)`, `ree_vectors(path, *, coord_cols, molecule_col, atom_id_col)`; a private `_per_timestep(values, name)` turns the dict into `pd.DataFrame({"timestep": keys, name: values})` and a bare float into a one-row frame with timestep 0.

- [ ] **Step 4: Routes and templates** — three view/download pairs using `Annotated[ChainColumns, Depends()]`; download stems `f"{path.stem}_rg_sq"`, `_ree_sq`, `_ree_vectors`; plot labels "Timestep" / "⟨Rg²⟩" and "⟨Ree²⟩". Templates: intro explaining what is computed and that unwrapped coordinates (`xu yu zu`) avoid periodic-boundary artefacts; `text_field` for each column name with help; rg page adds the optional mass column field.

- [ ] **Step 5: Run, lint, commit** — `git add webapp/src/lada_web/schemas/conformations.py webapp/src/lada_web/services/conformations_service.py webapp/src/lada_web/routers/conformations.py webapp/src/lada_web/templates/conformations webapp/tests/test_conformations.py` and commit `feat(webapp): conformation pages (Rg², Ree², end-to-end vectors)`.

---

### Task 5: Rheology pages — stress relaxation modulus and dynamic moduli

**Files:**
- Create: `webapp/src/lada_web/schemas/rheology.py`, `services/rheology_service.py`, `templates/rheology/stress-relaxation.html`, `templates/rheology/dynamic-moduli.html`, `webapp/tests/test_rheology.py`
- Modify: `webapp/src/lada_web/routers/rheology.py` (replace the stub)

**Interfaces:**
- Consumes: `lada.parsers.read_lammps_acf(path)`; `lada.analysis.calc_stress_relaxation(df, volume, temperature) -> DataFrame[lag_time, G_GK, G_FSR]` (raises `KeyError` naming the missing `ACF_S*/ACF_N*` columns, `ValueError` for non-positive volume/temperature); `calc_dynamic_moduli_prony(df, method="GK"|"FSR"|"both", t_min=1.0, t_cutoff=None, n_modes=50, n_omega=200, omega_min=None, omega_max=None) -> DataFrame` with columns `[omega, G_prime, G_dprime]` for a single method or `[omega, G_prime_GK, G_dprime_GK, G_prime_FSR, G_dprime_FSR]` for `"both"`. Fixture `tests/data/sample_stress_acf.txt` (Task 2): 100 lags at spacing 0.5.
- Produces: `POST /api/rheology/stress-relaxation` (+`/download`), fields `volume: float > 0`, `temperature: float > 0`; `POST /api/rheology/dynamic-moduli` (+`/download`), the same two plus `method`, `t_min` (>0, default 1.0), `t_cutoff` (optional), `n_modes` (≥1, default 50), `n_omega` (2–5000, default 200), `omega_min`, `omega_max` (optional, >0). Plots: G(t) `x="lag_time"`, `y=["G_GK","G_FSR"]`, linear axes by default; moduli `x="omega"`, `y=` every column except `omega`, `x_log=True`, `y_log=True`.

- [ ] **Step 1: Failing tests** (`test_rheology.py`): G(t) JSON for the fixture with `volume=1000, temperature=1` has columns `["lag_time","G_GK","G_FSR"]`, 100 rows, and equals `calc_stress_relaxation(read_lammps_acf(fixture), 1000, 1)`; `volume=0` → 422 (FastAPI validation, JSON `detail` is a list); uploading `sample_acf.txt` (acf_x schema) → 422 whose detail mentions `ACF_Sxy`; moduli JSON with defaults has columns `["omega","G_prime","G_dprime"]`, 200 rows, all `omega` increasing, plot `x_log` and `y_log` true; `method=both` yields the five suffixed columns; `n_modes=500` → 422 with the library's "Reduce n_modes" message; `t_cutoff` left empty (`""`) is accepted; pgfplots download of the moduli round-trips against the view; both pages render.

- [ ] **Step 2: Schemas** — `ThermoState(volume: Annotated[float, Form(gt=0)], temperature: Annotated[float, Form(gt=0)])` and `PronyParams(method: Annotated[Literal["GK","FSR","both"], Form()] = "GK", t_min: Annotated[float, Form(gt=0)] = 1.0, t_cutoff: Annotated[float | None, Form()] = None, n_modes: Annotated[int, Form(ge=1)] = 50, n_omega: Annotated[int, Form(ge=2, le=5000)] = 200, omega_min: Annotated[float | None, Form(gt=0)] = None, omega_max: Annotated[float | None, Form(gt=0)] = None)`.

- [ ] **Step 3: Service** — `stress_relaxation(path, volume, temperature) -> DataFrame` and `dynamic_moduli(path, volume, temperature, *, method, t_min, t_cutoff, n_modes, n_omega, omega_min, omega_max) -> DataFrame` (computes G(t) first, then the Prony transform; never passes `return_fit=True`).

- [ ] **Step 4: Routes and templates** — download stems `f"{path.stem}_G_t"` and `f"{path.stem}_moduli"`; plot labels "Lag time" / "G(t)" and "ω" / "G′, G″". Templates: explain LJ units (k_B = 1), that G_GK averages 3 shear components and G_FSR uses all 5 traceless components; the moduli page groups the Prony settings under a "Fit settings" heading with help text for each (log-spaced modes between t_min and the last lag time; leave t_cutoff empty for the full range). Use `f.select_field('method', 'Component', [('GK','Green-Kubo (shear only)'), ('FSR','Full stress relaxation'), ('both','Both')])`.

- [ ] **Step 5: Run, lint, commit** — `git add webapp/src/lada_web/schemas/rheology.py webapp/src/lada_web/services/rheology_service.py webapp/src/lada_web/routers/rheology.py webapp/src/lada_web/templates/rheology webapp/tests/test_rheology.py`; commit `feat(webapp): rheology pages (G(t) and dynamic moduli)`.

---

### Task 6: Dynamics pages — segment ACF, Rouse mode ACF, ISF from `.npz`

**Files:**
- Create: `webapp/src/lada_web/schemas/dynamics.py`, `services/dynamics_service.py`, `templates/dynamics/_topology.html`, `templates/dynamics/segment-acf.html`, `templates/dynamics/rouse-mode.html`, `templates/dynamics/isf.html`, `webapp/tests/test_dynamics.py`
- Modify: `webapp/src/lada_web/routers/dynamics.py` (replace the stub)

**Interfaces:**
- Consumes: `errors.TrajectoryTooLarge(n_frames, limit)` (Task 1); `settings.max_trajectory_bytes`, `settings.max_trajectory_frames`; `lada.analysis.calculate_segment_acf_from_trajectory(path, segment_pairs: ndarray (n,2), time_per_frame) -> ndarray (n_frames, 2)`; `calculate_rouse_mode_acf_from_trajectory(path, chain_indices: ndarray (n, beads), p, time_per_frame)` (raises `ValueError` when `p < 0 or p >= beads`); `calculate_isf_from_trajectory(path, time_per_frame, q_magnitude, n_vectors=50)`. Fixture `tests/data/sample_trajectory.npz`: 32 frames, 32 atoms = 4 chains × 8 beads, chain-major.
- Produces: `POST /api/dynamics/segment-acf`, `/rouse-mode`, `/isf` (+`/download` each). Common field `time_per_frame: float > 0`. Topology fields (segment-acf, rouse-mode): `mode: "uniform"|"custom"` (default uniform), `n_chains: int ≥ 1`, `beads_per_chain: int ≥ 2`, `first_index: int ≥ 0 = 0`, `custom: str = ""`. Rouse adds `p: int ≥ 0`. ISF fields: `q_magnitude: float > 0`, `n_vectors: int 1–1000 = 50`. Result columns: `[lag_time, acf]` (segment, rouse) and `[lag_time, isf]`. Plots `x="lag_time"`, linear axes. Pure functions: `resolve_chain_indices(...)`, `resolve_segment_pairs(...)`, `parse_custom_lines(text)`, `inspect_trajectory(path, max_frames) -> (n_frames, n_atoms)`.

- [ ] **Step 1: Failing tests** (`test_dynamics.py`)

Unit tests for the pure functions: `resolve_chain_indices("uniform", n_chains=2, beads_per_chain=3, first_index=10, custom="")` → `[[10,11,12],[13,14,15]]`; `resolve_segment_pairs` for the same → `[[10,12],[13,15]]`; custom text `"0, 7\n8 15\n\n# comment\n"` → pairs `[[0,7],[8,15]]`; custom bead lists `"0 1 2 3\n4 5 6 7"` → chain indices of shape (2,4) and pairs `[[0,3],[4,7]]`; `"0,7\nfoo,bar"` → `ValueError` whose message starts with `"Line 2:"`; unequal lengths `"0 1 2\n3 4"` for chain indices → `ValueError` mentioning "same number of beads"; uniform mode with `n_chains=None` → `ValueError` saying both counts are required; `inspect_trajectory(fixture, 100) == (32, 32)`; a 101-frame `.npz` written to `tmp_path` raises `TrajectoryTooLarge`; a `.npz` without `coords` (`np.savez(p, other=...)`) raises `ValueError` mentioning `'coords'`; a plain text file raises `ValueError` mentioning `.npz`.

Route tests: uniform segment ACF (`n_chains=4, beads_per_chain=8, time_per_frame=2.0`) → 32 rows, `rows[0] == [0.0, 1.0]`, equals the library result for pairs `[[0,7],[8,15],[16,23],[24,31]]`; custom `"0,7\n8,15"` equals the library result for those pairs; index ≥ 32 in custom mode → 422 mentioning "32 atoms"; Rouse `p=1` → 32 rows starting at 1.0; `p=8` → 422 with the library's message; ISF `q_magnitude=1.0, n_vectors=5` → 32 rows starting at 1.0; a 101-frame file (cap 100 in test settings) → 413 with "at most 100"; a 1.2 MB `.npz` → 413 upload limit; the htmx partial has `id="result-plot"`; csv download of the segment ACF round-trips; each page renders and the two topology pages contain `x-data` and `name="mode"`.

- [ ] **Step 2: Service**

```python
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lada.analysis import (
    calculate_isf_from_trajectory,
    calculate_rouse_mode_acf_from_trajectory,
    calculate_segment_acf_from_trajectory,
)
from numpy.lib import format as npformat

from ..errors import TrajectoryTooLarge


def parse_custom_lines(text: str) -> list[list[int]]:
    """One chain per line: integers separated by commas and/or spaces. Blank and '#' lines skip."""
    chains: list[list[int]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = [t for t in line.replace(",", " ").split() if t]
        try:
            values = [int(t) for t in tokens]
        except ValueError:
            raise ValueError(
                f"Line {number}: expected whole numbers separated by commas or spaces, got '{line}'."
            ) from None
        if len(values) < 2:
            raise ValueError(f"Line {number}: a chain needs at least two bead indices.")
        if min(values) < 0:
            raise ValueError(f"Line {number}: bead indices must be zero or positive.")
        chains.append(values)
    if not chains:
        raise ValueError("Custom topology is empty: enter one chain per line.")
    return chains


def _uniform_blocks(n_chains: int | None, beads_per_chain: int | None, first_index: int) -> np.ndarray:
    if n_chains is None or beads_per_chain is None:
        raise ValueError("Uniform topology needs both the number of chains and the beads per chain.")
    start = first_index + np.arange(n_chains)[:, None] * beads_per_chain
    return start + np.arange(beads_per_chain)[None, :]


def resolve_chain_indices(mode, *, n_chains, beads_per_chain, first_index, custom) -> np.ndarray:
    if mode == "uniform":
        return _uniform_blocks(n_chains, beads_per_chain, first_index)
    chains = parse_custom_lines(custom)
    if len({len(c) for c in chains}) != 1:
        raise ValueError("Every chain must list the same number of beads for a Rouse-mode analysis.")
    return np.array(chains, dtype=int)


def resolve_segment_pairs(mode, *, n_chains, beads_per_chain, first_index, custom) -> np.ndarray:
    if mode == "uniform":
        blocks = _uniform_blocks(n_chains, beads_per_chain, first_index)
        return np.column_stack([blocks[:, 0], blocks[:, -1]])
    return np.array([[c[0], c[-1]] for c in parse_custom_lines(custom)], dtype=int)


def inspect_trajectory(path: Path, max_frames: int) -> tuple[int, int]:
    """Read the 'coords' array header without loading the data; enforce the frame cap."""
    try:
        with zipfile.ZipFile(path) as archive:
            if "coords.npy" not in archive.namelist():
                raise ValueError(
                    "The .npz file must contain an array named 'coords' with shape "
                    "(n_frames, n_atoms, 3)."
                )
            with archive.open("coords.npy") as fh:
                version = npformat.read_magic(fh)
                reader = (
                    npformat.read_array_header_1_0
                    if version == (1, 0)
                    else npformat.read_array_header_2_0
                )
                shape, _, _ = reader(fh)
    except zipfile.BadZipFile:
        raise ValueError("This is not a NumPy .npz archive. Save the trajectory with np.savez.") from None
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"'coords' must have shape (n_frames, n_atoms, 3); this file has {shape}.")
    n_frames, n_atoms = int(shape[0]), int(shape[1])
    if n_frames > max_frames:
        raise TrajectoryTooLarge(n_frames, max_frames)
    return n_frames, n_atoms


def check_indices(indices: np.ndarray, n_atoms: int) -> None:
    if indices.size and int(indices.max()) >= n_atoms:
        raise ValueError(
            f"Bead index {int(indices.max())} is out of range: the trajectory has {n_atoms} atoms "
            f"(indices 0 to {n_atoms - 1})."
        )


def _frame(result: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame({"lag_time": result[:, 0], name: result[:, 1]})


def segment_acf(path, pairs, time_per_frame) -> pd.DataFrame:
    return _frame(calculate_segment_acf_from_trajectory(str(path), pairs, time_per_frame), "acf")


def rouse_mode_acf(path, chains, p, time_per_frame) -> pd.DataFrame:
    return _frame(calculate_rouse_mode_acf_from_trajectory(str(path), chains, p, time_per_frame), "acf")


def isf(path, time_per_frame, q_magnitude, n_vectors) -> pd.DataFrame:
    return _frame(calculate_isf_from_trajectory(str(path), time_per_frame, q_magnitude, n_vectors), "isf")
```

- [ ] **Step 3: Schema** — `TopologyForm` (constructor fields `mode: Annotated[Literal["uniform","custom"], Form()] = "uniform"`, `n_chains: Annotated[int | None, Form(ge=1)] = None`, `beads_per_chain: Annotated[int | None, Form(ge=2)] = None`, `first_index: Annotated[int, Form(ge=0)] = 0`, `custom: Annotated[str, Form()] = ""`) with methods `chain_indices() -> np.ndarray` and `segment_pairs() -> np.ndarray` delegating to the service; `FrameTime(time_per_frame: Annotated[float, Form(gt=0)])`; `RouseMode(p: Annotated[int, Form(ge=0)])`; `IsfParams(q_magnitude: Annotated[float, Form(gt=0)], n_vectors: Annotated[int, Form(ge=1, le=1000)] = 50)`.

- [ ] **Step 4: Routes** — each view/download pair: `save_upload(file, ws, settings.max_trajectory_bytes)` → `inspect_trajectory(path, settings.max_trajectory_frames)` → resolve topology → `check_indices` → compute → `table_result_from_df(..., meta={"frames": n_frames, "atoms": n_atoms, "chains": len(...)})`. Download stems `f"{path.stem}_segment_acf"`, `f"{path.stem}_rouse_p{p}"`, `f"{path.stem}_isf"`. Result titles mention the discard-the-tail advice: put in the page intro "Statistical noise grows with lag time; ignore roughly the last 10–20 % of points when fitting relaxation times."

- [ ] **Step 5: Templates** — `dynamics/_topology.html` is included by the two topology pages inside the form and contains the Alpine state: `<fieldset x-data="{ mode: 'uniform', beads: 8 }">` with two radio inputs `name="mode"` (`x-model="mode"`), the uniform fields (`n_chains`, `beads_per_chain` with `x-model.number="beads"`, `first_index`) inside `<div x-show="mode === 'uniform'">`, the custom textarea inside `<div x-show="mode === 'custom'">` with placeholder `0,7\n8,15` and help "one chain per line: `head,tail` or the full bead list", and the mandatory warning paragraph: ".npz files carry no atom ids. Atom order is assumed to match ascending atom id, 0-indexed. Check the first and last indices against your data file." The Rouse page's `p` input sets `:max="beads - 1"` and its help says p must be below the beads per chain. `base.html` already loads Alpine.

- [ ] **Step 6: Run, lint, commit** — `git add webapp/src/lada_web/schemas/dynamics.py webapp/src/lada_web/services/dynamics_service.py webapp/src/lada_web/routers/dynamics.py webapp/src/lada_web/templates/dynamics webapp/tests/test_dynamics.py`; commit `feat(webapp): dynamics pages from .npz trajectories with uniform/custom topology`.

---

### Task 7: Modifiers page — rewrite end beads (download only)

**Files:**
- Create: `webapp/src/lada_web/schemas/modifiers.py`, `services/modifiers_service.py`, `templates/modifiers/rewrite-end-beads.html`, `webapp/tests/test_modifiers.py`
- Modify: `webapp/src/lada_web/routers/modifiers.py` (replace the stub)

**Interfaces:**
- Consumes: `lada.modifiers.rewrite_end_beads(input_file, output_file, new_end_type, base_type=1)` (prints two lines to stdout; assumes column 2 of `Atoms` is the molecule id); `lada.parsers.read_data_file(path).metadata.get("atom style")` and `.get("Masses")`; fixture `tests/data/sample_molecular.data` (Task 2) and the atomic-style `tests/data/sample.data`.
- Produces: `POST /api/modifiers/rewrite-end-beads` returning the rewritten file as an attachment named `f"{path.stem}_endbeads.data"` (media type `text/plain`). Fields `new_end_type: int ≥ 1`, `base_type: int ≥ 1 = 1`. No view route, no `/download` suffix.

- [ ] **Step 1: Failing tests** (`test_modifiers.py`): happy path → 200, `content-disposition == 'attachment; filename="sample_molecular_endbeads.data"'`, and parsing the body with `read_data_file` shows atoms 1,4,5,8,9,12 with type 3 and `metadata["atom types"] == 3`; atomic-style `sample.data` → 422 whose detail mentions "molecular"; `base_type=9` → 422 mentioning "base type 9"; `new_end_type=0` → 422; a browser-style submit (`Accept: text/html`) of the atomic file returns an HTML page containing the message; oversized upload → 413; the page renders with a submit button and without `hx-post`; after the happy path `temp_root` is empty.

- [ ] **Step 2: Service**

```python
MOLECULAR_STYLES = {"bond", "molecular", "full"}  # column 2 is the molecule id in lada's schema


def rewrite_end_beads_file(input_path: Path, output_path: Path, *, new_end_type: int, base_type: int) -> Path:
    data = read_data_file(str(input_path))
    style = data.metadata.get("atom style")
    if style not in MOLECULAR_STYLES:
        raise ValueError(
            "The Atoms section must use a molecular-style atom style (bond, molecular or full) "
            f"so that column 2 is the molecule id; this file declares '{style or 'no style'}'. "
            "Add an 'Atoms # molecular' comment to the section header if the layout is right."
        )
    masses = data.sections.get("Masses")
    known_types = set() if masses is None or masses.size == 0 else set(np.atleast_2d(masses)[:, 0].astype(int))
    if base_type not in known_types:
        raise ValueError(
            f"There is no base type {base_type} in the Masses section (found {sorted(known_types)})."
        )
    rewrite_end_beads(str(input_path), str(output_path), new_end_type, base_type)
    return output_path
```

- [ ] **Step 3: Schema, route, template** — `EndBeadParams(new_end_type: Annotated[int, Form(ge=1)], base_type: Annotated[int, Form(ge=1)] = 1)`. Route body: `with download_workspace(settings) as ws: path = save_upload(...); out = rewrite_end_beads_file(path, ws / f"{path.stem}_endbeads.data", ...); return file_download(out, ws, "text/plain")`. Template: form as usual but the button is a plain `<button type="submit" formaction="/api/modifiers/rewrite-end-beads">Rewrite and download</button>` (native submit, no htmx) and the page says the download starts immediately and explains chain ends are the lowest and highest atom id per molecule.

- [ ] **Step 4: Run, lint, commit** — `git add webapp/src/lada_web/schemas/modifiers.py webapp/src/lada_web/services/modifiers_service.py webapp/src/lada_web/routers/modifiers.py webapp/src/lada_web/templates/modifiers webapp/tests/test_modifiers.py`; commit `feat(webapp): rewrite-end-beads page`.

---

### Task 8: Integration hardening, temp-cleanup regression tests, and UI polish

**Files:**
- Create: `webapp/tests/test_temp_cleanup.py`, `webapp/tests/test_body_limit.py`
- Modify: `webapp/src/lada_web/main.py` (body-size middleware), `webapp/tests/test_app.py` (every page renders), `webapp/src/lada_web/static/css/app.css`, `templates/index.html`, `templates/base.html`, `templates/_macros.html`, any `templates/**` page copy that the review below finds inconsistent, `catalog.py` copy if improved

**Interfaces:**
- Consumes: every route from Tasks 1–7; fixtures from Task 2.
- Produces: `main.create_app` additionally installs an HTTP middleware that answers 413 (via `errors.error_response`) when `Content-Length` exceeds `settings.max_trajectory_bytes + 1 MiB`, before the body is read.

- [ ] **Step 1: Temp-cleanup regression tests** — `test_temp_cleanup.py` runs a battery of requests against one client and asserts after **each** that `temp_root` does not exist or is empty. The battery must include: log view ok; log missing file (422); dump table csv download; rg view ok and rg with a bad column (422); oversized upload (413); segment-acf view ok and the 101-frame guardrail (413); stress-relaxation with the wrong schema file (422); dynamic-moduli pgfplots download; rewrite-end-beads ok and with the atomic file (422); an htmx request; and a request to a route that raises an unexpected error (monkeypatch `parsers_service.log_table` to raise `RuntimeError`; expect 500 and still no leftovers).

- [ ] **Step 2: Body-size middleware and its test** — in `create_app`, `@app.middleware("http")` reading `request.headers.get("content-length")`; if present and larger than the limit return `error_response(request, 413, "The request body is larger than this server accepts.", title="Too large")`. `test_body_limit.py`: a 2.1 MB multipart post to any route under the 1 MB test settings is 413 and `temp_root` stays empty; a normal request still works.

- [ ] **Step 3: Every page renders** — extend `test_app.py` with a parametrized test over `CALCULATIONS` asserting 200, `id="calc-form"`, a `<label for="file">`, exactly one `<form`, and either `hx-post=` or `formaction=` present.

- [ ] **Step 4: Polish pass** — invoke the `frontend-design:frontend-design` skill and review every page as a user: consistent intros, help texts and button labels; the landing page explains the flow (upload → calculate → download) and lists needed inputs; error partial and error page share one look; tables readable on narrow screens; keyboard focus visible; reduced-motion honoured. Fix copy in templates and `catalog.py`, and CSS in `app.css`. Do not change route paths, field names, macro signatures or IDs.

- [ ] **Step 5: Run the whole webapp suite, ruff, and the core suite; commit** — `git add webapp/src webapp/tests` (these paths are yours in this wave) and commit `feat(webapp): temp-cleanup tests, body-size guard and UI polish`.

---

### Task 9: Deploy readiness — Docker, CI, documentation

**Files:**
- Create: `webapp/Dockerfile`, `webapp/.dockerignore`, `webapp/docker-compose.yml`, `.github/workflows/webapp-tests.yml`
- Modify: `webapp/README.md` (replace the stub), `README.md` (root: add a short "Web front-end" section after "Package structure"), `WEBAPP_PLAN.md` (tick the four Status boxes)

**Interfaces:**
- Consumes: the `Settings` env names `LADA_WEB_MAX_UPLOAD_MB`, `LADA_WEB_MAX_TRAJECTORY_MB`, `LADA_WEB_MAX_TRAJECTORY_FRAMES`, `LADA_WEB_MAX_TABLE_ROWS`, `LADA_WEB_TEMP_ROOT`; the route list from Tasks 1–7; `/api/docs`.

- [ ] **Step 1: Dockerfile** (build context = `webapp/`, per spec; `lada` comes from PyPI — note in the README that the image builds only once `lada` ≥ 1.2.0 is published):

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 lada
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .
USER lada
EXPOSE 8000
CMD ["uvicorn", "lada_web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

`.dockerignore`: `tests/`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`. `docker-compose.yml`: one service `web` building `.`, `ports: ["8000:8000"]`, environment block with the five `LADA_WEB_*` variables at their defaults.

- [ ] **Step 2: CI** — `.github/workflows/webapp-tests.yml` triggered on push/PR to `main`/`develop` for paths `webapp/**`, `src/**`, `tests/data/**`, `.github/workflows/webapp-tests.yml`; one job on `ubuntu-latest`, Python 3.12: `pip install -e . -e "./webapp[dev]"`, `ruff check webapp/src webapp/tests`, `ruff format --check webapp/src webapp/tests`, `cd webapp && python -m pytest -q`.

- [ ] **Step 3: `webapp/README.md`** — sections: what it is (calculation front-end, not a 3D viewer); quick start (install from repo root, `uvicorn lada_web.main:app --reload`, open `http://127.0.0.1:8000`); the pages grouped by section with the file each needs; the JSON API (`/api/docs`, one `curl -F file=@... -F volume=1000 -F temperature=1 http://127.0.0.1:8000/api/rheology/stress-relaxation` example, downloads via `?fmt=`); configuration table of the env variables; limits and timeouts (dynamics requests are CPU-bound, `O(n_frames²)`; raise reverse-proxy read timeouts, e.g. nginx `proxy_read_timeout 600s`); Docker (`docker compose up`, PyPI pin caveat); testing; design notes (stateless recompute-on-download, no job queue, `/api/` reserved for a public API).

- [ ] **Step 4: Root README section and plan status** — 5–8 lines pointing to `webapp/README.md`; tick `- [x]` for all four phases in `WEBAPP_PLAN.md`.

- [ ] **Step 5: Check the workflow and commit** — there is no YAML linter in the venv, so keep the workflow minimal and read it back line by line (indentation, `run: |` blocks). Commit: `git add webapp/Dockerfile webapp/.dockerignore webapp/docker-compose.yml .github/workflows/webapp-tests.yml webapp/README.md README.md WEBAPP_PLAN.md` with message `docs(webapp): Dockerfile, compose, CI workflow and README`.
