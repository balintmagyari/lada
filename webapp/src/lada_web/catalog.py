"""The declarative registry of every tool the app offers.

Three levels: a `Workspace`, its `Section`s, and the `Calculation`s (tools) in each section. The
catalog drives the navigation, the overview page and the page router, so a new tool becomes
reachable by adding one `Calculation` entry. Version 1 has one workspace, the rheometer; the
simulation tools live on the `webapp-simulations` branch and slot back in as a second
`Workspace` with its own sections.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Workspace:
    key: str
    title: str
    blurb: str
    home: str
    # The three steps of working in this workspace, each as (verb, the rest of the sentence).
    steps: tuple[tuple[str, str], ...] = field(default=())
    note: str = ""


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    blurb: str
    workspace: str = "rheometer"


@dataclass(frozen=True)
class Calculation:
    section: str
    slug: str
    title: str
    summary: str
    input_kind: str  # a key of INPUT_KINDS
    # Rheometer pages share one template and live under /rheometer/<slug>. A tool without a
    # prefix is served at /<section>/<slug> from its own template.
    template_name: str = ""
    path_prefix: str = ""

    @property
    def path(self) -> str:
        if self.path_prefix:
            return f"{self.path_prefix}/{self.slug}"
        return f"/{self.section}/{self.slug}"

    @property
    def template(self) -> str:
        return self.template_name or f"{self.section}/{self.slug}.html"


WORKSPACES: list[Workspace] = [
    Workspace(
        "rheometer",
        "Rheometer",
        "Turn the files your rheometer exports into clear, publication-ready graphs.",
        "/rheometer",
        steps=(
            ("Import", "the exports from your rheometer or a spreadsheet."),
            ("Graph", "them with a ready-made rheology plot, or any column against any other."),
            ("Export", "a figure for your paper, or a pgfplots file for LaTeX."),
        ),
        note="Your data stays in this browser. The server reads each file and forgets it as "
        "soon as it has answered.",
    ),
]


SECTIONS: list[Section] = [
    Section(
        "mydata",
        "My data",
        "Bring in rheometer exports and check how every column was read.",
        workspace="rheometer",
    ),
    Section(
        "oscillation",
        "Oscillation",
        "Moduli from frequency, amplitude and temperature sweeps.",
        workspace="rheometer",
    ),
    Section(
        "flow",
        "Flow & transient",
        "Viscosity curves, creep and stress relaxation.",
        workspace="rheometer",
    ),
    Section(
        "superposition",
        "Superposition",
        "Master curves by time-temperature superposition.",
        workspace="rheometer",
    ),
    Section(
        "custom",
        "Custom plot",
        "Any column against any other, from any of your files.",
        workspace="rheometer",
    ),
]


def _rheometer(
    section: str,
    slug: str,
    title: str,
    summary: str,
    kind: str,
    template: str = "rheometry/studio.html",
) -> Calculation:
    return Calculation(
        section, slug, title, summary, kind, template_name=template, path_prefix="/rheometer"
    )


CALCULATIONS: list[Calculation] = [
    _rheometer(
        "mydata",
        "my-data",
        "Import files",
        "Upload rheometer exports and spreadsheets, and see what each column was read as.",
        "rheo-file",
        template="rheometry/mydata.html",
    ),
    _rheometer(
        "oscillation",
        "frequency-sweep",
        "Frequency sweep",
        "G′ and G″, with tan δ or |η*| if wanted, against angular frequency.",
        "rheo-data",
    ),
    _rheometer(
        "oscillation",
        "amplitude-sweep",
        "Amplitude sweep",
        "G′ and G″ against strain or stress amplitude.",
        "rheo-data",
    ),
    _rheometer(
        "oscillation",
        "temperature-sweep",
        "Temperature sweep",
        "G′, G″ and tan δ against temperature.",
        "rheo-data",
    ),
    _rheometer(
        "oscillation",
        "time-sweep",
        "Time sweep",
        "G′, G″ and tan δ against time, at one frequency and temperature.",
        "rheo-data",
    ),
    _rheometer(
        "flow",
        "flow-curve",
        "Flow curve",
        "Viscosity, and shear stress if wanted, against shear rate.",
        "rheo-data",
    ),
    _rheometer(
        "flow",
        "creep",
        "Creep",
        "Creep compliance J(t) against time.",
        "rheo-data",
    ),
    _rheometer(
        "flow",
        "stress-relaxation",
        "Stress relaxation",
        "Relaxation modulus G(t) against time.",
        "rheo-data",
    ),
    _rheometer(
        "superposition",
        "master-curve",
        "Master curve",
        "Shift sweeps at several temperatures into one curve, with aT and a WLF fit.",
        "rheo-data",
    ),
    _rheometer(
        "custom",
        "custom-plot",
        "Custom plot",
        "Choose the axes and the series yourself, from any column of any file.",
        "rheo-data",
    ),
]


# What each input kind is called: `short` labels the chip on a tool's page and in the
# overview, `needs` is the full sentence a screen reader hears beside it.
INPUT_KINDS: dict[str, dict[str, str]] = {
    "rheo-file": {
        "short": "Rheometer export",
        "needs": "Takes rheometer exports and spreadsheets: CSV, TXT, DAT, Excel, TRIOS, "
        "RheoCompass or TA Orchestrator",
    },
    "rheo-data": {"short": "Imported data", "needs": "Uses the files you imported on My data"},
}


SECTION_BY_KEY: dict[str, Section] = {section.key: section for section in SECTIONS}
WORKSPACE_BY_KEY: dict[str, Workspace] = {workspace.key: workspace for workspace in WORKSPACES}


def get_calculation_by_path(path: str) -> Calculation | None:
    """The catalog entry served at `path` (e.g. "/rheometer/creep")."""
    wanted = "/" + path.strip("/")
    for calc in CALCULATIONS:
        if calc.path == wanted:
            return calc
    return None


def workspace_of(calc: Calculation) -> Workspace:
    return WORKSPACE_BY_KEY[SECTION_BY_KEY[calc.section].workspace]


def calculations_by_section(
    workspace: str | None = None,
) -> list[tuple[Section, list[Calculation]]]:
    """Sections in catalog order with their tools, optionally only one workspace's."""
    return [
        (s, [c for c in CALCULATIONS if c.section == s.key])
        for s in SECTIONS
        if workspace is None or s.workspace == workspace
    ]


def navigation() -> list[tuple[Workspace, list[tuple[Section, list[Calculation]]]]]:
    """Every workspace with its sections and tools: what the sidebar draws."""
    return [(w, calculations_by_section(w.key)) for w in WORKSPACES]
