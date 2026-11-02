"""The declarative registry of every tool the app offers.

Three levels: a `Workspace` (simulation analysis, or rheometer data), its `Section`s, and the
`Calculation`s in each section. The catalog drives the navigation, the overview pages and the
page router, so a new tool becomes reachable by adding one `Calculation` entry and, for a
simulation tool, the matching template.
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
    workspace: str = "simulations"


@dataclass(frozen=True)
class Calculation:
    section: str
    slug: str
    title: str
    summary: str
    input_kind: str  # a key of INPUT_KINDS
    # Rheometer pages share one template and live under /rheometer/<slug>; simulation pages
    # keep their own template and the /<section>/<slug> address they have always had.
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
        "simulations",
        "Simulations",
        "Read and analyse LAMMPS output: conformations, dynamics and rheology of polymer chains.",
        "/simulations",
        steps=(
            ("Upload", "a simulation file and set the parameters."),
            ("Calculate", "and read the table and plot."),
            ("Download", "as CSV or as a pgfplots table for LaTeX."),
        ),
        note="Uploads are processed in a temporary folder and deleted as soon as the response is "
        "sent. Nothing is stored on the server.",
    ),
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
        "parsers",
        "Read files",
        "Turn LAMMPS output files into tables you can inspect and download.",
    ),
    Section(
        "conformations",
        "Chain conformations",
        "Size and shape of polymer chains from a dump trajectory.",
    ),
    Section(
        "dynamics",
        "Dynamics",
        "Time correlation functions from a NumPy .npz trajectory.",
    ),
    Section(
        "rheology",
        "Rheology",
        "Relaxation modulus and dynamic moduli from a stress autocorrelation file.",
    ),
    Section(
        "modifiers",
        "Topology tools",
        "Rewrite a LAMMPS data file and download the result.",
    ),
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

CALCULATIONS: list[Calculation] = [
    Calculation(
        "parsers",
        "dump",
        "Dump trajectory",
        "Preview the frames of a dump file or load the full atom table.",
        "dump",
    ),
    Calculation(
        "parsers",
        "log",
        "Thermo log",
        "Extract the thermodynamic output table from a LAMMPS log file.",
        "log",
    ),
    Calculation(
        "parsers",
        "data",
        "Data file",
        "Read one section of a LAMMPS data file, for example Atoms, Bonds or Masses.",
        "data",
    ),
    Calculation(
        "parsers",
        "acf",
        "Stress ACF file",
        "Read the last timestep block of a fix ave/correlate output file as a table.",
        "acf",
    ),
    Calculation(
        "conformations",
        "rg",
        "Radius of gyration",
        "Ensemble-average Rg² of the chains at every timestep.",
        "dump",
    ),
    Calculation(
        "conformations",
        "ree",
        "End-to-end distance",
        "Ensemble-average Ree² of the chains at every timestep.",
        "dump",
    ),
    Calculation(
        "conformations",
        "ree-vectors",
        "End-to-end vectors",
        "The end-to-end vector of every chain at every timestep.",
        "dump",
    ),
    Calculation(
        "dynamics",
        "segment-acf",
        "End-to-end vector autocorrelation",
        "Normalized autocorrelation of the chain end-to-end vectors.",
        "npz",
    ),
    Calculation(
        "dynamics",
        "rouse-mode",
        "Rouse mode autocorrelation",
        "Normalized autocorrelation of the amplitude of one Rouse mode.",
        "npz",
    ),
    Calculation(
        "dynamics",
        "isf",
        "Intermediate scattering function",
        "Coherent F(q,t) / F(q,0) at one scattering vector magnitude.",
        "npz",
    ),
    Calculation(
        "rheology",
        "stress-relaxation",
        "Stress relaxation modulus",
        "G(t) by the Green-Kubo and full stress relaxation methods.",
        "acf",
    ),
    Calculation(
        "rheology",
        "dynamic-moduli",
        "Dynamic moduli",
        "G'(ω) and G''(ω) from a Prony-series fit of G(t).",
        "acf",
    ),
    Calculation(
        "modifiers",
        "rewrite-end-beads",
        "Rewrite end beads",
        "Give every chain end a new atom type and download the modified data file.",
        "data",
    ),
]

SIMULATION_CALCULATIONS: list[Calculation] = list(CALCULATIONS)


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


RHEOMETER_CALCULATIONS: list[Calculation] = [
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

CALCULATIONS.extend(RHEOMETER_CALCULATIONS)


# What each input kind is called: `short` labels the chip on a tool's page and in the
# overview, `needs` is the full sentence a screen reader hears beside it.
INPUT_KINDS: dict[str, dict[str, str]] = {
    "dump": {"short": "Dump file", "needs": "Needs a LAMMPS dump file"},
    "log": {"short": "Log file", "needs": "Needs a LAMMPS log file"},
    "data": {"short": "Data file", "needs": "Needs a LAMMPS data file"},
    "acf": {"short": "Stress ACF", "needs": "Needs a stress autocorrelation file"},
    "npz": {"short": ".npz trajectory", "needs": "Needs a NumPy .npz trajectory"},
    "rheo-file": {
        "short": "Rheometer export",
        "needs": "Takes rheometer exports and spreadsheets: CSV, TXT, DAT, Excel, TRIOS or "
        "RheoCompass",
    },
    "rheo-data": {"short": "Imported data", "needs": "Uses the files you imported on My data"},
}


SECTION_BY_KEY: dict[str, Section] = {section.key: section for section in SECTIONS}
WORKSPACE_BY_KEY: dict[str, Workspace] = {workspace.key: workspace for workspace in WORKSPACES}


def get_calculation_by_path(path: str) -> Calculation | None:
    """The catalog entry served at `path` (e.g. "/parsers/log" or "/rheometer/creep")."""
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
