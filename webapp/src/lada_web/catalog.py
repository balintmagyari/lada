"""The declarative registry of every calculation the app offers.

The catalog drives the navigation, the index page and the page router, so a new
calculation becomes reachable by adding one `Calculation` entry and the matching
template.
"""

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


def get_calculation(section: str, slug: str) -> Calculation | None:
    for calc in CALCULATIONS:
        if calc.section == section and calc.slug == slug:
            return calc
    return None


def calculations_by_section() -> list[tuple[Section, list[Calculation]]]:
    return [(s, [c for c in CALCULATIONS if c.section == s.key]) for s in SECTIONS]
