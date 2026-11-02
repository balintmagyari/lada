"""The rheology quantities the importer recognises, and the units each may be given in.

One registry serves the whole workspace: the readers use it to recognise a column from its
header ("Storage modulus", "G'", "Speichermodul" ...) and to convert its values into the
quantity's canonical unit, and the browser fetches it to offer the same units for display.
Keeping it in one place means the Python and the JavaScript sides cannot disagree.

Canonical units are the ones rheologists read: Pa, Pa·s, rad/s, 1/s, s, °C, %. Every recognised
column is stored in its canonical unit; the unit it arrived in is kept as `source_unit`.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    symbol: str
    factor: float = 1.0  # canonical = value * factor + offset
    offset: float = 0.0
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quantity:
    key: str
    name: str
    symbol: str  # plain-text symbol for the interface
    units: tuple[Unit, ...]  # the first is canonical
    aliases: tuple[str, ...] = ()
    # Plotted on a log axis by default when it varies over decades.
    positive: bool = True

    @property
    def canonical(self) -> Unit:
        return self.units[0]


_PA = (
    Unit("Pa", 1.0, aliases=("n/m2", "n/m^2", "n/m²")),
    Unit("kPa", 1e3),
    Unit("MPa", 1e6),
    Unit("GPa", 1e9),
    Unit("mPa", 1e-3),
    Unit("dyn/cm²", 0.1, aliases=("dyn/cm2", "dyn/cm^2", "dyne/cm2", "dyne/cm²")),
)
_PAS = (
    Unit("Pa·s", 1.0, aliases=("pa.s", "pa s", "pas", "pa*s", "pa-s")),
    Unit("mPa·s", 1e-3, aliases=("mpa.s", "mpa s", "mpas", "mpa*s", "cp", "cps")),
    Unit("kPa·s", 1e3, aliases=("kpa.s", "kpa s", "kpas", "kpa*s")),
    Unit("P", 0.1, aliases=("poise",)),
)
_INV_PA = (
    Unit("1/Pa", 1.0, aliases=("pa^-1", "pa-1", "pa⁻¹")),
    Unit("1/kPa", 1e-3, aliases=("kpa^-1", "kpa-1", "kpa⁻¹")),
    Unit("1/MPa", 1e-6, aliases=("mpa^-1", "mpa-1", "mpa⁻¹")),
    Unit("1/GPa", 1e-9, aliases=("gpa^-1", "gpa-1", "gpa⁻¹")),
)
_TIME = (
    Unit("s", 1.0, aliases=("sec", "second", "seconds")),
    Unit("ms", 1e-3),
    Unit("min", 60.0, aliases=("minute", "minutes")),
    Unit("h", 3600.0, aliases=("hr", "hour", "hours")),
)
_DIMENSIONLESS = (Unit("", 1.0, aliases=("1", "-", "–", "—", "()", "none")),)

QUANTITIES: tuple[Quantity, ...] = (
    Quantity(
        "angular_frequency",
        "Angular frequency",
        "ω",
        (Unit("rad/s", 1.0, aliases=("rad s-1", "rad/sec", "rad·s⁻¹", "rad.s-1")),),
        aliases=(
            "angular frequency",
            "ang. frequency",
            "ang frequency",
            "omega",
            "ω",
            "kreisfrequenz",
            "angular freq",
            "angular freq.",
        ),
    ),
    Quantity(
        "frequency",
        "Frequency",
        "f",
        (Unit("Hz", 1.0), Unit("kHz", 1e3), Unit("mHz", 1e-3)),
        aliases=("frequency", "freq", "freq.", "oscillation frequency", "frequenz", "f"),
    ),
    Quantity(
        "storage_modulus",
        "Storage modulus",
        "G′",
        _PA,
        aliases=(
            "storage modulus",
            "g'",
            "g′",
            "g prime",
            "speichermodul",
            "elastic modulus",
            "storage mod.",
            "gp",
            "g1",
        ),
    ),
    Quantity(
        "loss_modulus",
        "Loss modulus",
        "G″",
        _PA,
        aliases=(
            "loss modulus",
            "g''",
            "g″",
            'g"',
            "g double prime",
            "verlustmodul",
            "viscous modulus",
            "loss mod.",
            "gpp",
            "g2",
        ),
    ),
    Quantity(
        "tan_delta",
        "Loss factor",
        "tan δ",
        _DIMENSIONLESS,
        aliases=(
            "tan(delta)",
            "tan delta",
            "tan δ",
            "tan(δ)",
            "tand",
            "tan_delta",
            "loss factor",
            "damping factor",
            "verlustfaktor",
            "loss tangent",
        ),
    ),
    Quantity(
        "phase_angle",
        "Phase angle",
        "δ",
        (Unit("°", 1.0, aliases=("deg", "degree", "degrees")), Unit("rad", 57.29577951308232)),
        aliases=("phase angle", "delta", "δ", "phasenwinkel", "phase shift angle", "phase shift"),
    ),
    Quantity(
        "complex_modulus",
        "Complex modulus",
        "|G*|",
        _PA,
        aliases=(
            "complex modulus",
            "|g*|",
            "g*",
            "complex shear modulus",
            "komplexer schubmodul",
            "komplexer modul",
        ),
    ),
    Quantity(
        "complex_viscosity",
        "Complex viscosity",
        "|η*|",
        _PAS,
        aliases=(
            "complex viscosity",
            "|eta*|",
            "eta*",
            "|η*|",
            "η*",
            "complex visc.",
            "komplexe viskosität",
            "komplexe viskositat",
        ),
    ),
    Quantity(
        "strain",
        "Strain",
        "γ",
        (Unit("%", 1.0, aliases=("percent", "pct")), Unit("", 100.0, aliases=("1", "-", "–"))),
        aliases=(
            "strain",
            "oscillation strain",
            "shear strain",
            "strain amplitude",
            "deformation",
            "gamma",
            "γ",
            "γ0",
            "γ₀",
            "osc. strain",
            "scherdeformation",
        ),
    ),
    Quantity(
        "stress",
        "Shear stress",
        "τ",
        _PA,
        aliases=(
            "shear stress",
            "stress",
            "oscillation stress",
            "osc. stress",
            "stress amplitude",
            "schubspannung",
            "tau",
            "τ",
            "τ0",
            "τ₀",
            "sigma",
            "σ",
        ),
    ),
    Quantity(
        "shear_rate",
        "Shear rate",
        "γ̇",
        (Unit("1/s", 1.0, aliases=("s^-1", "s-1", "s⁻¹", "1/sec")),),
        aliases=(
            "shear rate",
            "rate",
            "schergeschwindigkeit",
            "scherrate",
            "gamma dot",
            "γ̇",
            "gammadot",
        ),
    ),
    Quantity(
        "viscosity",
        "Viscosity",
        "η",
        _PAS,
        aliases=(
            "viscosity",
            "apparent viscosity",
            "shear viscosity",
            "dynamic viscosity",
            "viskosität",
            "viskositat",
            "eta",
            "η",
        ),
    ),
    Quantity(
        "compliance",
        "Creep compliance",
        "J",
        _INV_PA,
        aliases=(
            "compliance",
            "creep compliance",
            "j(t)",
            "j",
            "nachgiebigkeit",
            "shear compliance",
        ),
    ),
    Quantity(
        "relaxation_modulus",
        "Relaxation modulus",
        "G(t)",
        _PA,
        aliases=("relaxation modulus", "g(t)", "stress relaxation modulus", "relaxationsmodul"),
    ),
    Quantity(
        "temperature",
        "Temperature",
        "T",
        (
            Unit("°C", 1.0, aliases=("degc", "deg c", "celsius", "oc")),
            Unit("K", 1.0, -273.15, aliases=("kelvin",)),
            Unit("°F", 5 / 9, -160 / 9, aliases=("degf",)),
        ),
        # Not "T": aliases are matched case-folded, and a bare "t" is far more often time.
        aliases=("temperature", "temp", "temp.", "temperatur", "sample temperature"),
        positive=False,
    ),
    Quantity(
        "time",
        "Time",
        "t",
        _TIME,
        aliases=(
            "time",
            "step time",
            "total time",
            "zeit",
            "measuring time",
            "interval time",
            "point time",
            "elapsed time",
        ),
    ),
    Quantity(
        "normal_force",
        "Normal force",
        "F_N",
        (Unit("N", 1.0), Unit("mN", 1e-3), Unit("kN", 1e3)),
        aliases=("normal force", "normalkraft", "axial force", "fn"),
        positive=False,
    ),
    Quantity(
        "torque",
        "Torque",
        "M",
        (
            Unit("N·m", 1.0, aliases=("nm", "n.m", "n m", "n*m")),
            Unit("mN·m", 1e-3, aliases=("mnm", "mn.m", "mn m", "mn*m")),
            Unit("µN·m", 1e-6, aliases=("µnm", "µn.m", "µn m", "un.m", "unm")),
        ),
        aliases=("torque", "drehmoment", "oscillation torque"),
        positive=False,
    ),
)

QUANTITY_BY_KEY: dict[str, Quantity] = {q.key: q for q in QUANTITIES}

# A column that only numbers the points: recognised so it is not offered as a data axis first.
INDEX_ALIASES = frozenset(
    {
        "point no.",
        "point no",
        "point",
        "meas. pts.",
        "meas. pts",
        "meas pts",
        "messpunkt",
        "no.",
        "no",
        "index",
        "#",
        "n",
        "point number",
        "data point",
    }
)


def fold(text: str) -> str:
    """Lower-case, NFKC-normalised, inner whitespace collapsed: the form aliases are matched in."""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", text)


_ALIAS_INDEX: dict[str, str] = {}
for _q in QUANTITIES:
    for _alias in (_q.key.replace("_", " "), _q.name, *_q.aliases):
        _ALIAS_INDEX.setdefault(fold(_alias), _q.key)


# "Storage modulus (Pa)", "G' [Pa]", "Storage modulus in Pa", "Storage modulus, Pa"
_UNIT_IN_HEADER = re.compile(
    r"^(?P<name>.*?)\s*(?:\[(?P<b>[^\]]*)\]|\((?P<p>[^)]*)\)|\bin\s+(?P<i>\S+)|,\s*(?P<c>\S+))\s*$"
)


def _unit_key(text: str) -> str:
    key = fold(text).strip("[]() ")
    return key.replace("·", ".").replace("⋅", ".").replace("μ", "µ")


def parse_unit(text: str | None, quantity: str | None = None) -> Unit | None:
    """The registry unit that `text` spells, searched within `quantity` when given."""
    if text is None:
        return None
    wanted = _unit_key(text)
    pools = [QUANTITY_BY_KEY[quantity]] if quantity else list(QUANTITIES)
    for q in pools:
        for unit in q.units:
            if any(_unit_key(name) == wanted for name in (unit.symbol, *unit.aliases)):
                return unit
    return None


def split_header(header: str) -> tuple[str, str | None]:
    """Separate a header like "Storage modulus (Pa)" into ("Storage modulus", "Pa")."""
    text = header.strip()
    match = _UNIT_IN_HEADER.match(text)
    if match and match.group("name"):
        unit = next(g for g in match.group("b", "p", "i", "c") if g is not None)
        # "tan(delta)" and "J(t)" carry a parenthesis that belongs to the name, not a unit;
        # square brackets are always a unit.
        if match.group("b") is not None or parse_unit(unit) is not None:
            return match.group("name").strip(), unit.strip()
    return text, None


def recognise(header: str) -> str | None:
    """The quantity key a column header names, or None.

    Whole-name alias matches only, so "Complex viscosity" is never read as "Viscosity".
    """
    name, _ = split_header(header)
    folded = fold(name)
    if folded in _ALIAS_INDEX:
        return _ALIAS_INDEX[folded]
    # Some exports put a symbol before or after the name: "G' Storage modulus".
    if " " in folded:
        for part in (folded.split(" ", 1)[1], folded.rsplit(" ", 1)[0]):
            if part in _ALIAS_INDEX:
                return _ALIAS_INDEX[part]
    return None


def is_index(header: str) -> bool:
    return fold(split_header(header)[0]) in INDEX_ALIASES


def looks_like_unit(text: str) -> bool:
    """True for a cell from a units row: empty, bracketed, or a unit the registry knows."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[0] in "[(" and stripped[-1] in "])":
        return True
    return parse_unit(stripped) is not None


def to_canonical(values: list[float | None], unit: Unit) -> list[float | None]:
    if unit.factor == 1.0 and unit.offset == 0.0:
        return values
    return [None if v is None else v * unit.factor + unit.offset for v in values]


def registry_json() -> list[dict]:
    """The registry as the browser needs it. A display value is (canonical - offset) / factor."""
    return [
        {
            "key": q.key,
            "name": q.name,
            "symbol": q.symbol,
            "unit": q.canonical.symbol,
            "log": q.positive,
            "units": [
                {"symbol": u.symbol, "factor": u.factor, "offset": u.offset} for u in q.units
            ],
        }
        for q in QUANTITIES
    ]
