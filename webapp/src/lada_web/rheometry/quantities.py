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


# Unit names are matched with their case: "mPa" is a thousandth of a pascal and "MPa" a million
# pascals. Aliases are therefore written in their proper case too (see parse_unit).
_PA = (
    Unit("Pa", 1.0, aliases=("N/m2", "N/m^2", "N/m²")),
    Unit("kPa", 1e3),
    Unit("MPa", 1e6),
    Unit("GPa", 1e9),
    Unit("mPa", 1e-3),
    Unit("dyn/cm²", 0.1, aliases=("dyn/cm2", "dyn/cm^2", "dyne/cm2", "dyne/cm²")),
)
_PAS = (
    Unit("Pa·s", 1.0, aliases=("Pa.s", "Pa s", "Pas", "Pa*s", "Pa-s")),
    Unit("mPa·s", 1e-3, aliases=("mPa.s", "mPa s", "mPas", "mPa*s", "cP", "cps")),
    Unit("kPa·s", 1e3, aliases=("kPa.s", "kPa s", "kPas", "kPa*s")),
    Unit("MPa·s", 1e6, aliases=("MPa.s", "MPa s", "MPas", "MPa*s")),
    Unit("P", 0.1, aliases=("poise",)),
)
_INV_PA = (
    Unit("1/Pa", 1.0, aliases=("Pa^-1", "Pa-1", "Pa⁻¹")),
    Unit("1/kPa", 1e-3, aliases=("kPa^-1", "kPa-1", "kPa⁻¹")),
    Unit("1/MPa", 1e-6, aliases=("MPa^-1", "MPa-1", "MPa⁻¹")),
    Unit("1/GPa", 1e-9, aliases=("GPa^-1", "GPa-1", "GPa⁻¹")),
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
        # A strain "in no unit" is a fraction. An empty source unit, though, means the file gave
        # none and the canonical % was assumed, so a column read as a fraction records "1".
        (
            Unit("%", 1.0, aliases=("percent", "pct")),
            Unit("", 100.0, aliases=("1", "-", "–", "mm/mm")),
        ),
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
            "viskosität",
            "viskositat",
            "eta",
            "η",
        ),
    ),
    Quantity(
        "dynamic_viscosity",
        "Dynamic viscosity",
        "η′",
        _PAS,
        # η′ = G″/ω, the real part of η*: what TRIOS and RheoCompass call "dynamic viscosity".
        aliases=("dynamic viscosity", "eta'", "η'", "η′", "dynamische viskosität"),
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
            Unit("°C", 1.0, aliases=("degC", "deg C", "Celsius", "oC", "C")),
            Unit("K", 1.0, -273.15, aliases=("Kelvin", "°K")),
            Unit("°F", 5 / 9, -160 / 9, aliases=("degF", "deg F", "F")),
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
            Unit("N·m", 1.0, aliases=("Nm", "N.m", "N m", "N*m")),
            Unit("mN·m", 1e-3, aliases=("mNm", "mN.m", "mN m", "mN*m")),
            Unit("µN·m", 1e-6, aliases=("µNm", "µN.m", "µN m", "uN.m", "uNm", "uN m")),
            # Gram-force centimetres: what TA Orchestrator (ARES) reports torque in.
            Unit(
                "g·cm",
                9.80665e-5,
                aliases=("g-cm", "gcm", "g cm", "g.cm", "gf·cm", "gf-cm", "gf cm", "gf.cm"),
            ),
        ),
        aliases=("torque", "drehmoment", "oscillation torque"),
        positive=False,
    ),
)

QUANTITY_BY_KEY: dict[str, Quantity] = {q.key: q for q in QUANTITIES}

# Quantities a header may name for one another, which the unit tells apart: "Frequency (rad/s)".
SIBLINGS: dict[str, str] = {"frequency": "angular_frequency", "angular_frequency": "frequency"}

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
    """NFKC-normalised, outer brackets dropped, inner whitespace collapsed; case kept."""
    key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip("[]() ")
    return key.replace("·", ".").replace("⋅", ".").replace("μ", "µ")


def _unit_names(units) -> list[tuple[Unit, str]]:
    return [(unit, _unit_key(name)) for unit in units for name in (unit.symbol, *unit.aliases)]


_ALL_UNIT_KEYS = frozenset(
    key.casefold() for _q in QUANTITIES for _, key in _unit_names(_q.units) if key
)


def _prefix_flipped(given: str, name: str) -> bool:
    """True when `given` differs from `name` in the case of a leading m that is a prefix:
    "MPa" against "mPa", "MHz" against "mHz". The case is all that tells milli from mega."""
    first = next((i for i, ch in enumerate(name) if ch.isalpha()), None)
    if first is None or name[first] not in "mM" or given[first : first + 1] == name[first]:
        return False
    return name[first + 1 :].casefold() in _ALL_UNIT_KEYS


def parse_unit(text: str | None, quantity: str | None = None) -> Unit | None:
    """The registry unit that `text` spells, searched within `quantity` when given.

    Matched with its case first. A spelling that matches only when case is ignored ("PA",
    "pa.s", "HZ") is accepted when exactly one unit fits and no SI prefix changes meaning
    with the case: "MPa·s" is never read as mPa·s, it is simply not a unit we know.
    """
    if text is None:
        return None
    wanted = _unit_key(text)
    pools = [QUANTITY_BY_KEY[quantity]] if quantity else list(QUANTITIES)
    names = [pair for q in pools for pair in _unit_names(q.units)]
    for unit, key in names:
        if key == wanted:
            return unit
    folded = wanted.casefold()
    hits = list(
        dict.fromkeys(
            unit
            for unit, key in names
            if key.casefold() == folded and not _prefix_flipped(wanted, key)
        )
    )
    return hits[0] if len(hits) == 1 else None


def _paren_is_unit(header: str, name: str, unit: str) -> bool:
    """Whether "(...)" at the end of a header is its unit. "Strain (mm/mm)" and "Stress (psi)"
    carry a unit, known or not; "tan(delta)", "J(t)" and "G(t)" carry part of the name."""
    if parse_unit(unit) is not None:
        return True
    if fold(header) in _ALIAS_INDEX:
        return False
    # "J(t)" has the bracket against the name; "Stress (psi)" has a space before it.
    attached = header[len(name) : len(name) + 1] == "("
    return not attached or bool(re.search(r"[/.%°·⋅^*\d²³⁻¹]", unit))


def split_header(header: str) -> tuple[str, str | None]:
    """Separate a header like "Storage modulus (Pa)" into ("Storage modulus", "Pa")."""
    text = header.strip()
    match = _UNIT_IN_HEADER.match(text)
    if match and match.group("name"):
        unit = next(g for g in match.group("b", "p", "i", "c") if g is not None).strip()
        name = match.group("name")
        if match.group("b") is not None:
            # Square brackets are always a unit; empty ones ("[]") say there is none.
            return name.strip(), unit or None
        if match.group("p") is not None:
            if _paren_is_unit(text, name, unit):
                return name.strip(), unit or None
        elif parse_unit(unit) is not None:
            return name.strip(), unit
    return text, None


def recognise(header: str) -> str | None:
    """The quantity key a column header names, or None.

    Whole-name alias matches only, so "Complex viscosity" is never read as "Viscosity" and
    "Normal stress" never as shear stress.
    """
    name, _ = split_header(header)
    folded = fold(name)
    if folded in _ALIAS_INDEX:
        return _ALIAS_INDEX[folded]
    # Some exports put a symbol before or after the name: "G' Storage modulus", "Shear stress
    # τ". The extra word must name the same quantity; otherwise it changes what is meant.
    if " " in folded:
        head, rest = folded.split(" ", 1)
        body, tail = folded.rsplit(" ", 1)
        for extra, part in ((head, rest), (tail, body)):
            key = _ALIAS_INDEX.get(part)
            if key is not None and _ALIAS_INDEX.get(extra) == key:
                return key
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
                {
                    "symbol": u.symbol,
                    "factor": u.factor,
                    "offset": u.offset,
                    # The spellings a file may use, so the page matches them as parse_unit does.
                    "aliases": list(u.aliases),
                }
                for u in q.units
            ],
        }
        for q in QUANTITIES
    ]
