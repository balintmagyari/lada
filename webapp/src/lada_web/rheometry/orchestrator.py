"""TA Instruments Orchestrator data files (.RSD01, .RSD02 …), read without Orchestrator.

An Orchestrator file does not hold the table Orchestrator shows. It holds the instrument's raw
signals per point (motor displacement and torque amplitudes and their phases, frequency,
temperature, gap change, time) plus the test set-up, and Orchestrator computes G′, G″, stress
and strain from them when the file is opened. This reader does the same, so the columns come
out as Orchestrator's own text export of the file would give them.

The layout was worked out from real files and checked column by column against Orchestrator's
text exports of the same tests (frequency, strain and time sweeps, 8 mm parallel plates, both
torque transducer ranges): every value agrees to the float32 precision the file stores.

The file is "Rheo", a short header, then a chain of records, each

    tag (u32) | payload length (u32) | version (u32) | 12 reserved bytes | payload

The records read here:

    0x65  test name            0x7d  operator            0x7a  notes (RTF)
    0x7f  start (FILETIME)     0x80  Orchestrator version
    0x7c  set-up: float32s; [0] plate radius (mm), [1] gap (mm), [6] transducer in use,
          [7]/[8] full-scale torque of transducer 0/1 (g·cm)
    0x82  channel index: 16 bytes per channel, the first u32 its code
    0x83  one per channel, in index order: float32 per point (time: u16, see _time)
    0xca  column name, 0xcb its unit: the columns Orchestrator shows, in its order

Raw channel codes (the 0x82 index):

    1 frequency (rad/s)        2 temperature (°C)       3 commanded strain (%)
    4 time (s, u16)            5 displacement amplitude 7 displacement phase (rad)
    6 torque (fraction of the transducer's full scale)  8 torque phase (rad)
    9 change of gap (mm)

    strain (%)  = displacement · 100 · R / (2 (gap + gap change))
    torque      = raw torque · full scale of the transducer in use          (g·cm)
    stress (Pa) = 2 M / (π R³)
    δ           = torque phase − displacement phase
    G′, G″      = (stress / strain) · cos δ, sin δ

The strain and stress constants are the parallel-plate ones. They are the only geometry the
reference exports cover, so every file carries a warning saying so.
"""

import datetime
import math
import re
import struct

import numpy as np

MAGIC = b"Rheo"
SUFFIXES = tuple(f".rsd{n:02d}" for n in range(100))

_FIRST_RECORD = 0x30
_RECORD_HEADER = 24

_NAME, _NOTES, _SETUP, _OPERATOR, _STARTED, _VERSION = 0x65, 0x7A, 0x7C, 0x7D, 0x7F, 0x80
_INDEX, _CHANNEL, _COLUMN_NAME, _COLUMN_UNIT = 0x82, 0x83, 0xCA, 0xCB

_FREQ, _TEMP, _CMD_STRAIN, _TIME = 1, 2, 3, 4
_DISP, _TORQUE, _DISP_PHASE, _TORQUE_PHASE, _GAP_CHANGE = 5, 6, 7, 8, 9

G_CM = 9.80665e-5  # one gram-force centimetre, in N·m

DAMAGED = (
    "This TA Orchestrator file could not be read: {}. If Orchestrator opens it, export the "
    "data as text (File > Export) and import that instead."
)


def is_orchestrator(raw: bytes) -> bool:
    return raw[:4] == MAGIC


def _records(raw: bytes) -> list[tuple[int, bytes]]:
    records = []
    offset = _FIRST_RECORD
    while offset + _RECORD_HEADER <= len(raw):
        tag, length, version = struct.unpack_from("<III", raw, offset)
        end = offset + _RECORD_HEADER + length
        if version > 50 or end > len(raw):
            raise ValueError(DAMAGED.format("its records do not fit together"))
        records.append((tag, raw[offset + _RECORD_HEADER : end]))
        offset = end
    if not records:
        raise ValueError(DAMAGED.format("it holds no records"))
    return records


def _text(payload: bytes) -> str:
    return payload.split(b"\x00", 1)[0].decode("cp1252", errors="replace").strip()


def _plain_notes(rtf: str) -> str:
    """The words of an RTF note, without its formatting."""
    text = re.sub(r"\{\\fonttbl.*?\}\}", "", rtf, flags=re.DOTALL)
    text = re.sub(r"\\par[d]?\b", "\n", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _started(payload: bytes) -> str | None:
    if len(payload) < 8:
        return None
    ticks = struct.unpack_from("<Q", payload)[0]  # 100 ns since 1601, UTC
    try:
        moment = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=ticks / 10)
    except OverflowError:
        return None
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _time(payload: bytes, n: int) -> np.ndarray | None:
    """Seconds since the test started. The record holds u16 seconds: the offset of the first
    point, a zero, then each point's time since the first, wrapping after 65 535 s."""
    raw = np.frombuffer(payload[: len(payload) // 2 * 2], "<u2").astype(float)
    if len(raw) < n + 2:
        return None
    since_first = raw[2 : 2 + n]
    wraps = np.concatenate(([0.0], np.cumsum(np.diff(since_first) < 0)))
    return raw[0] + since_first + 65536.0 * wraps


class Orchestrator:
    """What read() found: the test's metadata and name, and its table as Orchestrator shows
    it (labels, units, one list of values per column), plus warnings for the import."""

    def __init__(self, metadata, name, labels, units, values, warnings):
        self.metadata: dict[str, str] = metadata
        self.name: str = name
        self.labels: list[str] = labels
        self.units: list[str] = units
        self.values: list[list[float]] = values
        self.warnings: list[str] = warnings


def read(raw: bytes) -> Orchestrator:
    if not is_orchestrator(raw):
        raise ValueError(DAMAGED.format("it does not start like one"))
    first: dict[int, bytes] = {}
    channels: list[bytes] = []
    names: list[str] = []
    units: list[str] = []
    for tag, payload in _records(raw):
        first.setdefault(tag, payload)
        if tag == _CHANNEL:
            channels.append(payload)
        elif tag == _COLUMN_NAME:
            names.append(_text(payload))
        elif tag == _COLUMN_UNIT:
            units.append(payload.split(b"\x00", 1)[0].decode("cp1252", errors="replace").strip())
    if _SETUP not in first or _INDEX not in first or not channels:
        raise ValueError(DAMAGED.format("its test set-up or its data is missing"))

    index = first[_INDEX]
    codes = [struct.unpack_from("<I", index, i)[0] for i in range(0, len(index) - 15, 16)]
    if len(codes) != len(channels):
        raise ValueError(DAMAGED.format("its channel index does not match its data"))
    by_code = dict(zip(codes, channels, strict=True))
    needed = (_FREQ, _TEMP, _CMD_STRAIN, _DISP, _TORQUE, _DISP_PHASE, _TORQUE_PHASE)
    if any(code not in by_code for code in needed):
        raise ValueError(
            DAMAGED.format(
                "it does not hold the oscillation signals this app reads (only dynamic "
                "oscillation tests are supported so far)"
            )
        )

    def floats(code: int) -> np.ndarray:
        payload = by_code[code]
        return np.frombuffer(payload[: len(payload) // 4 * 4], "<f4").astype(float)

    frequency = floats(_FREQ)
    n = len(frequency)
    if n == 0:
        raise ValueError(DAMAGED.format("it holds no measured points"))
    signals = {code: floats(code) for code in needed}
    if any(len(v) != n for v in signals.values()):
        raise ValueError(DAMAGED.format("its channels hold different numbers of points"))
    gap_change = floats(_GAP_CHANGE) if _GAP_CHANGE in by_code else np.zeros(n)
    if len(gap_change) != n:
        gap_change = np.zeros(n)

    setup = np.frombuffer(first[_SETUP][: len(first[_SETUP]) // 4 * 4], "<f4").astype(float)
    if len(setup) < 9:
        raise ValueError(DAMAGED.format("its test set-up is too short"))
    radius, gap, transducer = setup[0], setup[1], setup[6]
    if not (0.5 <= radius <= 50 and 0 < gap <= 10) or transducer not in (0.0, 1.0):
        raise ValueError(
            DAMAGED.format(
                "its geometry is not one this app understands (a radius of "
                f"{radius:g} mm and a gap of {gap:g} mm were read)"
            )
        )
    full_scale = setup[7] if transducer == 0.0 else setup[8]

    strain = signals[_DISP] * 100 * radius / (2 * (gap + gap_change))  # %
    torque = signals[_TORQUE] * full_scale  # g·cm
    stress = torque * G_CM * 2 / (math.pi * (radius / 1000) ** 3)  # Pa
    delta = np.mod(signals[_TORQUE_PHASE] - signals[_DISP_PHASE] + math.pi, 2 * math.pi)
    delta = delta - math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        modulus = np.where(strain != 0, stress / (strain / 100), np.nan)
        computed = {
            "freq": frequency,
            "g'": modulus * np.cos(delta),
            'g"': modulus * np.sin(delta),
            "g*": modulus,
            "tan_delta": np.tan(delta),
            "tan(delta)": np.tan(delta),
            "delta": np.degrees(delta),
            "torque": torque,
            "cmd strain": signals[_CMD_STRAIN],
            "strain": strain,
            "temp": signals[_TEMP],
            "stress": stress,
            "eta*": modulus / frequency,
            "eta'": modulus * np.sin(delta) / frequency,
            'eta"': modulus * np.cos(delta) / frequency,
        }
    elapsed = _time(by_code[_TIME], n) if _TIME in by_code else None
    if elapsed is not None:
        computed["time"] = elapsed

    labels: list[str] = []
    column_units: list[str] = []
    values: list[list[float]] = []
    skipped: list[str] = []
    units = units + [""] * (len(names) - len(units))
    for name, unit in zip(names, units, strict=False):
        key = name.lower()
        if key == "clocktime":
            continue  # time of day: the start is in the metadata and "time" counts from it
        if key not in computed:
            skipped.append(name)
            continue
        series = computed[key]
        if key == "delta" and unit == "rad":
            series = delta
        labels.append(name)
        column_units.append(unit)
        values.append([float(f"{v:.7g}") if math.isfinite(v) else math.nan for v in series])
    if elapsed is not None and "time" not in (label.lower() for label in labels):
        labels.append("time")
        column_units.append("s")
        values.append([float(v) for v in elapsed])

    warnings: list[str] = []
    if skipped:
        warnings.append(
            "Orchestrator columns this app does not compute yet were left out: "
            + ", ".join(f"“{s}”" for s in skipped)
            + ". Export the file as text from Orchestrator to get them."
        )
    warnings.append(
        "Computed from the raw signals in the Orchestrator file with the parallel-plate "
        "formulas, which were checked against Orchestrator's own exports for 8 mm plates. For "
        "a cone-and-plate or another geometry, compare one point with Orchestrator before "
        "relying on the moduli."
    )

    metadata: dict[str, str] = {}
    test_name = _text(first.get(_NAME, b""))
    if test_name:
        metadata["Test"] = test_name
    if first.get(_OPERATOR):
        metadata["Operator"] = _text(first[_OPERATOR])
    if first.get(_STARTED) and (started := _started(first[_STARTED])):
        metadata["Started"] = started
    if first.get(_VERSION):
        metadata["Orchestrator version"] = _text(first[_VERSION])
    metadata["Plate diameter"] = f"{2 * radius:g} mm"
    metadata["Gap"] = f"{gap:.4g} mm"
    metadata["Transducer range"] = f"{full_scale:.5g} g·cm"
    if first.get(_NOTES) and (notes := _plain_notes(_text(first[_NOTES]))):
        metadata["Notes"] = notes[:2000]
    return Orchestrator(metadata, test_name, labels, column_units, values, warnings)
