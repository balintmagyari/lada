"""Time-temperature superposition: shift frequency sweeps measured at several temperatures
onto one master curve at a reference temperature.

Each sweep is shifted horizontally by log aT, found by overlap: the shift that makes its
log G′ and log G″ against log ω coincide best with the neighbouring sweep that is already in
place, working outwards from the reference temperature (whose aT is 1 by definition). An
optional vertical correction divides the moduli by bT = T / T_ref (temperatures in kelvin),
the usual entropic correction at constant density.

Any shift can be set by hand instead; the fits then use the values actually applied:

    WLF          log aT = -C1 (T - T_ref) / (C2 + T - T_ref)
    Arrhenius    ln aT  = (Ea / R) (1/T - 1/T_ref)
"""

import math
import warnings
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit, minimize_scalar

R_GAS = 8.314462618  # J/(mol K)
KELVIN = 273.15
SEARCH_DECADES = 15.0  # how far a shift is searched for, either way
# A shift only counts when the sweeps really lie on one curve after it: an rms mismatch above
# this many decades (a factor of about 1.8) means overlap found no shift worth trusting.
MAX_RMS_DECADES = 0.25


class Curve(BaseModel):
    id: str
    label: str = ""
    temperature: float  # °C
    omega: list[float | None]  # rad/s
    storage: list[float | None] = Field(default_factory=list)  # G′, Pa
    loss: list[float | None] = Field(default_factory=list)  # G″, Pa


class TTSRequest(BaseModel):
    curves: list[Curve] = Field(min_length=1, max_length=60)
    reference: float | None = None  # °C; the measured temperature nearest to it is used
    use: Literal["both", "storage", "loss"] = "both"
    vertical: bool = False
    manual: dict[str, float] = Field(default_factory=dict)  # curve id -> log10 aT


class Shift(BaseModel):
    id: str
    label: str
    temperature: float
    log_at: float
    auto_log_at: float | None  # what overlap found; None where it could not
    log_bt: float = 0.0
    manual: bool = False
    overlap: int = 0  # points that overlapped the neighbour it was shifted onto


class WLF(BaseModel):
    c1: float
    c2: float  # a temperature difference, so the same in K and °C
    rms: float  # in log aT


class Arrhenius(BaseModel):
    ea_kj_per_mol: float
    rms: float


class TTSResult(BaseModel):
    reference: float  # °C, the measured temperature everything is shifted to
    shifts: list[Shift]
    wlf: WLF | None = None
    wlf_note: str = ""
    arrhenius: Arrhenius | None = None
    arrhenius_note: str = ""
    notes: list[str] = Field(default_factory=list)


# --- Overlap shifting ------------------------------------------------------------------------


def _log_series(omega, values, log_bt: float = 0.0):
    """(log ω, log G) for the points where both are positive and finite, sorted by ω."""
    pairs = sorted(
        (math.log10(w), math.log10(g) - log_bt)
        for w, g in zip(omega, values, strict=False)
        if w is not None
        and g is not None
        and w > 0
        and g > 0
        and math.isfinite(w)
        and math.isfinite(g)
    )
    if len(pairs) < 2:
        return None
    x, y = np.array(pairs).T
    keep = np.concatenate(([True], np.diff(x) > 0))  # PCHIP needs strictly increasing ω
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return None
    return x, y, PchipInterpolator(x, y)


def _one_way(shift: float, moving, fixed) -> tuple[float, int]:
    """Mean squared log mismatch of `moving` (shifted by `shift` decades) against `fixed`,
    interpolated, over the points of `moving` that fall inside `fixed`'s range."""
    x = moving[0] + shift
    inside = (x >= fixed[0][0]) & (x <= fixed[0][-1])
    n = int(inside.sum())
    if n == 0:
        return math.inf, 0
    # A shape-preserving cubic through the fixed sweep: straight lines between its points
    # would cut the corner of a curved modulus and bias every shift the same way.
    curve = fixed[2] if len(fixed) > 2 else PchipInterpolator(fixed[0], fixed[1])
    residual = moving[1][inside] - curve(x[inside])
    return float(np.mean(residual**2)), n


def _mismatch(shift: float, pairs, min_overlap: int) -> tuple[float, int]:
    """Symmetric mismatch over every modulus in use; inf when the overlap is too small."""
    total, weight, overlap = 0.0, 0, 0
    for moving, fixed in pairs:
        a, na = _one_way(shift, moving, fixed)
        b, nb = _one_way(-shift, fixed, moving)
        if na < min_overlap or nb < min_overlap:
            continue
        total += a * na + b * nb
        weight += na + nb
        overlap = max(overlap, na)
    if weight == 0:
        return math.inf, 0
    return total / weight, overlap


def best_shift(pairs) -> tuple[float | None, int]:
    """The horizontal shift, in decades, that lays each `moving` series onto its `fixed`
    partner; None when the two never overlap well enough to say."""
    sizes = [min(len(m[0]), len(f[0])) for m, f in pairs]
    if not sizes:
        return None, 0
    grid = np.arange(-SEARCH_DECADES, SEARCH_DECADES, 0.02)
    # Ask for a real overlap first (a quarter of the shorter sweep, at least 3 points);
    # only if no shift gives that, settle for 2.
    for min_overlap in (max(3, math.ceil(min(sizes) / 4)), 2):
        scores = np.array([_mismatch(s, pairs, min_overlap)[0] for s in grid])
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        coarse = float(grid[int(np.argmin(np.where(finite, scores, np.inf)))])
        fine = minimize_scalar(
            lambda s, mo=min_overlap: _mismatch(s, pairs, mo)[0],
            bounds=(coarse - 0.03, coarse + 0.03),
            method="bounded",
            options={"xatol": 1e-5},
        )
        shift = float(fine.x) if np.isfinite(fine.fun) else coarse
        mse, overlap = _mismatch(shift, pairs, min_overlap)
        at_edge = abs(shift) > SEARCH_DECADES - 0.1
        if at_edge or not math.isfinite(mse) or math.sqrt(mse) > MAX_RMS_DECADES:
            return None, overlap
        return shift, overlap
    return None, 0


def _log_bt(temperature: float, reference: float) -> float:
    return math.log10((temperature + KELVIN) / (reference + KELVIN))


def shift_factors(request: TTSRequest) -> TTSResult:
    curves = sorted(request.curves, key=lambda c: c.temperature)
    notes: list[str] = []
    temps = [c.temperature for c in curves]
    target = request.reference if request.reference is not None else temps[len(temps) // 2]
    ref_index = min(range(len(curves)), key=lambda i: abs(temps[i] - target))
    reference = temps[ref_index]
    if request.reference is not None and abs(reference - request.reference) > 1e-6:
        notes.append(
            f"No sweep was measured at {request.reference:g} °C, so the nearest one, "
            f"{reference:g} °C, is the reference."
        )

    keys = {"both": ("storage", "loss"), "storage": ("storage",), "loss": ("loss",)}[request.use]
    series = []
    for c in curves:
        log_bt = _log_bt(c.temperature, reference) if request.vertical else 0.0
        series.append(({k: _log_series(c.omega, getattr(c, k), log_bt) for k in keys}, log_bt))

    auto: dict[int, float | None] = {ref_index: 0.0}
    overlap: dict[int, int] = {ref_index: 0}
    applied: dict[int, float] = {ref_index: 0.0}

    def place(i: int, neighbour: int) -> None:
        pairs = []
        for k in keys:
            moving, fixed = series[i][0][k], series[neighbour][0][k]
            if moving is not None and fixed is not None:
                x = fixed[0] + applied[neighbour]
                pairs.append((moving, (x, fixed[1], PchipInterpolator(x, fixed[1]))))
        shift, n = best_shift(pairs)
        auto[i], overlap[i] = shift, n
        manual = request.manual.get(curves[i].id)
        if manual is not None:
            applied[i] = manual
        elif shift is not None:
            applied[i] = shift
        else:
            # No overlap with the neighbour: carry its shift so the sweep is at least placed
            # beside it, and say so.
            applied[i] = applied[neighbour]
            notes.append(
                f"{curves[i].label or curves[i].id} does not overlap the sweep next to it, or "
                "does not match its shape, so its shift could not be found; set it by hand."
            )

    if curves[ref_index].id in request.manual:
        notes.append("The reference sweep is never shifted; its value stays at aT = 1.")
    for i in range(ref_index + 1, len(curves)):
        place(i, i - 1)
    for i in range(ref_index - 1, -1, -1):
        place(i, i + 1)

    shifts = [
        Shift(
            id=c.id,
            label=c.label,
            temperature=c.temperature,
            log_at=applied[i],
            auto_log_at=auto.get(i),
            log_bt=series[i][1],
            manual=i != ref_index and c.id in request.manual,
            overlap=overlap.get(i, 0),
        )
        for i, c in enumerate(curves)
    ]
    result = TTSResult(reference=reference, shifts=shifts, notes=notes)
    _fit(result)
    return result


# --- Fits -----------------------------------------------------------------------------------


def _wlf(dt, c1, c2):
    return -c1 * dt / (c2 + dt)


def _fit(result: TTSResult) -> None:
    dts = np.array([s.temperature - result.reference for s in result.shifts])
    ys = np.array([s.log_at for s in result.shifts])
    others = np.abs(dts) > 1e-9

    if others.sum() >= 2:
        # Start from the linear form  ΔT / log aT = -C2/C1 - ΔT / C1.
        x, v = dts[others], ys[others]
        ok = np.abs(v) > 1e-12
        c1_0, c2_0 = 10.0, 100.0
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(x[ok], x[ok] / v[ok], 1)
            if slope < 0:
                c1_0 = -1.0 / slope
                c2_0 = -intercept * c1_0 if -intercept * c1_0 > 0 else 100.0
        try:
            # Only the parameters are used; with as many points as parameters scipy cannot
            # estimate their covariance and says so, which is expected here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                (c1, c2), _ = curve_fit(_wlf, dts, ys, p0=(c1_0, c2_0), maxfev=20000)
            if c2 <= 0 or dts.min() <= -c2 <= dts.max():
                raise RuntimeError("the WLF pole falls inside the measured range")
            rms = float(np.sqrt(np.mean((_wlf(dts, c1, c2) - ys) ** 2)))
            result.wlf = WLF(c1=float(c1), c2=float(c2), rms=rms)
        except (RuntimeError, ValueError):
            result.wlf_note = "The WLF form does not fit these shift factors."
    else:
        result.wlf_note = "A WLF fit needs sweeps at three or more temperatures."

    if others.sum() >= 1:
        tk = dts + result.reference + KELVIN
        x = 1.0 / tk - 1.0 / (result.reference + KELVIN)
        y = ys * math.log(10)
        slope = float(np.dot(x, y) / np.dot(x, x))  # through the origin: aT(T_ref) = 1
        rms = float(np.sqrt(np.mean((slope * x - y) ** 2)) / math.log(10))
        result.arrhenius = Arrhenius(ea_kj_per_mol=slope * R_GAS / 1000.0, rms=rms)
    else:
        result.arrhenius_note = "An Arrhenius fit needs sweeps at two or more temperatures."
