"""Time-temperature superposition: shift frequency sweeps measured at several temperatures
onto one master curve at a reference temperature.

Each sweep is shifted horizontally by log aT, found by overlap, working outwards from the
reference temperature (whose aT is 1 by definition). A sweep is first matched against the
nearest sweep already in place whose shift is known (found or set by hand): the shift that
makes their log G′ and log G″ against log ω coincide best. That shift is then refined against
every sweep already in place, so the master curve is built against itself rather than as a
chain of neighbours, whose small errors would add up. A sweep whose shift cannot be found is
never used to place another, and is left out of the fits. An optional vertical correction
divides the moduli by bT = T / T_ref (temperatures in kelvin), the usual entropic correction
at constant density.

A shift is not found when the sweeps do not overlap, do not match in shape, only overlap at
the very edge of what the search allows while the match is still improving (the true shift
lies beyond), or are so flat over the overlap that a wide range of shifts fits equally well.

Any shift can be set by hand instead; the fits then use the values actually applied:

    WLF          log aT = -C1 (T - T_ref) / (C2 + T - T_ref)
    Arrhenius    ln aT  = (Ea / R) (1/T - 1/T_ref)
"""

import math
import warnings
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit, minimize_scalar

R_GAS = 8.314462618  # J/(mol K)
KELVIN = 273.15
SEARCH_DECADES = 15.0  # how far a shift is searched for, either way
GRID_STEP = 0.02  # decades between the shifts tried before refining
# A shift only counts when the sweeps really lie on one curve after it: an rms mismatch above
# this many decades (a factor of about 1.8) means overlap found no shift worth trusting.
MAX_RMS_DECADES = 0.25
# A minimum is undetermined when shifts over more than this many decades fit about as well
# as the best one (the mismatch rises by less than a quarter of its own size, or by less than
# 0.01 decade rms): the moduli are too flat over the overlap to say where the sweep belongs.
FLAT_DECADES = 0.5
FLAT_FLOOR = 1e-4  # mean squared log mismatch, i.e. 0.01 decade rms
# The refinement against the whole master curve looks only this far from the first shift.
REFINE_DECADES = 0.1
MASTER_SWEEPS = 12  # the placed sweeps nearest in temperature it is refined against
SPARE_ANCHORS = 2  # other placed sweeps tried when the nearest one gives no shift
MAX_POINTS = 5000  # per curve and per column

Finite = Annotated[float, Field(allow_inf_nan=False)]


class Curve(BaseModel):
    id: str
    label: str = ""
    temperature: float = Field(gt=-KELVIN, allow_inf_nan=False)  # °C
    omega: list[float | None] = Field(max_length=MAX_POINTS)  # rad/s
    storage: list[float | None] = Field(default_factory=list, max_length=MAX_POINTS)  # G′, Pa
    loss: list[float | None] = Field(default_factory=list, max_length=MAX_POINTS)  # G″, Pa


class TTSRequest(BaseModel):
    curves: list[Curve] = Field(min_length=1, max_length=60)
    # °C; the measured temperature nearest to it is used
    reference: float | None = Field(default=None, allow_inf_nan=False)
    use: Literal["both", "storage", "loss"] = "both"
    vertical: bool = False
    manual: dict[str, Finite] = Field(default_factory=dict)  # curve id -> log10 aT


class Shift(BaseModel):
    id: str
    label: str
    temperature: float
    log_at: float  # applied; where no shift was found, the shift of the sweep it was matched to
    auto_log_at: float | None  # what overlap found; None where it could not
    log_bt: float = 0.0
    manual: bool = False
    overlap: int = 0  # points that overlapped the sweep it was matched against
    fitted: bool = True  # whether the WLF and Arrhenius fits use it


class WLF(BaseModel):
    c1: float
    c2: float  # a temperature difference, so the same in K and °C
    rms: float  # in log aT


class Arrhenius(BaseModel):
    ea_kj_per_mol: float
    rms: float


class TTSResult(BaseModel):
    reference: float  # °C, the measured temperature everything is shifted to
    reference_id: str = ""  # the one sweep that is the reference (others may share its T)
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


def _penalised(pairs, min_overlap: int):
    """The mismatch as a function a bounded minimiser can handle: no infinities."""

    def f(s: float) -> float:
        value = _mismatch(s, pairs, min_overlap)[0]
        return value if math.isfinite(value) else 1e6

    return f


def _search(pairs, min_overlap: int):
    """Grid search then refinement for one overlap requirement.

    Returns (shift, mse, overlap, problem): problem is "" for a trustworthy minimum, or
    "none" (no shift gives the overlap), "edge" (the best shift sits where the overlap runs
    out, still improving towards it), "flat" (a wide range of shifts fits as well), or
    "shape" (even the best shift leaves the sweeps far apart)."""
    grid = np.arange(-SEARCH_DECADES, SEARCH_DECADES, GRID_STEP)
    scores = np.array([_mismatch(s, pairs, min_overlap)[0] for s in grid])
    finite = np.isfinite(scores)
    if not finite.any():
        return None, math.inf, 0, "none"
    k = int(np.argmin(np.where(finite, scores, np.inf)))
    best = float(scores[k])

    # How wide is the bottom of the valley? Walk out while the mismatch stays near its best.
    tol = max(0.25 * best, FLAT_FLOOR)
    lo = k
    while lo > 0 and finite[lo - 1] and scores[lo - 1] <= best + tol:
        lo -= 1
    hi = k
    while hi < len(grid) - 1 and finite[hi + 1] and scores[hi + 1] <= best + tol:
        hi += 1

    f = _penalised(pairs, min_overlap)
    coarse = float(grid[k])
    fine = minimize_scalar(
        f,
        bounds=(coarse - 1.5 * GRID_STEP, coarse + 1.5 * GRID_STEP),
        method="bounded",
        options={"xatol": 1e-5},
    )
    shift = float(fine.x) if fine.fun < 1e6 else coarse
    mse, overlap = _mismatch(shift, pairs, min_overlap)
    if not math.isfinite(mse):
        shift, (mse, overlap) = coarse, _mismatch(coarse, pairs, min_overlap)

    if math.sqrt(mse) > MAX_RMS_DECADES:
        return shift, mse, overlap, "shape"
    if (hi - lo) * GRID_STEP > FLAT_DECADES:
        return shift, mse, overlap, "flat"
    # Pinned to where the overlap runs out: a step further would leave too few points, and
    # the mismatch is still falling in that direction, so the real minimum lies beyond.
    step = 5e-3
    for direction in (-1.0, 1.0):
        edge_index = k + int(direction)
        near_edge = edge_index < 0 or edge_index >= len(grid) or not finite[edge_index]
        if not near_edge:
            continue
        beyond = _mismatch(shift + direction * step, pairs, min_overlap)[0]
        if not math.isfinite(beyond) or beyond < mse:
            return shift, mse, overlap, "edge"
    if abs(shift) > SEARCH_DECADES - 0.1:
        return shift, mse, overlap, "edge"
    return shift, mse, overlap, ""


def _min_overlaps(pairs) -> tuple[int, ...]:
    sizes = [min(len(m[0]), len(f[0])) for m, f in pairs]
    # Ask for a real overlap first (a quarter of the shorter sweep, at least 3 points);
    # only if no shift gives that, or the best one is pinned where it runs out, settle for 2.
    return (max(3, math.ceil(min(sizes) / 4)), 2)


def best_shift(pairs) -> tuple[float | None, int, str]:
    """The horizontal shift, in decades, that lays each `moving` series onto its `fixed`
    partner, the points that overlapped, and why it is None when it could not be found
    ("" when found; otherwise "none", "edge", "flat" or "shape", as in `_search`)."""
    if not pairs:
        return None, 0, "none"
    problem, overlap = "none", 0
    for min_overlap in _min_overlaps(pairs):
        shift, _, n, why = _search(pairs, min_overlap)
        if why == "":
            return shift, n, ""
        if why != "none":
            problem, overlap = why, n
        if why in ("flat", "shape"):
            break  # a smaller overlap would not make it any less flat or better matched
    return None, overlap, problem


def _refine(start: float, pairs) -> float:
    """`start`, moved to where the sweep best matches every sweep already in place."""
    if not pairs:
        return start
    min_overlap = _min_overlaps(pairs)[0]
    f = _penalised(pairs, min_overlap)
    if f(start) >= 1e6:
        min_overlap = 2
        f = _penalised(pairs, min_overlap)
        if f(start) >= 1e6:
            return start
    fine = minimize_scalar(
        f,
        bounds=(start - REFINE_DECADES, start + REFINE_DECADES),
        method="bounded",
        options={"xatol": 1e-5},
    )
    shift = float(fine.x)
    mse = _mismatch(shift, pairs, min_overlap)[0]
    if not math.isfinite(mse) or math.sqrt(mse) > MAX_RMS_DECADES or fine.fun > f(start):
        return start
    return shift


def _log_bt(temperature: float, reference: float) -> float:
    return math.log10((temperature + KELVIN) / (reference + KELVIN))


def _why(name: str, problem: str, anchor: str) -> str:
    if problem == "edge":
        reason = (
            f"{name} overlaps {anchor} only at the very edge, and the match was still "
            "improving there, so the real shift lies beyond what the overlap shows"
        )
    elif problem == "flat":
        reason = (
            f"{name} is too flat where it overlaps {anchor}: shifts over a wide range fit "
            "it equally well (try the other modulus, or both)"
        )
    else:
        reason = (
            f"{name} does not overlap the sweep next to it ({anchor}), or does not match its shape"
        )
    return (
        f"{reason}, so its shift could not be found. It is drawn at the shift of {anchor} "
        "and left out of the fits; set it by hand."
    )


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
    # Sweeps whose place is known (found or set by hand), and each one's moduli moved there.
    placed: dict[int, dict] = {}

    def settle(i: int) -> None:
        moved = {}
        for k in keys:
            s = series[i][0][k]
            if s is not None:
                x = s[0] + applied[i]
                moved[k] = (x, s[1], PchipInterpolator(x, s[1]))
        placed[i] = moved

    def pairs_against(i: int, others) -> list:
        pairs = []
        for j in others:
            for k in keys:
                moving, fixed = series[i][0][k], placed[j].get(k)
                if moving is not None and fixed is not None:
                    pairs.append((moving, fixed))
        return pairs

    def name(i: int) -> str:
        return curves[i].label or curves[i].id

    def place(i: int, anchor: int) -> None:
        # The placed sweeps nearest in temperature, and always the reference itself.
        master = sorted(placed, key=lambda j: abs(temps[j] - temps[i]))[:MASTER_SWEEPS]
        if ref_index not in master:
            master.append(ref_index)
        shift, n, problem = best_shift(pairs_against(i, [anchor]))
        # A sweep that does not match the one next to it (a hand-set shift on poor data, say)
        # may still match another already in place.
        for other in [j for j in master if j != anchor][:SPARE_ANCHORS]:
            if shift is not None:
                break
            shift, n, _ = best_shift(pairs_against(i, [other]))
        if shift is not None:
            # Then against the master curve assembled so far, not only one sweep, so small
            # errors do not add up along the chain. A dozen sweeps keeps long series quick.
            shift = _refine(shift, pairs_against(i, master))
        auto[i], overlap[i] = shift, n
        manual = request.manual.get(curves[i].id)
        if manual is not None:
            applied[i] = manual
        elif shift is not None:
            applied[i] = shift
        else:
            applied[i] = applied[anchor]  # beside the sweep it was matched to, and say so
            notes.append(_why(name(i), problem, name(anchor)))
            return
        settle(i)

    settle(ref_index)
    if curves[ref_index].id in request.manual:
        notes.append("The reference sweep is never shifted; its value stays at aT = 1.")
    anchor = ref_index
    for i in range(ref_index + 1, len(curves)):
        place(i, anchor)
        anchor = i if i in placed else anchor  # chain only from sweeps whose place is known
    anchor = ref_index
    for i in range(ref_index - 1, -1, -1):
        place(i, anchor)
        anchor = i if i in placed else anchor

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
            fitted=i in placed,
        )
        for i, c in enumerate(curves)
    ]
    result = TTSResult(
        reference=reference, reference_id=curves[ref_index].id, shifts=shifts, notes=notes
    )
    _fit(result)
    return result


# --- Fits -----------------------------------------------------------------------------------


def _wlf(dt, c1, c2):
    return -c1 * dt / (c2 + dt)


def _fit(result: TTSResult) -> None:
    # Only shifts that were found or set by hand: a placeholder would pull the fit anywhere.
    used = [s for s in result.shifts if s.fitted]
    dts = np.array([s.temperature - result.reference for s in used])
    ys = np.array([s.log_at for s in used])
    others = np.abs(dts) > 1e-9
    # Two parameters need more distinct temperatures than that, or they fit exactly and say
    # nothing; repeats at one temperature do not count twice.
    distinct = len({round(float(t), 6) for t in dts[others]})

    if distinct >= 3:
        # Start from the linear form  ΔT / log aT = -C2/C1 - ΔT / C1.
        x, v = dts[others], ys[others]
        ok = np.abs(v) > 1e-12
        c1_0, c2_0 = 10.0, 100.0
        if len({round(float(t), 6) for t in x[ok]}) >= 2:
            slope, intercept = np.polyfit(x[ok], x[ok] / v[ok], 1)
            if slope < 0:
                c1_0 = -1.0 / slope
                c2_0 = -intercept * c1_0 if -intercept * c1_0 > 0 else 100.0
        try:
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
        result.wlf_note = (
            "A WLF fit needs shifts (found or set by hand) at three or more temperatures "
            "besides the reference."
        )

    if distinct >= 1:
        tk = dts + result.reference + KELVIN
        x = 1.0 / tk - 1.0 / (result.reference + KELVIN)
        y = ys * math.log(10)
        slope = float(np.dot(x, y) / np.dot(x, x))  # through the origin: aT(T_ref) = 1
        rms = float(np.sqrt(np.mean((slope * x - y) ** 2)) / math.log(10))
        result.arrhenius = Arrhenius(ea_kj_per_mol=slope * R_GAS / 1000.0, rms=rms)
    else:
        result.arrhenius_note = (
            "An Arrhenius fit needs shifts (found or set by hand) at two or more temperatures, "
            "the reference and one more."
        )
