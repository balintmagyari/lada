"""Quantities a block does not carry but can be computed from the ones it does.

Only what follows exactly from the definitions is added, never anything fitted:

    tan δ = G″ / G′          |G*| = √(G′² + G″²)          |η*| = |G*| / ω
    ω = 2π f                  f = ω / 2π                   δ = atan(tan δ)  (in degrees)

A derived column is flagged `derived=True`, so the interface can say so, and is dropped and
recomputed whenever the block is re-derived (after the user corrects a column's quantity).
"""

import math

from .model import Block, Column
from .quantities import QUANTITY_BY_KEY


def _first(block: Block, quantity: str) -> Column | None:
    for column in block.columns:
        if column.quantity == quantity:
            return column
    return None


def _combine(fn, *series: list[float | None]) -> list[float | None]:
    out: list[float | None] = []
    for values in zip(*series, strict=True):
        if any(v is None for v in values):
            out.append(None)
            continue
        try:
            result = fn(*values)
        except (ZeroDivisionError, ValueError):
            result = None
        out.append(result if result is not None and math.isfinite(result) else None)
    return out


def _add(block: Block, quantity: str, values: list[float | None]) -> None:
    if all(v is None for v in values):
        return
    spec = QUANTITY_BY_KEY[quantity]
    keys = {c.key for c in block.columns}
    key = quantity if quantity not in keys else f"{quantity}_derived"
    block.columns.append(
        Column(
            key=key,
            label=spec.name,
            quantity=quantity,
            unit=spec.canonical.symbol,
            source_unit="",
            derived=True,
            values=values,
        )
    )


def derive(block: Block) -> Block:
    """Recompute the block's derived columns in place, and return it."""
    block.columns = [c for c in block.columns if not c.derived]

    omega = _first(block, "angular_frequency")
    freq = _first(block, "frequency")
    if omega is None and freq is not None:
        _add(block, "angular_frequency", _combine(lambda f: 2 * math.pi * f, freq.values))
    elif freq is None and omega is not None:
        _add(block, "frequency", _combine(lambda w: w / (2 * math.pi), omega.values))
    omega = _first(block, "angular_frequency")

    g1 = _first(block, "storage_modulus")
    g2 = _first(block, "loss_modulus")
    if g1 and g2:
        if _first(block, "tan_delta") is None:
            _add(block, "tan_delta", _combine(lambda a, b: b / a, g1.values, g2.values))
        if _first(block, "complex_modulus") is None:
            _add(block, "complex_modulus", _combine(math.hypot, g1.values, g2.values))

    tan = _first(block, "tan_delta")
    if tan and _first(block, "phase_angle") is None:
        _add(block, "phase_angle", _combine(lambda t: math.degrees(math.atan(t)), tan.values))

    gstar = _first(block, "complex_modulus")
    if gstar and omega and _first(block, "complex_viscosity") is None:
        _add(block, "complex_viscosity", _combine(lambda g, w: g / w, gstar.values, omega.values))
    return block
