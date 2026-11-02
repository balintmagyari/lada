"""Guess what kind of test a block holds from which quantity was varied.

The guess only picks the ready-made plot a block is offered under; it never hides data, and
any block can still be drawn with Custom plot. When the evidence is ambiguous the answer is
"unknown" rather than a confident wrong label.
"""

import math
import statistics

from .model import Block, TestType

# A swept quantity spans at least this many decades (frequency, strain, rate...) or this many
# degrees (temperature). Anything less is treated as held constant.
MIN_DECADES = 0.5
MIN_KELVIN = 5.0
CONSTANT_KELVIN = 1.5


def _values(block: Block, quantity: str) -> list[float]:
    for column in sorted(block.columns, key=lambda c: c.derived):
        if column.quantity == quantity:
            return [v for v in column.values if v is not None]
    return []


def _has(block: Block, quantity: str) -> bool:
    return any(c.quantity == quantity for c in block.columns)


def _decades(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return 0.0
    return math.log10(max(positive) / min(positive))


def _span(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def detect_test_type(block: Block) -> TestType:
    if _has(block, "relaxation_modulus"):
        return "stress_relaxation"
    if _has(block, "compliance"):
        return "creep"
    if _has(block, "storage_modulus") or _has(block, "loss_modulus"):
        frequency = _values(block, "angular_frequency") or _values(block, "frequency")
        if _decades(frequency) >= MIN_DECADES:
            return "frequency_sweep"
        amplitude = _values(block, "strain") or _values(block, "stress")
        if _decades(amplitude) >= MIN_DECADES:
            return "amplitude_sweep"
        if _span(_values(block, "temperature")) >= MIN_KELVIN:
            return "temperature_sweep"
        if _span(_values(block, "time")) > 0:
            return "time_sweep"
        return "unknown"
    if _has(block, "viscosity") and (
        _decades(_values(block, "shear_rate")) >= MIN_DECADES
        or _decades(_values(block, "stress")) >= MIN_DECADES
    ):
        return "flow_curve"
    return "unknown"


def block_temperature(block: Block, metadata_value: float | None = None) -> float | None:
    """The one temperature a block was measured at, in °C, or None if it varied."""
    temperatures = _values(block, "temperature")
    if temperatures:
        if _span(temperatures) <= CONSTANT_KELVIN:
            # To 0.1 °C: the logged value wanders by a few hundredths around the set point.
            return round(statistics.median(temperatures), 1)
        return None
    return metadata_value
