"""What an imported file becomes: a Dataset of Blocks of Columns.

A *block* is one table of measured points: one TRIOS step, one RheoCompass interval, one
spreadsheet sheet, or the whole of a plain text file. Every column keeps its original header
and the unit it arrived in, and carries the quantity it was recognised as (or None) with its
values converted to that quantity's canonical unit.
"""

from typing import Literal

from pydantic import BaseModel, Field

Format = Literal["trios", "anton_paar", "excel", "delimited"]
TestType = Literal[
    "frequency_sweep",
    "amplitude_sweep",
    "temperature_sweep",
    "time_sweep",
    "flow_curve",
    "creep",
    "stress_relaxation",
    "unknown",
]

FORMAT_LABELS: dict[str, str] = {
    "trios": "TA Instruments TRIOS export",
    "anton_paar": "Anton Paar RheoCompass export",
    "excel": "Spreadsheet",
    "delimited": "Text table",
}

TEST_TYPE_LABELS: dict[str, str] = {
    "frequency_sweep": "Frequency sweep",
    "amplitude_sweep": "Amplitude sweep",
    "temperature_sweep": "Temperature sweep",
    "time_sweep": "Time sweep",
    "flow_curve": "Flow curve",
    "creep": "Creep",
    "stress_relaxation": "Stress relaxation",
    "unknown": "Not recognised",
}


class Column(BaseModel):
    key: str  # unique within its block
    label: str  # the header as the file wrote it
    quantity: str | None = None  # a key of quantities.QUANTITY_BY_KEY
    unit: str = ""  # the unit `values` are in: the quantity's canonical unit when recognised
    source_unit: str = ""  # the unit the file gave, as written
    derived: bool = False
    is_index: bool = False
    values: list[float | None]


class Block(BaseModel):
    id: str
    name: str
    test_type: TestType = "unknown"
    temperature: float | None = None  # °C, when the block was measured at one temperature
    n_points: int
    metadata: dict[str, str] = Field(default_factory=dict)
    columns: list[Column]


class Dataset(BaseModel):
    id: str
    filename: str
    format: Format
    format_label: str
    metadata: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    blocks: list[Block]


class ParseError(BaseModel):
    filename: str
    message: str


class ParseResponse(BaseModel):
    datasets: list[Dataset]
    errors: list[ParseError] = Field(default_factory=list)
