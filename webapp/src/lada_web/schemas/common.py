"""The result models every calculation returns."""

from pydantic import BaseModel, Field

Cell = float | int | str | None


class PlotSpec(BaseModel):
    """Which result columns to plot. `series` holds the full-length data for x and every y."""

    x: str
    y: list[str]
    x_label: str = ""
    y_label: str = ""
    x_log: bool = False
    y_log: bool = False
    series: dict[str, list[float | None]] = Field(default_factory=dict)


class TableResult(BaseModel):
    """A computed table: the rows shown on screen (capped) plus metadata and an optional plot."""

    title: str
    columns: list[str]
    rows: list[list[Cell]]
    n_rows_total: int
    truncated: bool = False
    meta: dict[str, str | int | float] = Field(default_factory=dict)
    plot: PlotSpec | None = None
    download_stem: str
