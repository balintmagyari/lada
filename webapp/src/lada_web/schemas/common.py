"""The result models every calculation returns."""

from typing import Any

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


# `/api/docs` is linked from every page, so the routes that stream a file have to say so:
# left to the default they would advertise a JSON body they never return. The view routes
# carry `response_model=TableResult` instead, which FastAPI documents but does not enforce,
# since those handlers hand back a Response themselves (JSON, or the htmx HTML partial).
FILE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The result as a file attachment.",
        "content": {
            "text/csv": {"schema": {"type": "string", "format": "binary"}},
            "text/plain": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}
