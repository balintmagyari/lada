"""Turn a DataFrame into a downloadable file or into the on-screen table model."""

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from lada.exporters import write_pgfplots_table

from ..schemas.common import Cell, PlotSpec, TableResult

Format = Literal["csv", "pgfplots"]

# fmt -> (delimiter for write_pgfplots_table, file suffix, media type)
_FORMATS: dict[str, tuple[str, str, str]] = {
    "csv": (",", ".csv", "text/csv"),
    "pgfplots": (" ", ".dat", "text/plain"),
}

# `write_pgfplots_table` defaults to '%.6g', which silently rounds a download to six
# significant figures: five Rg² values differing in the 8th digit all become '10', and a
# timestep of 10000000 becomes '1e+07'. 17 significant digits is the shortest format that
# round-trips every float64 exactly, so a downloaded file carries the same numbers as the
# table and the JSON. A researcher can round the file later; lost digits cannot be recovered.
FLOAT_FORMAT = "%.17g"


def media_type(fmt: Format) -> str:
    return _FORMATS[fmt][2]


def output_filename(stem: str, fmt: Format) -> str:
    return f"{stem}{_FORMATS[fmt][1]}"


def write_table(df: pd.DataFrame, directory: Path, stem: str, fmt: Format) -> Path:
    """One code path for both formats: write_pgfplots_table with ',' (csv) or ' ' (pgfplots)."""
    path = directory / output_filename(stem, fmt)
    write_pgfplots_table(df, str(path), delimiter=_FORMATS[fmt][0], fmt=FLOAT_FORMAT)
    return path


def _cell(value: object) -> Cell:
    # bool is a subclass of int, so it has to be caught before the int branch below,
    # or a boolean column would leave as True/False instead of 1/0.
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if value is None or isinstance(value, int | str):
        return value
    return str(value)


def _series(df: pd.DataFrame, column: str) -> list[float | None]:
    return [_cell(v) for v in df[column].to_numpy()]  # type: ignore[misc]


def table_result_from_df(
    df: pd.DataFrame,
    *,
    title: str,
    download_stem: str,
    max_rows: int,
    plot: PlotSpec | None = None,
    meta: dict[str, str | int | float] | None = None,
) -> TableResult:
    """Cap the on-screen rows at max_rows; plot series always carry every row."""
    shown = df.iloc[:max_rows]
    rows = [[_cell(v) for v in row] for row in shown.itertuples(index=False, name=None)]
    if plot is not None:
        plot = plot.model_copy(
            update={"series": {name: _series(df, name) for name in [plot.x, *plot.y]}}
        )
    return TableResult(
        title=title,
        columns=[str(c) for c in df.columns],
        rows=rows,
        n_rows_total=len(df),
        truncated=len(df) > max_rows,
        meta=dict(meta or {}),
        plot=plot,
        download_stem=download_stem,
    )
