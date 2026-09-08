from pathlib import Path

import numpy as np
import pandas as pd

from lada_web.schemas.common import PlotSpec
from lada_web.services.export_service import (
    media_type,
    output_filename,
    table_result_from_df,
    write_table,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "g": [5.0, 4.0, 3.0, 2.0, 1.0]})


def test_write_table_csv_round_trips(tmp_path: Path):
    df = frame()
    path = write_table(df, tmp_path, "t", "csv")
    assert path == tmp_path / "t.csv"
    back = pd.read_csv(path)
    assert list(back.columns) == list(df.columns)
    assert np.allclose(back.to_numpy(), df.to_numpy())


def test_write_table_pgfplots_round_trips(tmp_path: Path):
    df = frame()
    path = write_table(df, tmp_path, "t", "pgfplots")
    assert path == tmp_path / "t.dat"
    assert path.read_text().splitlines()[0] == "t g"
    assert np.allclose(np.loadtxt(path, skiprows=1), df.to_numpy())


def test_format_helpers():
    assert output_filename("run", "csv") == "run.csv"
    assert output_filename("run", "pgfplots") == "run.dat"
    assert media_type("csv") == "text/csv"
    assert media_type("pgfplots") == "text/plain"


def test_table_result_caps_the_rows_it_shows():
    result = table_result_from_df(frame(), title="T", download_stem="t", max_rows=2)
    assert len(result.rows) == 2
    assert result.n_rows_total == 5
    assert result.truncated is True
    assert result.columns == ["t", "g"]
    assert result.download_stem == "t"


def test_table_result_is_not_truncated_when_it_fits():
    result = table_result_from_df(frame(), title="T", download_stem="t", max_rows=20)
    assert len(result.rows) == 5
    assert result.truncated is False


def test_nan_becomes_none_and_numpy_ints_become_int():
    df = pd.DataFrame({"n": np.array([1, 2], dtype=np.int64), "x": [np.nan, 1.5]})
    result = table_result_from_df(df, title="T", download_stem="t", max_rows=10)
    assert result.rows[0] == [1, None]
    assert isinstance(result.rows[0][0], int) and not isinstance(result.rows[0][0], bool)
    assert result.rows[1] == [2, 1.5]


def test_plot_series_carry_every_row_even_when_truncated():
    plot = PlotSpec(x="t", y=["g"])
    result = table_result_from_df(frame(), title="T", download_stem="t", max_rows=2, plot=plot)
    assert result.plot is not None
    assert result.plot.series["t"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result.plot.series["g"] == [5.0, 4.0, 3.0, 2.0, 1.0]
    assert len(result.rows) == 2


def test_meta_is_copied_onto_the_result():
    result = table_result_from_df(
        frame(), title="T", download_stem="t", max_rows=10, meta={"rows": 5}
    )
    assert result.meta == {"rows": 5}


def test_boolean_columns_become_ints_not_true_false():
    """bool is an int subclass, so it has to be converted before the int branch."""
    df = pd.DataFrame({"flag": [True, False]})
    result = table_result_from_df(df, title="T", download_stem="t", max_rows=10)
    assert [type(row[0]) for row in result.rows] == [int, int]
    assert result.rows == [[1], [0]]

    plot = PlotSpec(x="flag", y=["flag"])
    with_plot = table_result_from_df(df, title="T", download_stem="t", max_rows=10, plot=plot)
    assert with_plot.plot is not None
    assert with_plot.plot.series["flag"] == [1.0, 0.0]
