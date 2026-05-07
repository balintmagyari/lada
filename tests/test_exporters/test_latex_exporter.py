"""Tests for write_pgfplots_table."""

import pytest
import numpy as np
import pandas as pd
from lada.exporters import write_pgfplots_table


def read_lines(path):
    """Return non-empty lines from a file as a list of strings."""
    return [l.rstrip('\n') for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class TestDataFrameInput:

    def test_header_from_column_names(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        assert read_lines(tmp_path / 'out.dat')[0] == 'x y'

    def test_values_are_written(self, tmp_path):
        df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        lines = read_lines(tmp_path / 'out.dat')
        assert lines[1] == '1 3'
        assert lines[2] == '2 4'

    def test_row_count_matches(self, tmp_path):
        df = pd.DataFrame({'a': range(10), 'b': range(10)})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        lines = read_lines(tmp_path / 'out.dat')
        assert len(lines) == 11  # 1 header + 10 data rows

    def test_columns_arg_overrides_df_column_names(self, tmp_path):
        df = pd.DataFrame({'old_x': [1.0], 'old_y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat', columns=['new_x', 'new_y'])
        assert read_lines(tmp_path / 'out.dat')[0] == 'new_x new_y'


class TestNumpyInput:

    def test_2d_array_with_columns_writes_header(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        write_pgfplots_table(arr, tmp_path / 'out.dat', columns=['a', 'b'])
        assert read_lines(tmp_path / 'out.dat')[0] == 'a b'

    def test_2d_array_without_columns_has_no_header(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        write_pgfplots_table(arr, tmp_path / 'out.dat')
        lines = read_lines(tmp_path / 'out.dat')
        # Without columns there should be exactly 2 data lines, no header
        assert len(lines) == 2
        assert lines[0] == '1 2'

    def test_1d_array_written_as_single_column(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        write_pgfplots_table(arr, tmp_path / 'out.dat', columns=['val'])
        lines = read_lines(tmp_path / 'out.dat')
        assert lines[0] == 'val'
        assert lines[1] == '1'
        assert lines[2] == '2'
        assert lines[3] == '3'

    def test_1d_array_without_columns_has_no_header(self, tmp_path):
        arr = np.array([1.0, 2.0])
        write_pgfplots_table(arr, tmp_path / 'out.dat')
        lines = read_lines(tmp_path / 'out.dat')
        assert len(lines) == 2


class TestDictInput:

    def test_keys_become_header(self, tmp_path):
        write_pgfplots_table({'t': [0.0, 1.0], 'G': [9.0, 4.5]}, tmp_path / 'out.dat')
        assert read_lines(tmp_path / 'out.dat')[0] == 't G'

    def test_values_are_written(self, tmp_path):
        write_pgfplots_table({'t': [0.0], 'G': [9.0]}, tmp_path / 'out.dat')
        lines = read_lines(tmp_path / 'out.dat')
        assert lines[1] == '0 9'

    def test_columns_arg_overrides_dict_keys(self, tmp_path):
        write_pgfplots_table({'a': [1.0], 'b': [2.0]}, tmp_path / 'out.dat', columns=['x', 'y'])
        assert read_lines(tmp_path / 'out.dat')[0] == 'x y'

    def test_unequal_array_lengths_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            write_pgfplots_table({'a': [1, 2, 3], 'b': [1, 2]}, tmp_path / 'out.dat')


# ---------------------------------------------------------------------------
# Delimiter
# ---------------------------------------------------------------------------

class TestDelimiter:

    def test_default_space_delimiter(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        assert ' ' in read_lines(tmp_path / 'out.dat')[0]

    def test_comma_delimiter_in_header(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.csv', delimiter=',')
        assert read_lines(tmp_path / 'out.csv')[0] == 'x,y'

    def test_comma_delimiter_in_data(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.csv', delimiter=',')
        assert read_lines(tmp_path / 'out.csv')[1] == '1,2'

    def test_tab_delimiter_in_header(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        write_pgfplots_table(df, tmp_path / 'out.tsv', delimiter='\t')
        assert read_lines(tmp_path / 'out.tsv')[0] == 'x\ty'


# ---------------------------------------------------------------------------
# Number format
# ---------------------------------------------------------------------------

class TestFormat:

    def test_default_fmt_suppresses_trailing_zeros(self, tmp_path):
        df = pd.DataFrame({'x': [1.5]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        assert read_lines(tmp_path / 'out.dat')[1] == '1.5'

    def test_default_fmt_six_significant_figures(self, tmp_path):
        df = pd.DataFrame({'x': [1.23456789]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        assert read_lines(tmp_path / 'out.dat')[1] == '1.23457'

    def test_custom_fmt_fixed_decimal(self, tmp_path):
        df = pd.DataFrame({'x': [1.5]})
        write_pgfplots_table(df, tmp_path / 'out.dat', fmt='%.2f')
        assert read_lines(tmp_path / 'out.dat')[1] == '1.50'

    def test_custom_fmt_scientific(self, tmp_path):
        df = pd.DataFrame({'x': [1500.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat', fmt='%.2e')
        assert read_lines(tmp_path / 'out.dat')[1] == '1.50e+03'


# ---------------------------------------------------------------------------
# Comment block
# ---------------------------------------------------------------------------

class TestComment:

    def test_single_line_comment_prefixed_with_percent(self, tmp_path):
        df = pd.DataFrame({'x': [1.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat', comment='run A')
        first = path_first_line(tmp_path / 'out.dat')
        assert first == '% run A'

    def test_multiline_comment_each_line_prefixed(self, tmp_path):
        df = pd.DataFrame({'x': [1.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat', comment='line1\nline2')
        raw_lines = (tmp_path / 'out.dat').read_text().splitlines()
        assert raw_lines[0] == '% line1'
        assert raw_lines[1] == '% line2'

    def test_comment_appears_before_header(self, tmp_path):
        df = pd.DataFrame({'x': [1.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat', comment='meta')
        raw_lines = (tmp_path / 'out.dat').read_text().splitlines()
        assert raw_lines[0].startswith('%')
        assert raw_lines[1] == 'x'

    def test_no_comment_no_percent_lines(self, tmp_path):
        df = pd.DataFrame({'x': [1.0]})
        write_pgfplots_table(df, tmp_path / 'out.dat')
        assert not any(l.startswith('%') for l in (tmp_path / 'out.dat').read_text().splitlines())


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestFileCreation:

    def test_file_is_created(self, tmp_path):
        df = pd.DataFrame({'x': [1.0]})
        out = tmp_path / 'result.dat'
        write_pgfplots_table(df, out)
        assert out.exists()

    def test_file_is_overwritten_on_second_call(self, tmp_path):
        out = tmp_path / 'result.dat'
        write_pgfplots_table(pd.DataFrame({'x': [1.0, 2.0]}), out)
        write_pgfplots_table(pd.DataFrame({'x': [99.0]}), out)
        lines = read_lines(out)
        assert len(lines) == 2  # header + 1 data row, not 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_type_error_for_string_input(self, tmp_path):
        with pytest.raises(TypeError, match="pd.DataFrame"):
            write_pgfplots_table("not_data", tmp_path / 'out.dat')

    def test_type_error_for_list_input(self, tmp_path):
        with pytest.raises(TypeError):
            write_pgfplots_table([1, 2, 3], tmp_path / 'out.dat')

    def test_value_error_columns_too_short(self, tmp_path):
        arr = np.ones((3, 2))
        with pytest.raises(ValueError, match="2 column"):
            write_pgfplots_table(arr, tmp_path / 'out.dat', columns=['only_one'])

    def test_value_error_columns_too_long(self, tmp_path):
        arr = np.ones((3, 2))
        with pytest.raises(ValueError, match="2 column"):
            write_pgfplots_table(arr, tmp_path / 'out.dat', columns=['a', 'b', 'c'])

    def test_value_error_columns_override_wrong_length_on_dataframe(self, tmp_path):
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        with pytest.raises(ValueError):
            write_pgfplots_table(df, tmp_path / 'out.dat', columns=['only_one'])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def path_first_line(path):
    return path.read_text().splitlines()[0]
