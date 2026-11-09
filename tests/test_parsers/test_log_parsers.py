"""Tests for LAMMPS log file parser."""

import os

import numpy as np
import pytest

from lada import read_lammps_log

# Get the directory containing test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def create_minimal_log_file(filepath):
    """Create a minimal valid LAMMPS log file for testing.

    Parameters
    ----------
    filepath : str
        Path where the log file will be created
    """
    with open(filepath, "w") as f:
        # Pre-run message
        f.write("LAMMPS (2 Aug 2023 - Update 3)\n")
        f.write("Using OpenMP\n")

        # First thermo block
        f.write("Per MPI rank memory allocation (min/avg/max) = 4.195 | 4.195 | 4.195 Mbytes\n")
        f.write("Step Temp E_pair E_mol TotEng Press\n")
        f.write("0 300.0 -4.1234 0.0 -2.5678 1000.5\n")
        f.write("100 301.2 -4.1340 0.0 -2.5690 1001.2\n")
        f.write("200 302.1 -4.1425 0.0 -2.5750 1002.1\n")
        f.write("Loop time of 0.123 on 1 procs for 200 steps\n")

        # Second thermo block
        f.write("Per MPI rank memory allocation (min/avg/max) = 4.195 | 4.195 | 4.195 Mbytes\n")
        f.write("Step Temp E_pair E_mol TotEng Press\n")
        f.write("200 302.1 -4.1425 0.0 -2.5750 1002.1\n")
        f.write("300 303.0 -4.1500 0.0 -2.5800 1003.0\n")
        f.write("400 304.2 -4.1620 0.0 -2.5900 1004.2\n")
        f.write("Loop time of 0.115 on 1 procs for 200 steps\n")


@pytest.fixture
def synthetic_log_file(tmp_path):
    """Fixture providing a synthetic log file for testing."""
    log_file = tmp_path / "test_synthetic.log"
    create_minimal_log_file(str(log_file))
    return str(log_file)


@pytest.fixture
def sample_log_file():
    """Fixture providing the sample log file if it exists."""
    sample_path = os.path.join(TEST_DATA_DIR, "sample_log.lammps")
    if os.path.exists(sample_path):
        return sample_path
    return None


class TestReadLammpsLog:
    """Tests for read_lammps_log() function."""

    def test_read_lammps_log_synthetic(self, synthetic_log_file):
        """Test reading synthetic log file."""
        thermo = read_lammps_log(synthetic_log_file)

        assert thermo is not None
        assert thermo.columns is not None
        assert thermo.data is not None

    def test_read_lammps_log_has_columns(self, synthetic_log_file):
        """Test that column names are properly extracted."""
        thermo = read_lammps_log(synthetic_log_file)

        assert "Step" in thermo.columns
        assert "Temp" in thermo.columns
        assert "E_pair" in thermo.columns
        assert "TotEng" in thermo.columns

    def test_read_lammps_log_get_property(self, synthetic_log_file):
        """Test retrieving individual properties."""
        thermo = read_lammps_log(synthetic_log_file)

        # Test getting a property
        temps = thermo.get("Temp")
        assert temps is not None
        assert len(temps) > 0

        # Test getting another property
        energies = thermo.get("E_pair")
        assert energies is not None
        assert len(energies) > 0

    def test_read_lammps_log_invalid_property(self, synthetic_log_file):
        """Test that requesting non-existent property raises error."""
        thermo = read_lammps_log(synthetic_log_file)

        with pytest.raises(ValueError):
            thermo.get("NonExistentProperty")

    def test_read_lammps_log_to_pandas(self, synthetic_log_file):
        """Test conversion to pandas DataFrame."""
        thermo = read_lammps_log(synthetic_log_file)

        df = thermo.to_pandas()
        assert df is not None
        assert len(df) > 0
        assert "Step" in df.columns
        assert "Temp" in df.columns

    def test_read_lammps_log_data_types(self, synthetic_log_file):
        """Test that data is properly typed as floats/ints."""
        thermo = read_lammps_log(synthetic_log_file)

        temps = thermo.get("Temp")
        # Should be numeric
        assert all(isinstance(x, (int, float)) or hasattr(x, "__float__") for x in temps)

    def test_read_lammps_log_multiple_blocks(self, synthetic_log_file):
        """Test that multiple thermo blocks are handled."""
        thermo = read_lammps_log(synthetic_log_file)

        # Synthetic file has 2 blocks with data
        # This tests that the parser can extract from multiple blocks
        assert thermo.data.shape[0] > 0

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample_log.lammps")),
        reason="sample_log.lammps not available",
    )
    def test_read_lammps_log_with_sample_data(self, sample_log_file):
        """Test with actual sample log file if provided."""
        thermo = read_lammps_log(sample_log_file)

        assert thermo is not None
        assert len(thermo.columns) > 0
        assert thermo.data.shape[0] > 0


TWO_RUNS_DIFFERENT_COLUMNS = """LAMMPS
Per MPI rank memory allocation = 2.5 Mbytes
Step Temp Press
0 1.0 0.5
100 1.1 0.6
Loop time of 0.1 on 1 procs
Per MPI rank memory allocation = 2.5 Mbytes
Step PotEng KinEng
100 -5.0 1.6
200 -4.0 1.7
Loop time of 0.1 on 1 procs
"""


class TestEachRunUsesItsOwnHeader:
    """Earlier runs were labelled with the last run's header (D2)."""

    def test_columns_are_the_union_in_order_of_appearance(self, tmp_path):
        path = tmp_path / "two.log"
        path.write_text(TWO_RUNS_DIFFERENT_COLUMNS)
        thermo = read_lammps_log(str(path))
        assert thermo.columns == ["Step", "Temp", "Press", "PotEng", "KinEng"]

    def test_values_sit_under_their_own_column_and_missing_ones_are_nan(self, tmp_path):
        path = tmp_path / "two.log"
        path.write_text(TWO_RUNS_DIFFERENT_COLUMNS)
        thermo = read_lammps_log(str(path))
        np.testing.assert_array_equal(thermo.get("Step"), [0, 100, 100, 200])
        np.testing.assert_array_equal(thermo.get("Temp"), [1.0, 1.1, np.nan, np.nan])
        np.testing.assert_array_equal(thermo.get("PotEng"), [np.nan, np.nan, -5.0, -4.0])
        np.testing.assert_array_equal(thermo.run_index, [0, 0, 1, 1])
        assert list(thermo.to_pandas().columns) == thermo.columns

    def test_runs_with_different_column_counts(self, tmp_path):
        path = tmp_path / "diff.log"
        path.write_text(
            TWO_RUNS_DIFFERENT_COLUMNS.replace("PotEng KinEng", "PotEng")
            .replace("-5.0 1.6", "-5.0")
            .replace("-4.0 1.7", "-4.0")
        )
        thermo = read_lammps_log(str(path))
        assert thermo.data.shape == (4, 4)
        np.testing.assert_array_equal(thermo.get("PotEng"), [np.nan, np.nan, -5.0, -4.0])

    def test_same_header_in_every_run_is_unchanged(self, synthetic_log_file):
        thermo = read_lammps_log(synthetic_log_file)
        assert thermo.columns == ["Step", "Temp", "E_pair", "E_mol", "TotEng", "Press"]
        assert thermo.data.shape == (6, 6)
        assert not np.isnan(thermo.data).any()
        np.testing.assert_array_equal(thermo.run_index, [0, 0, 0, 1, 1, 1])


class TestNonNumericLinesInsideABlock:
    """One WARNING line inside a thermo block made the whole log unreadable (D3)."""

    @pytest.mark.parametrize(
        "intruder",
        [
            "WARNING: Bond/angle/dihedral extent > half of periodic box length (domain.cpp:936)",
            "SHAKE stats (type/ave/delta/count) on step 100",
            "Loop time 0.1",  # a variant without "of"
            "100 1.1",  # a truncated row
        ],
    )
    def test_the_line_is_skipped(self, tmp_path, intruder):
        path = tmp_path / "warn.log"
        path.write_text(
            "LAMMPS\nPer MPI rank memory allocation = 2.5 Mbytes\n"
            "   Step          Temp          Press\n"
            "         0   1.0   0.5\n"
            f"{intruder}\n"
            "       100   1.1   0.6\n"
            "Loop time of 0.1 on 1 procs\n"
        )
        thermo = read_lammps_log(str(path))
        assert thermo.columns == ["Step", "Temp", "Press"]
        np.testing.assert_array_equal(thermo.data, [[0, 1.0, 0.5], [100, 1.1, 0.6]])

    def test_a_warning_before_the_header_is_not_the_header(self, tmp_path):
        path = tmp_path / "warn.log"
        path.write_text(
            "Per MPI rank memory allocation = 2.5 Mbytes\n"
            "WARNING: something\n"
            "Step Temp\n0 1.0\nLoop time of 0.1 on 1 procs\n"
        )
        thermo = read_lammps_log(str(path))
        assert thermo.columns == ["Step", "Temp"]
        np.testing.assert_array_equal(thermo.data, [[0, 1.0]])
