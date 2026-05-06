"""Tests for LAMMPS log file parser."""

import os
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
    with open(filepath, 'w') as f:
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
        assert all(isinstance(x, (int, float)) or hasattr(x, '__float__') for x in temps)

    def test_read_lammps_log_multiple_blocks(self, synthetic_log_file):
        """Test that multiple thermo blocks are handled."""
        thermo = read_lammps_log(synthetic_log_file)

        # Synthetic file has 2 blocks with data
        # This tests that the parser can extract from multiple blocks
        assert thermo.data.shape[0] > 0

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample_log.lammps")),
        reason="sample_log.lammps not available"
    )
    def test_read_lammps_log_with_sample_data(self, sample_log_file):
        """Test with actual sample log file if provided."""
        thermo = read_lammps_log(sample_log_file)

        assert thermo is not None
        assert len(thermo.columns) > 0
        assert thermo.data.shape[0] > 0
