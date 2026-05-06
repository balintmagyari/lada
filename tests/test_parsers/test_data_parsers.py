"""Tests for LAMMPS data file parser."""

import os
import pytest
from lada import read_data_file, read_lammps_acf


# Get the directory containing test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def create_minimal_data_file(filepath, num_atoms=10, num_bonds=5):
    """Create a minimal valid LAMMPS data file for testing.

    Parameters
    ----------
    filepath : str
        Path where the data file will be created
    num_atoms : int
        Number of atoms to include
    num_bonds : int
        Number of bonds to include
    """
    with open(filepath, 'w') as f:
        f.write("Test data file\n\n")

        # Header
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{num_bonds} bonds\n")
        f.write("0 angles\n")
        f.write("0 dihedrals\n")
        f.write("0 impropers\n\n")

        f.write("2 atom types\n")
        f.write("1 bond types\n\n")

        # Box dimensions
        f.write("0.0 10.0 xlo xhi\n")
        f.write("0.0 10.0 ylo yhi\n")
        f.write("0.0 10.0 zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write("1 1.0\n")
        f.write("2 1.0\n\n")

        # Atoms section
        f.write("Atoms # atomic\n\n")
        for i in range(1, num_atoms + 1):
            atom_type = (i % 2) + 1
            x = (i * 1.0) % 10
            y = (i * 0.5) % 10
            z = (i * 0.3) % 10
            f.write(f"{i} {atom_type} {x:.4f} {y:.4f} {z:.4f}\n")

        # Bonds section
        f.write("\nBonds\n\n")
        for i in range(1, num_bonds + 1):
            atom1 = i
            atom2 = i + 1 if i < num_atoms else i - 1
            f.write(f"{i} 1 {atom1} {atom2}\n")


@pytest.fixture
def synthetic_data_file(tmp_path):
    """Fixture providing a synthetic data file for testing."""
    data_file = tmp_path / "test_synthetic.data"
    create_minimal_data_file(str(data_file), num_atoms=10, num_bonds=5)
    return str(data_file)


@pytest.fixture
def sample_data_file():
    """Fixture providing the sample data file if it exists."""
    sample_path = os.path.join(TEST_DATA_DIR, "sample.data")
    if os.path.exists(sample_path):
        return sample_path
    return None


class TestReadDataFile:
    """Tests for read_data_file() function."""

    def test_read_data_file_synthetic(self, synthetic_data_file):
        """Test reading synthetic data file."""
        lammps_data = read_data_file(synthetic_data_file)

        assert lammps_data is not None

    def test_read_data_file_atoms_section(self, synthetic_data_file):
        """Test that Atoms section is properly read."""
        lammps_data = read_data_file(synthetic_data_file)

        atoms = lammps_data.get("Atoms")
        assert atoms is not None
        assert len(atoms) == 10  # Synthetic file has 10 atoms

    def test_read_data_file_bonds_section(self, synthetic_data_file):
        """Test that Bonds section is properly read."""
        lammps_data = read_data_file(synthetic_data_file)

        bonds = lammps_data.get("Bonds")
        assert bonds is not None
        assert len(bonds) == 5  # Synthetic file has 5 bonds

    def test_read_data_file_to_pandas(self, synthetic_data_file):
        """Test conversion to pandas DataFrame."""
        lammps_data = read_data_file(synthetic_data_file)

        # Convert Atoms section to DataFrame
        df_atoms = lammps_data.to_pandas(section="Atoms")
        assert df_atoms is not None
        assert len(df_atoms) == 10

        # Convert Bonds section to DataFrame
        df_bonds = lammps_data.to_pandas(section="Bonds")
        assert df_bonds is not None
        assert len(df_bonds) == 5

    def test_read_data_file_atom_types(self, synthetic_data_file):
        """Test that atom type columns are detected."""
        lammps_data = read_data_file(synthetic_data_file)

        atoms = lammps_data.get("Atoms")
        # Should have at least: id, type, x, y, z columns
        assert atoms.shape[1] >= 5

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample.data")),
        reason="sample.data not available"
    )
    def test_read_data_file_with_sample_data(self, sample_data_file):
        """Test with actual sample data file if provided."""
        lammps_data = read_data_file(sample_data_file)

        assert lammps_data is not None
        atoms = lammps_data.get("Atoms")
        assert atoms is not None
        assert len(atoms) > 0


class TestReadLammpsACF:
    """Tests for read_lammps_acf() function."""

    def create_minimal_acf_file(self, filepath):
        """Create a minimal valid ACF file for testing.

        Format from LAMMPS fix ave/correlate/long command:
        Line 1: acf_x,acf_y,acf_z (comma-separated column names)
        Then: # Timestep: N blocks with data rows
        lag_time acf_value1 acf_value2 ...
        """
        with open(filepath, 'w') as f:
            # LINE 1: Column names (comma-separated) — REQUIRED by parser
            f.write("acf_x,acf_y,acf_z\n")

            # First timestep block (reference block at t=0)
            f.write("# Timestep: 0\n")
            f.write("0.0 1.0 0.95 0.92\n")
            f.write("1.0 0.98 0.90 0.88\n")
            f.write("2.0 0.95 0.85 0.82\n")
            f.write("3.0 0.90 0.78 0.75\n")
            f.write("4.0 0.85 0.70 0.68\n")

            # Second timestep block (required for parser)
            f.write("# Timestep: 100\n")
            f.write("0.0 1.0 0.94 0.91\n")
            f.write("1.0 0.97 0.89 0.87\n")
            f.write("2.0 0.94 0.84 0.81\n")
            f.write("3.0 0.89 0.77 0.74\n")
            f.write("4.0 0.84 0.69 0.67\n")

    @pytest.fixture
    def synthetic_acf_file(self, tmp_path):
        """Fixture providing a synthetic ACF file for testing."""
        acf_file = tmp_path / "test_synthetic.txt"
        self.create_minimal_acf_file(str(acf_file))
        return str(acf_file)

    def test_read_lammps_acf_synthetic(self, tmp_path):
        """Test reading synthetic ACF file."""
        acf_file = tmp_path / "test.txt"
        self.create_minimal_acf_file(str(acf_file))

        df = read_lammps_acf(str(acf_file))

        assert df is not None
        assert len(df) > 0

    def test_read_lammps_acf_columns(self, tmp_path):
        """Test that ACF columns are properly read."""
        acf_file = tmp_path / "test.txt"
        self.create_minimal_acf_file(str(acf_file))

        df = read_lammps_acf(str(acf_file))

        # Should have numeric columns
        assert len(df.columns) > 0

    def test_read_lammps_acf_returns_dataframe(self, tmp_path):
        """Test that read_lammps_acf returns a DataFrame."""
        acf_file = tmp_path / "test.txt"
        self.create_minimal_acf_file(str(acf_file))

        df = read_lammps_acf(str(acf_file))

        # Check it's a DataFrame-like object
        assert hasattr(df, 'shape')
        assert hasattr(df, 'columns')

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample_acf.txt")),
        reason="sample_acf.txt not available"
    )
    def test_read_lammps_acf_with_sample_data(self):
        """Test with actual sample ACF file if provided."""
        sample_path = os.path.join(TEST_DATA_DIR, "sample_acf.txt")
        df = read_lammps_acf(sample_path)

        assert df is not None
        assert len(df) > 0
