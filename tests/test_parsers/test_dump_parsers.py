"""Tests for LAMMPS dump file parsers."""

import os
import pytest
import numpy as np
from lada import iter_dump_frames, dump_frames, read_dump


# Get the directory containing test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def create_minimal_dump_file(filepath, num_atoms=10, num_timesteps=2):
    """Create a minimal valid LAMMPS dump file for testing.

    Parameters
    ----------
    filepath : str
        Path where the dump file will be created
    num_atoms : int
        Number of atoms per timestep
    num_timesteps : int
        Number of timesteps to include
    """
    with open(filepath, 'w') as f:
        for ts in range(num_timesteps):
            timestep = ts * 1000  # Timestep values
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")

            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{num_atoms}\n")

            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")

            f.write("ITEM: ATOMS id type x y z\n")
            for atom_id in range(1, num_atoms + 1):
                atom_type = (atom_id % 2) + 1  # Alternate between type 1 and 2
                x = np.random.random() * 10
                y = np.random.random() * 10
                z = np.random.random() * 10
                f.write(f"{atom_id} {atom_type} {x:.4f} {y:.4f} {z:.4f}\n")


@pytest.fixture
def synthetic_dump_file(tmp_path):
    """Fixture providing a synthetic dump file for testing."""
    dump_file = tmp_path / "test_synthetic.dump"
    create_minimal_dump_file(str(dump_file), num_atoms=10, num_timesteps=2)
    return str(dump_file)


@pytest.fixture
def sample_dump_file():
    """Fixture providing the sample dump file if it exists."""
    sample_path = os.path.join(TEST_DATA_DIR, "sample_dump.dump")
    if os.path.exists(sample_path):
        return sample_path
    return None


class TestDumpFrames:
    """Tests for dump_frames() function (streaming parser)."""

    def test_dump_frames_iteration_synthetic(self, synthetic_dump_file):
        """Test that dump_frames can iterate over frames (synthetic data)."""
        frame_count = 0
        for frame in dump_frames(synthetic_dump_file):
            frame_count += 1
            assert frame.metadata is not None
            assert frame.columns is not None
            assert frame.data is not None
            assert len(frame.data) > 0

        assert frame_count == 2, "Expected 2 frames in synthetic dump"

    def test_dump_frames_metadata(self, synthetic_dump_file):
        """Test that metadata is properly extracted."""
        for frame in dump_frames(synthetic_dump_file):
            # Check required metadata keys
            assert "TIMESTEP" in frame.metadata
            assert "NUMBER OF ATOMS" in frame.metadata
            assert "BOX BOUNDS pp pp pp" in frame.metadata

            # Check types
            assert isinstance(frame.metadata["TIMESTEP"], (int, np.integer))
            assert isinstance(frame.metadata["NUMBER OF ATOMS"], (int, np.integer))
            break  # Just test first frame

    def test_dump_frames_column_access(self, synthetic_dump_file):
        """Test column access methods."""
        for frame in dump_frames(synthetic_dump_file):
            # Test get_column
            ids = frame.get_column("id")
            assert len(ids) == 10
            assert all(isinstance(x, (int, float, np.integer, np.floating)) for x in ids)

            # Test column_index
            col_idx = frame.column_index("type")
            assert isinstance(col_idx, int)

            # Test get_column_or with existing column
            x_coords = frame.get_column_or("x")
            assert x_coords is not None

            # Test get_column_or with non-existing column
            nonexistent = frame.get_column_or("nonexistent", default=None)
            assert nonexistent is None
            break  # Just test first frame

    def test_dump_frames_to_pandas(self, synthetic_dump_file):
        """Test conversion to pandas DataFrame."""
        for frame in dump_frames(synthetic_dump_file):
            df = frame.to_pandas()
            assert df is not None
            assert len(df) == 10
            assert "id" in df.columns
            assert "type" in df.columns
            assert "x" in df.columns
            break  # Just test first frame

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample_dump.dump")),
        reason="sample_dump.dump not available"
    )
    def test_dump_frames_with_sample_data(self, sample_dump_file):
        """Test with actual sample dump file if provided."""
        frame_count = 0
        for frame in dump_frames(sample_dump_file):
            frame_count += 1
            assert frame.data is not None
            assert len(frame.data) > 0

        assert frame_count > 0, "Sample dump file should have at least one frame"


class TestReadDump:
    """Tests for read_dump() function (bulk parser)."""

    def test_read_dump_synthetic(self, synthetic_dump_file):
        """Test bulk reading of dump file (synthetic data)."""
        df = read_dump(synthetic_dump_file, timestep_col="Timestep")

        assert df is not None
        assert len(df) > 0
        assert "Timestep" in df.columns
        assert "id" in df.columns or "type" in df.columns

    def test_read_dump_returns_dataframe(self, synthetic_dump_file):
        """Test that read_dump returns a proper DataFrame."""
        df = read_dump(synthetic_dump_file)

        # Check it's a DataFrame-like object
        assert hasattr(df, 'shape')
        assert hasattr(df, 'columns')
        assert len(df.shape) == 2

    def test_read_dump_timestep_column(self, synthetic_dump_file):
        """Test that timestep column is properly added."""
        df = read_dump(synthetic_dump_file, timestep_col="ts")

        assert "ts" in df.columns
        # Synthetic file has timesteps 0 and 1000
        assert 0 in df["ts"].values
        assert 1000 in df["ts"].values

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(TEST_DATA_DIR, "sample_dump.dump")),
        reason="sample_dump.dump not available"
    )
    def test_read_dump_with_sample_data(self, sample_dump_file):
        """Test bulk reading with actual sample data if provided."""
        df = read_dump(sample_dump_file)

        assert df is not None
        assert len(df) > 0
        # Sample should have at least column data
        assert df.shape[1] > 0
