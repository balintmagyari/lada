"""The committed generated fixtures must match what their generator scripts produce."""

import importlib.util
from pathlib import Path

import numpy as np

from lada.analysis import calculate_segment_acf_from_trajectory
from lada.parsers import read_lammps_acf

DATA = Path(__file__).resolve().parent / "data"


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, DATA / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_generator_reproduces_committed_file(tmp_path):
    out = tmp_path / "regenerated.npz"
    _load("generate_sample_trajectory").main(out)
    committed = np.load(DATA / "sample_trajectory.npz")["coords"]
    assert committed.shape == (32, 32, 3)
    assert np.array_equal(np.load(out)["coords"], committed)


def test_stress_acf_generator_reproduces_committed_file(tmp_path):
    out = tmp_path / "regenerated.txt"
    _load("generate_sample_stress_acf").main(out)
    assert out.read_text() == (DATA / "sample_stress_acf.txt").read_text()


def test_stress_acf_fixture_has_the_schema_calc_stress_relaxation_needs():
    df = read_lammps_acf(DATA / "sample_stress_acf.txt")
    assert list(df.columns) == [
        "lag_time",
        "ACF_Sxy",
        "ACF_Sxz",
        "ACF_Syz",
        "ACF_Nxy",
        "ACF_Nxz",
        "ACF_Nyz",
        "timestep",
    ]
    assert len(df) == 100 and df["timestep"].iloc[0] == 1000
    assert df["lag_time"].iloc[1] == 0.5


def test_trajectory_fixture_works_with_dynamics_functions():
    pairs = np.array([[0, 7], [8, 15], [16, 23], [24, 31]])
    acf = calculate_segment_acf_from_trajectory(
        str(DATA / "sample_trajectory.npz"), pairs, time_per_frame=1.0
    )
    assert acf.shape == (32, 2) and acf[0, 1] == 1.0
