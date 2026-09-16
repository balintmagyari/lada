"""Unit tests for `services/merge.py`: averaging across files, and merging across regimes."""

import numpy as np
import pandas as pd
import pytest

from lada_web.services import merge


def test_average_per_timestep_single_file_is_unchanged():
    df = pd.DataFrame({"timestep": [0, 1000], "rg_sq": [1.0, 2.0]})
    out = merge.average_per_timestep([df], "rg_sq")
    assert out is df


def test_average_per_timestep_averages_matching_grids():
    a = pd.DataFrame({"timestep": [0, 1000], "rg_sq": [1.0, 3.0]})
    b = pd.DataFrame({"timestep": [0, 1000], "rg_sq": [3.0, 5.0]})
    out = merge.average_per_timestep([a, b], "rg_sq")
    assert out["timestep"].tolist() == [0, 1000]
    assert out["rg_sq"].tolist() == [2.0, 4.0]


def test_average_per_timestep_ignores_row_order():
    a = pd.DataFrame({"timestep": [1000, 0], "rg_sq": [3.0, 1.0]})
    b = pd.DataFrame({"timestep": [0, 1000], "rg_sq": [3.0, 5.0]})
    out = merge.average_per_timestep([a, b], "rg_sq")
    assert out["timestep"].tolist() == [0, 1000]
    assert out["rg_sq"].tolist() == [2.0, 4.0]


def test_average_per_timestep_rejects_mismatched_timesteps():
    a = pd.DataFrame({"timestep": [0, 1000], "rg_sq": [1.0, 2.0]})
    b = pd.DataFrame({"timestep": [0, 2000], "rg_sq": [1.0, 2.0]})
    with pytest.raises(ValueError, match="different timesteps"):
        merge.average_per_timestep([a, b], "rg_sq")


def test_concat_with_repeat_label_single_file_has_no_repeat_column():
    df = pd.DataFrame({"timestep": [0], "mol": [1], "dx": [1.0], "dy": [0.0], "dz": [0.0]})
    out = merge.concat_with_repeat_label([df])
    assert out is df
    assert "repeat" not in out.columns


def test_concat_with_repeat_label_stacks_and_labels():
    a = pd.DataFrame({"timestep": [0], "mol": [1], "dx": [1.0]})
    b = pd.DataFrame({"timestep": [0], "mol": [1], "dx": [2.0]})
    out = merge.concat_with_repeat_label([a, b])
    assert len(out) == 2
    assert out["repeat"].tolist() == [1, 2]
    assert out["dx"].tolist() == [1.0, 2.0]


def test_average_rheology_single_file_is_unchanged():
    df = pd.DataFrame({"lag_time": [0.0, 1.0], "G_GK": [10.0, 5.0], "G_FSR": [6.0, 3.0]})
    out = merge.average_rheology([df], ("G_GK", "G_FSR"))
    assert out is df


def test_average_rheology_averages_matching_grids():
    a = pd.DataFrame({"lag_time": [0.0, 1.0], "G_GK": [10.0, 4.0], "G_FSR": [6.0, 2.0]})
    b = pd.DataFrame({"lag_time": [0.0, 1.0], "G_GK": [12.0, 6.0], "G_FSR": [8.0, 4.0]})
    out = merge.average_rheology([a, b], ("G_GK", "G_FSR"))
    assert out["lag_time"].tolist() == [0.0, 1.0]
    assert out["G_GK"].tolist() == [11.0, 5.0]
    assert out["G_FSR"].tolist() == [7.0, 3.0]


def test_average_rheology_truncates_to_the_shortest_file():
    a = pd.DataFrame({"lag_time": [0.0, 1.0, 2.0], "G_GK": [10.0, 5.0, 2.0], "G_FSR": [1.0] * 3})
    b = pd.DataFrame({"lag_time": [0.0, 1.0], "G_GK": [12.0, 7.0], "G_FSR": [1.0, 1.0]})
    out = merge.average_rheology([a, b], ("G_GK", "G_FSR"))
    assert len(out) == 2
    assert out["G_GK"].tolist() == [11.0, 6.0]


def test_average_rheology_rejects_mismatched_lag_time_grids():
    a = pd.DataFrame({"lag_time": [0.0, 1.0], "G_GK": [1.0, 1.0], "G_FSR": [1.0, 1.0]})
    b = pd.DataFrame({"lag_time": [0.0, 2.0], "G_GK": [1.0, 1.0], "G_FSR": [1.0, 1.0]})
    with pytest.raises(ValueError, match="lag-time grids"):
        merge.average_rheology([a, b], ("G_GK", "G_FSR"))


def test_average_dynamics_single_file_is_unchanged():
    df = pd.DataFrame({"lag_time": [0.0, 1.0], "acf": [1.0, 0.5]})
    out = merge.average_dynamics([df], "acf")
    assert out is df


def test_average_dynamics_averages_and_truncates():
    a = pd.DataFrame({"lag_time": [0.0, 1.0, 2.0], "acf": [1.0, 0.6, 0.4]})
    b = pd.DataFrame({"lag_time": [0.0, 1.0], "acf": [1.0, 0.4]})
    out = merge.average_dynamics([a, b], "acf")
    assert len(out) == 2
    assert out["lag_time"].tolist() == [0.0, 1.0]
    assert out["acf"].tolist() == pytest.approx([1.0, 0.5])


def test_merge_regimes_concatenates_and_sorts_non_overlapping_ranges():
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([3.0, 4.0, 5.0])
    x, cols = merge.merge_regimes(
        [x1, x2], [{"acf": np.array([1.0, 0.5, 0.3])}, {"acf": np.array([0.2, 0.1, 0.05])}]
    )
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert cols["acf"].tolist() == [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def test_merge_regimes_sorts_out_of_order_regimes():
    x1 = np.array([3.0, 4.0])
    x2 = np.array([0.0, 1.0])
    x, cols = merge.merge_regimes(
        [x1, x2], [{"acf": np.array([0.2, 0.1])}, {"acf": np.array([1.0, 0.5])}]
    )
    assert x.tolist() == [0.0, 1.0, 3.0, 4.0]
    assert cols["acf"].tolist() == [1.0, 0.5, 0.2, 0.1]


def test_merge_regimes_dedupes_an_agreeing_exact_duplicate():
    x1 = np.array([0.0, 1.0])
    x2 = np.array([1.0, 2.0])
    x, cols = merge.merge_regimes(
        [x1, x2], [{"acf": np.array([1.0, 0.5])}, {"acf": np.array([0.5, 0.3])}]
    )
    assert x.tolist() == [0.0, 1.0, 2.0]
    assert cols["acf"].tolist() == [1.0, 0.5, 0.3]


def test_merge_regimes_rejects_a_disagreeing_exact_duplicate():
    x1 = np.array([0.0, 1.0])
    x2 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="disagrees"):
        merge.merge_regimes(
            [x1, x2], [{"acf": np.array([1.0, 0.5])}, {"acf": np.array([0.9, 0.3])}]
        )


def test_merge_regimes_rejects_a_genuine_overlap():
    x1 = np.array([0.0, 1.0, 2.5])
    x2 = np.array([2.0, 3.0])
    with pytest.raises(ValueError, match="overlap"):
        merge.merge_regimes(
            [x1, x2], [{"acf": np.array([1.0, 0.5, 0.3])}, {"acf": np.array([0.35, 0.2])}]
        )


def test_merge_regimes_requires_at_least_one_regime():
    with pytest.raises(ValueError, match="At least one regime"):
        merge.merge_regimes([], [])
