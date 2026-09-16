"""JSON request bodies for merging several already-computed regime results into one.

Unlike every other route in this app, these take no file upload: each regime's numeric result
was already computed by the ordinary per-file/per-regime calculation route (client-orchestrated,
see static/js/app.js), and this step only combines the resulting series.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DynamicsRegime(BaseModel):
    label: str
    lag_time: list[float]
    value: list[float]


class DynamicsMergeRequest(BaseModel):
    regimes: list[DynamicsRegime] = Field(min_length=1)
    value_column: str
    title: str
    y_label: str
    download_stem: str


class RheologyRegime(BaseModel):
    label: str
    lag_time: list[float]
    G_GK: list[float]
    G_FSR: list[float]


class RheologyMergeRequest(BaseModel):
    regimes: list[RheologyRegime] = Field(min_length=1)
    download_stem: str = "regimes"


class DynamicModuliFromRegimesRequest(BaseModel):
    """Merge regimes' G(t) first, then fit the Prony series once on the combined curve.

    Two regimes' independently-fit omega grids aren't meaningfully concatenable the way two
    non-overlapping lag-time windows are, so the merge happens here — before the one fit — not
    on the frequency-domain output the way the other three dynamics/rheology merges do.
    """

    regimes: list[RheologyRegime] = Field(min_length=1)
    method: Literal["GK", "FSR", "both"] = "GK"
    t_min: float = Field(gt=0, default=1.0)
    t_cutoff: float | None = None
    n_modes: int = Field(ge=1, default=50)
    n_omega: int = Field(ge=2, le=5000, default=200)
    omega_min: float | None = Field(gt=0, default=None)
    omega_max: float | None = Field(gt=0, default=None)
    download_stem: str = "regimes"
