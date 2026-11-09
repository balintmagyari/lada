"""JSON request bodies for merging several already-computed regime results into one.

Unlike every other route in this app, these take no file upload: each regime's numeric result
was already computed by the ordinary per-file/per-regime calculation route (client-orchestrated,
see static/js/app.js), and this step only combines the resulting series.
"""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, Field, FiniteFloat, field_validator, model_validator

from ..uploads import safe_filename

# A regime is one run's correlation function: a few hundred points from a multi-tau correlator,
# at most max_trajectory_frames (3000 by default) from a trajectory. The cap leaves ample room
# for that while keeping the Prony design matrix (n_points x n_modes) a bounded size.
MAX_POINTS_PER_REGIME = 100_000
# The page adds regimes one at a time (fast / mid / slow); nobody stitches more than a handful.
MAX_REGIMES = 20
# Number of Maxwell modes in a Prony fit. NNLS on the n_points x n_modes matrix grows quickly
# with the mode count, and a handful of modes per decade of lag time is already generous.
MAX_PRONY_MODES = 200


def _safe_stem(stem: str) -> str:
    """The stem names the file written into the workspace; it must not be able to leave it."""
    return safe_filename(stem, default="regimes")


# Every body that carries a download name passes it through here, so `"/abs/x"` becomes "x"
# and `"../../x"` becomes "x" before any route can join it onto a directory.
DownloadStem = Annotated[str, AfterValidator(_safe_stem)]
LagTimes = Annotated[list[FiniteFloat], Field(max_length=MAX_POINTS_PER_REGIME)]
Values = Annotated[list[float], Field(max_length=MAX_POINTS_PER_REGIME)]


def _check_regime(label: str, lag_time: list[float], **series: list[float]) -> None:
    for name, values in series.items():
        if len(values) != len(lag_time):
            raise ValueError(
                f"Regime '{label}' has {len(lag_time)} lag times but {len(values)} values in "
                f"'{name}'; every series must have one value per lag time."
            )
    # The merge orders points by lag time, so two points at one lag time would come out in an
    # arbitrary order and the curve would zigzag there. A correlator never repeats a lag.
    if len(set(lag_time)) != len(lag_time):
        raise ValueError(
            f"Regime '{label}' repeats a lag time; each lag time may appear only once."
        )


class DynamicsRegime(BaseModel):
    label: str
    lag_time: LagTimes
    value: Values

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _check_regime(self.label, self.lag_time, value=self.value)
        return self


class DynamicsMergeRequest(BaseModel):
    regimes: list[DynamicsRegime] = Field(min_length=1, max_length=MAX_REGIMES)
    value_column: str = Field(min_length=1)
    title: str
    y_label: str
    download_stem: DownloadStem

    @field_validator("value_column")
    @classmethod
    def _not_the_lag_column(cls, value: str) -> str:
        if value == "lag_time":
            raise ValueError("value_column cannot be 'lag_time': that is the merged x column.")
        return value


class RheologyRegime(BaseModel):
    label: str
    lag_time: LagTimes
    G_GK: Values
    G_FSR: Values

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _check_regime(self.label, self.lag_time, G_GK=self.G_GK, G_FSR=self.G_FSR)
        return self


class RheologyMergeRequest(BaseModel):
    regimes: list[RheologyRegime] = Field(min_length=1, max_length=MAX_REGIMES)
    download_stem: DownloadStem = "regimes"


class DynamicModuliFromRegimesRequest(BaseModel):
    """Merge regimes' G(t) first, then fit the Prony series once on the combined curve.

    Two regimes' independently-fit omega grids aren't meaningfully concatenable the way two
    non-overlapping lag-time windows are, so the merge happens here — before the one fit — not
    on the frequency-domain output the way the other three dynamics/rheology merges do.
    """

    regimes: list[RheologyRegime] = Field(min_length=1, max_length=MAX_REGIMES)
    method: Literal["GK", "FSR", "both"] = "GK"
    t_min: float = Field(gt=0, default=1.0)
    t_cutoff: float | None = None
    n_modes: int = Field(ge=1, le=MAX_PRONY_MODES, default=50)
    n_omega: int = Field(ge=2, le=5000, default=200)
    omega_min: float | None = Field(gt=0, default=None)
    omega_max: float | None = Field(gt=0, default=None)
    download_stem: DownloadStem = "regimes"
