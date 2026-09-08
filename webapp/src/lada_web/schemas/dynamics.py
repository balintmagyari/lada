"""Form fields shared by the view and download route of each dynamics calculation."""

from typing import Annotated, Literal

import numpy as np
from fastapi import Form

from ..services import dynamics_service


class TopologyForm:
    """How the beads of the trajectory are grouped into chains."""

    def __init__(
        self,
        mode: Annotated[Literal["uniform", "custom"], Form()] = "uniform",
        n_chains: Annotated[int | None, Form(ge=1)] = None,
        beads_per_chain: Annotated[int | None, Form(ge=2)] = None,
        first_index: Annotated[int, Form(ge=0)] = 0,
        custom: Annotated[str, Form()] = "",
    ) -> None:
        self.mode = mode
        self.n_chains = n_chains
        self.beads_per_chain = beads_per_chain
        self.first_index = first_index
        self.custom = custom

    def chain_indices(self) -> np.ndarray:
        return dynamics_service.resolve_chain_indices(
            self.mode,
            n_chains=self.n_chains,
            beads_per_chain=self.beads_per_chain,
            first_index=self.first_index,
            custom=self.custom,
        )

    def segment_pairs(self) -> np.ndarray:
        return dynamics_service.resolve_segment_pairs(
            self.mode,
            n_chains=self.n_chains,
            beads_per_chain=self.beads_per_chain,
            first_index=self.first_index,
            custom=self.custom,
        )


class FrameTime:
    """The simulation time between two saved frames."""

    def __init__(self, time_per_frame: Annotated[float, Form(gt=0)]) -> None:
        self.time_per_frame = time_per_frame


class RouseMode:
    """Which Rouse mode to correlate."""

    def __init__(self, p: Annotated[int, Form(ge=0)]) -> None:
        self.p = p


class IsfParams:
    """The scattering vector magnitude and how many directions to average over."""

    def __init__(
        self,
        q_magnitude: Annotated[float, Form(gt=0)],
        # Two directions is the floor: lada spreads the vectors over n_vectors - 1 steps.
        n_vectors: Annotated[int, Form(ge=2, le=1000)] = 50,
    ) -> None:
        self.q_magnitude = q_magnitude
        self.n_vectors = n_vectors
