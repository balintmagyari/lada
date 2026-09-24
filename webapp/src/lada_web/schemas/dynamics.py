"""Form fields shared by the view and download route of each dynamics calculation."""

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, File, Form, UploadFile
from lada.analysis import ChainTopology

from ..config import Settings, get_settings
from ..services import dynamics_service
from ..uploads import get_workspace, save_upload


class TopologyForm:
    """How the beads of the trajectory are grouped into chains.

    `ws` is a sub-dependency rather than an argument because FastAPI caches dependencies per
    request: a view route that already declares one receives the very same directory here, and a
    download or job-start route -- which declares none -- gets its own, cleaned up with the
    request. Either way no route signature has to carry a workspace through to this form.
    """

    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_settings)],
        ws: Annotated[Path, Depends(get_workspace)],
        mode: Annotated[Literal["uniform", "custom", "data_file"], Form()] = "uniform",
        n_chains: Annotated[int | None, Form(ge=1)] = None,
        beads_per_chain: Annotated[int | None, Form(ge=2)] = None,
        first_index: Annotated[int, Form(ge=0)] = 0,
        custom: Annotated[str, Form()] = "",
        topology_file: Annotated[UploadFile | None, File()] = None,
    ) -> None:
        self.settings = settings
        self.ws = ws
        self.mode = mode
        self.n_chains = n_chains
        self.beads_per_chain = beads_per_chain
        self.first_index = first_index
        self.custom = custom
        self.topology_file = topology_file
        self._topology: ChainTopology | None = None

    def topology(self) -> ChainTopology | None:
        """The uploaded data file's chains, parsed once.

        An UploadFile can only be read once -- a second read returns no bytes, and the file then
        looks empty -- so the result is kept. Nothing asks twice per request today, but a route
        wanting both the chains and their ends would otherwise fail in a way that reads as a
        corrupt upload rather than as a bug here.
        """
        if self.mode != "data_file" or self.topology_file is None:
            return None
        if self._topology is None:
            directory = self.ws / "topology"
            directory.mkdir(parents=True, exist_ok=True)
            # Its own subdirectory: download routes write the result table into the workspace
            # root, so a data file named results.csv would otherwise collide with it.
            path = save_upload(self.topology_file, directory, self.settings.max_topology_bytes)
            self._topology = dynamics_service.read_topology(path)
        return self._topology

    def summary(self) -> str | None:
        """One line naming what a data file gave us, for the result's metadata."""
        topology = self.topology()
        return None if topology is None else dynamics_service.topology_summary(topology)

    def notes(self) -> list[str]:
        topology = self.topology()
        return [] if topology is None else topology.notes

    # n_atoms comes from the trajectory header, which is read before either of these runs:
    # a uniform layout is range-checked arithmetically rather than after it is built.
    def chain_indices(self, n_atoms: int) -> np.ndarray:
        return dynamics_service.resolve_chain_indices(
            self.mode,
            n_chains=self.n_chains,
            beads_per_chain=self.beads_per_chain,
            first_index=self.first_index,
            custom=self.custom,
            n_atoms=n_atoms,
            topology=self.topology(),
        )

    def segment_pairs(self, n_atoms: int) -> np.ndarray:
        return dynamics_service.resolve_segment_pairs(
            self.mode,
            n_chains=self.n_chains,
            beads_per_chain=self.beads_per_chain,
            first_index=self.first_index,
            custom=self.custom,
            n_atoms=n_atoms,
            topology=self.topology(),
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
