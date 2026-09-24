"""What the page shows once a chain-topology data file has been chosen."""

from pydantic import BaseModel, Field


class ChainLength(BaseModel):
    """How many chains hold a given number of beads."""

    beads: int
    chains: int


class TopologyPreview(BaseModel):
    """A read of a data file, shown before anything is calculated.

    The spread of chain lengths matters as much as the count. A `full`-style melt in water
    yields thousands of three-atom "chains" alongside the real ones, and a preview reporting
    only "4238 chains" would look like a success; the histogram is what makes that visible.

    It deliberately says nothing about the trajectory: checking that the atom counts agree
    would mean uploading the `.npz` twice, once to look at and once to compute with, so that
    check stays where the calculation is.
    """

    n_atoms: int
    n_chains: int
    # Set only when every chain is the same length -- the Rouse-mode page needs that number,
    # and null is the honest answer for a polydisperse melt.
    beads_per_chain: int | None = None
    shortest: int
    longest: int
    median: int
    lengths: list[ChainLength] = Field(default_factory=list)
    source: str
    summary: str
    notes: list[str] = Field(default_factory=list)
