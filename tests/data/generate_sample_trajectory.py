"""Generate tests/data/sample_trajectory.npz: a small deterministic synthetic polymer trajectory.

Run from anywhere:  python tests/data/generate_sample_trajectory.py

Layout: N_CHAINS chains of BEADS_PER_CHAIN beads. Atom index = chain * BEADS_PER_CHAIN + bead,
0-indexed and ascending, i.e. the "uniform" topology convention used by lada-web. Frame 0 is a
freely-jointed chain per molecule; every later frame adds an independent Gaussian kick to each
bead, so end-to-end and Rouse-mode autocorrelations decay within the 32 frames.
"""

from pathlib import Path

import numpy as np

N_FRAMES = 32
N_CHAINS = 4
BEADS_PER_CHAIN = 8
BOND_LENGTH = 1.0
KICK_SIGMA = 0.35
SEED = 20260908
OUT = Path(__file__).with_name("sample_trajectory.npz")


def build_coords(rng: np.random.Generator) -> np.ndarray:
    n_atoms = N_CHAINS * BEADS_PER_CHAIN
    coords = np.empty((N_FRAMES, n_atoms, 3))
    origins = rng.uniform(-5.0, 5.0, size=(N_CHAINS, 3))
    steps = rng.normal(size=(N_CHAINS, BEADS_PER_CHAIN - 1, 3))
    steps *= BOND_LENGTH / np.linalg.norm(steps, axis=2, keepdims=True)
    first = np.concatenate(
        [origins[:, None, :], origins[:, None, :] + np.cumsum(steps, axis=1)], axis=1
    )
    coords[0] = first.reshape(n_atoms, 3)
    for t in range(1, N_FRAMES):
        coords[t] = coords[t - 1] + rng.normal(scale=KICK_SIGMA, size=(n_atoms, 3))
    return coords


def main(out: Path = OUT, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    np.savez_compressed(out, coords=build_coords(rng))


if __name__ == "__main__":
    main()
    # A second seed-repeat of the same system (same chain layout, independent noise), for
    # multi-file averaging tests: same n_atoms so it's a valid repeat, different values so an
    # averaged result is distinguishable from either file alone.
    main(out=Path(__file__).with_name("sample_trajectory_repeat2.npz"), seed=SEED + 1)
