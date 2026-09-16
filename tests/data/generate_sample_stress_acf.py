"""Generate tests/data/sample_stress_acf.txt: a synthetic stress-autocorrelation file in the
`fix ave/correlate` layout that lada.parsers.read_lammps_acf reads and with the exact column
names lada.analysis.calc_stress_relaxation requires (sample_acf.txt uses acf_x/y/z instead).

Run from anywhere:  python tests/data/generate_sample_stress_acf.py

Two timestep blocks are written (t=0 reference, then 1000); read_lammps_acf uses the last one.
Each ACF is a two-exponential decay with a small seeded noise term and stays positive.
"""

from pathlib import Path

import numpy as np

COLUMNS = ["ACF_Sxy", "ACF_Sxz", "ACF_Syz", "ACF_Nxy", "ACF_Nxz", "ACF_Nyz"]
AMPLITUDES = {
    "ACF_Sxy": 1.00,
    "ACF_Sxz": 0.95,
    "ACF_Syz": 1.05,
    "ACF_Nxy": 3.60,
    "ACF_Nxz": 3.80,
    "ACF_Nyz": 4.00,
}
N_LAGS = 100
DT = 0.5
TAU_FAST = 2.0
TAU_SLOW = 20.0
SLOW_FRACTION = 0.3
SEED = 20260908
OUT = Path(__file__).with_name("sample_stress_acf.txt")


def acf_values(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    base = amplitude * (
        (1 - SLOW_FRACTION) * np.exp(-t / TAU_FAST) + SLOW_FRACTION * np.exp(-t / TAU_SLOW)
    )
    noise = rng.normal(scale=1e-3 * amplitude, size=t.shape)
    noise[0] = 0.0
    return np.maximum(base + noise, 1e-6)


def main(out: Path = OUT, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(N_LAGS) * DT
    lines = [",".join(COLUMNS)]
    for timestep, scale in ((0, 1.0), (1000, 0.98)):
        lines.append(f"# Timestep: {timestep}")
        block = np.column_stack([acf_values(t, AMPLITUDES[c] * scale, rng) for c in COLUMNS])
        for lag, row in zip(t, block, strict=True):
            lines.append(f"{lag:g} " + " ".join(f"{v:.6f}" for v in row))
        lines.append("")
    out.write_text("\n".join(lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
    # A second seed-repeat (same lag-time grid and amplitudes, independent noise), for
    # multi-file averaging tests.
    main(out=Path(__file__).with_name("sample_stress_acf_repeat2.txt"), seed=SEED + 1)
