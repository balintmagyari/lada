# LaDa

[![Tests](https://github.com/balintmagyari/lada/actions/workflows/tests.yml/badge.svg)](https://github.com/balintmagyari/lada/actions/workflows/tests.yml)

**LaDa** (LAMMPS Data Access) is a Python package for parsing LAMMPS output files, performing polymer MD analysis, and exporting results for LaTeX/pgfplots. The name is, quite intentionally, borrowed from the legendary Soviet car brand **LADA**—because much like its cars, this library aims to be simple, reliable, and able to run just about anywhere without unnecessary luxury features.

---

## Installation

```bash
pip install lada
```

> Requires Python 3.12+.

---

## Quick example

```python
from lada import read_dump, read_lammps_acf
from lada.analysis import calculate_avg_rg_sq, calc_stress_relaxation
from lada.exporters import write_pgfplots_table

# Load a dump trajectory and compute Rg² per timestep
df = read_dump("trajectory.dump")
rg_sq = calculate_avg_rg_sq(df, coord_cols=['xu', 'yu', 'zu'], molecule_col='mol')

# Load stress ACF data and compute the relaxation modulus
acf_df = read_lammps_acf("acf_output.txt")
G = calc_stress_relaxation(acf_df, volume=500.0, temperature=1.0)

# Export G(t) for pgfplots
write_pgfplots_table(G, "G_t.dat", comment="Stress relaxation modulus, T=1.0")
```

---

## Package structure

| Submodule | Purpose |
|---|---|
| `lada.parsers` | Read LAMMPS dump, log, data, and ACF files |
| `lada.analysis` | Vectorized polymer MD calculations |
| `lada.exporters` | Export data to LaTeX/pgfplots-compatible files |
| `lada.modifiers` | Modify LAMMPS topology data files |

---

## 1) Parsers

### 1a) Dump files — `dump_frames` / `read_dump`

#### Iterate frame-by-frame

Useful when you want to process one timestep at a time without loading the full trajectory into memory.

```python
from lada import dump_frames

for frame in dump_frames("trajectory.dump"):
    timestep = frame.metadata["TIMESTEP"]              # int
    box      = frame.metadata["BOX BOUNDS pp pp pp"]   # np.ndarray (3, 2)

    ids = frame.get_column("id")
    xs  = frame.get_column("x")

    df = frame.to_pandas()
```

**`DumpFrame` column helpers:**

```python
ids    = frame.get_column("id")                       # raises KeyError if absent
result = frame.get_column_or("charge", default=None)  # returns None if absent
idx    = frame.column_index("type")                   # raw integer index
df     = frame.to_pandas(copy=False)                  # skip the defensive copy for read-only use
```

> **Deprecated alias:** `iter_dump_frames` behaves the same as `dump_frames` but emits a `DeprecationWarning` and will be removed in lada 2.0.0. Use `dump_frames` in new code.

**Metadata conversion rules:**
- `TIMESTEP` → `int`
- `NUMBER OF ...` entries → `int`
- `BOX BOUNDS ...` → `np.ndarray` of shape `(3, 2)` for orthogonal boxes, `(3, 3)` for triclinic boxes (third column contains tilt factors xy, xz, yz)

#### Load entire trajectory into a single DataFrame

```python
from lada import read_dump

df = read_dump("trajectory.dump")
# df has a leading 'timestep' column followed by all atom data columns
```

The `timestep_col` argument (default: `'timestep'`) controls the name of the prepended column.

---

### 1b) Log files — `read_lammps_log`

Extracts the thermodynamic table written between the `Per MPI rank memory allocation` and `Loop time` markers.

```python
from lada import read_lammps_log

thermo = read_lammps_log("log.lammps")
print(thermo.columns)          # ['Step', 'Temp', 'E_pair', ...]
energy = thermo.get("E_pair")  # np.ndarray
df = thermo.to_pandas()        # pd.DataFrame
```

---

### 1c) Data files — `read_data_file`

Parses LAMMPS data files written by `write_data`. Auto-detects the atom style from the `Atoms # style` comment.

```python
from lada import read_data_file

data = read_data_file("system.data")

# Global header values are stored on `metadata`
style       = data.metadata["atom style"]   # auto-detected from the `Atoms # <style>` comment
description = data.metadata["description"]  # raw first-line description of the file
n_atoms     = data.metadata.get("atoms")    # header counts (atoms, bonds, angles, ...)

atoms = data.get("Atoms")   # np.ndarray
bonds = data.get("Bonds")   # np.ndarray

df_atoms = data.to_pandas(section="Atoms")   # columns inferred from atom style
df_bonds = data.to_pandas(section="Bonds")   # columns: bond_id, bond_type, atom1_id, atom2_id
```

**Allowed `section` values:** `Atoms`, `Bonds`, `Masses`, `Velocities`, `Angles`, `Dihedrals`, `Impropers`, `Nonbond Coeffs`, `Bond Coeffs`, `Angle Coeffs`, `Dihedral Coeffs`, `Improper Coeffs`. Hardcoded column names are provided for `Atoms` (via atom style), `Bonds`, `Masses`, and `Velocities`; the remaining sections fall back to generic `col_0`, `col_1`, ... with a warning.

**Supported atom styles:** `atomic`, `charge`, `bond`, `molecular`, `full`.

---

### 1d) ACF files — `read_lammps_acf`

Reads output from LAMMPS [`fix ave/correlate/long`](https://docs.lammps.org/fix_ave_correlate_long.html). Automatically locates and returns the **last** `# Timestep: N` block in the file (earlier blocks are preliminary averages).

```python
from lada import read_lammps_acf

df = read_lammps_acf("acf_output.txt")
# Columns: lag_time, <ACF columns from line 1 of the file>, timestep
# e.g. lag_time, ACF_Sxy, ACF_Sxz, ACF_Syz, ACF_Nxy, ACF_Nxz, ACF_Nyz, timestep
```

The ACF column names are read from the **comma-separated header on line 1** of the input file — they are not hardcoded. The `lag_col` argument (default: `'lag_time'`) controls the name of the lag-time column.

> Requires at least two `# Timestep:` blocks in the file (the t=0 reference plus at least one production block); otherwise raises `ValueError`.

---

## 2) Analysis

All DataFrame-based functions accept either a `pd.DataFrame` or a `np.ndarray` (pass `columns=list_of_names` for arrays). Single-frame data returns a `float`; multi-frame trajectories return `dict[timestep, float]`.

Functions that operate on `.npz` trajectory files expect the archive to contain a `'coords'` key with shape `(n_frames, n_atoms, 3)` and return a `np.ndarray` of shape `(n_frames, 2)` with columns `[lag_time, value]`.

---

### 2a) Radius of gyration — `calculate_avg_rg_sq`

Ensemble-average squared radius of gyration, optionally mass-weighted. Use **unwrapped** coordinates (`xu`, `yu`, `zu`) to avoid periodic-boundary artifacts.

```python
from lada.analysis import calculate_avg_rg_sq

# Single timestep → float
rg_sq = calculate_avg_rg_sq(df, coord_cols=['xu', 'yu', 'zu'], molecule_col='mol')

# Multi-frame trajectory → dict[timestep, float]
rg_sq = calculate_avg_rg_sq(df, coord_cols=['xu', 'yu', 'zu'],
                             molecule_col='mol', timestep_col='timestep')

# With mass weighting
rg_sq = calculate_avg_rg_sq(df, coord_cols=['xu', 'yu', 'zu'],
                             molecule_col='mol', mass_col='mass')
```

---

### 2b) End-to-end distance — `calculate_avg_ree_sq`

Ensemble-average squared end-to-end distance. Chain ends are identified as the minimum and maximum atom ID within each molecule.

```python
from lada.analysis import calculate_avg_ree_sq

ree_sq = calculate_avg_ree_sq(df, coord_cols=['xu', 'yu', 'zu'],
                               molecule_col='mol', timestep_col='timestep')
```

---

### 2c) End-to-end vectors — `calculate_ree_vectors`

Returns the full end-to-end vector for every molecule at every timestep as a DataFrame with columns `[mol, dx, dy, dz]` (plus `timestep` when the input has multiple frames).

```python
from lada.analysis import calculate_ree_vectors

vectors = calculate_ree_vectors(df, coord_cols=['xu', 'yu', 'zu'], molecule_col='mol')
```

---

### 2d) Segment end-to-end ACF — `calculate_segment_acf`

Normalized autocorrelation function C(t) = ⟨R(t)·R(0)⟩ / ⟨R(0)·R(0)⟩ of the chain end-to-end vector, averaged over all chains.

```python
from lada.analysis import calculate_segment_acf
import numpy as np

# segment_pairs: (n_chains, 2) array of 0-indexed [head_bead, tail_bead] indices
segment_pairs = np.array([[0, 49], [50, 99]])  # two 50-bead chains
result = calculate_segment_acf("trajectory.npz", segment_pairs, time_per_frame=0.5)
# shape: (n_frames, 2) — columns [lag_time, C(t)]
```

> Discard the last 10–20 % of the output when fitting relaxation times, as statistical quality decreases at long lags.

---

### 2e) Rouse mode ACF — `calculate_rouse_mode_acf`

Normalized ACF for the Rouse mode amplitude X_p(t), computed via a discrete cosine projection. Used to extract mode-dependent relaxation times τ_p.

```python
from lada.analysis import calculate_rouse_mode_acf
import numpy as np

# chain_indices: (n_chains, beads_per_chain) array of 0-indexed bead indices
chain_indices = np.arange(100).reshape(2, 50)
result = calculate_rouse_mode_acf("trajectory.npz", chain_indices, p=1, time_per_frame=0.5)
# shape: (n_frames, 2) — columns [lag_time, C_p(t)]
```

- `p=0`: center-of-mass translation (does not decay to zero)
- `p=1`: fundamental (whole-chain) relaxation mode
- `p>1`: increasingly local segmental motions

Raises `ValueError` if `p >= beads_per_chain`.

---

### 2f) Intermediate scattering function — `calculate_isf`

Coherent intermediate scattering function F(q, t) / F(q, 0), computed via the density-fluctuation autocorrelation method. Isotropic orientational averaging is performed over `n_vectors` scattering vectors distributed on a Fibonacci lattice.

```python
from lada.analysis import calculate_isf

result = calculate_isf("trajectory.npz", time_per_frame=0.5, q_magnitude=7.0, n_vectors=50)
# shape: (n_frames, 2) — columns [lag_time, F(q,t)/F(q,0)]
```

---

### 2g) Stress relaxation modulus — `calc_stress_relaxation`

Computes G(t) from a stress-ACF DataFrame (as returned by `read_lammps_acf`) using two methods:

- **G_GK** — standard Green-Kubo formula: average over the three independent shear stress ACFs
- **G_FSR** — full stress relaxation formula: extends Green-Kubo with the three normal stress difference ACFs, using the isotropic identity ⟨(σ_αα − σ_ββ)²⟩ = 4⟨σ_αβ²⟩ to set the relative weighting

```python
from lada.analysis import calc_stress_relaxation

G = calc_stress_relaxation(acf_df, volume=500.0, temperature=1.0)
# G is a DataFrame with columns: lag_time, G_GK, G_FSR
```

**Required input columns** (matched exactly — no typo tolerance):

| Column | Description |
|---|---|
| `ACF_Sxy`, `ACF_Sxz`, `ACF_Syz` | Shear stress ACFs |
| `ACF_Nxy`, `ACF_Nxz`, `ACF_Nyz` | Normal stress difference ACFs |
| `lag_time` | Lag-time column (name configurable via `lag_col`) |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `volume` | — | System volume in length³ |
| `temperature` | — | System temperature (energy units) |
| `lag_col` | `'lag_time'` | Name of the lag-time column in the input |
| `kB` | `1.0` | Boltzmann constant. Default is for LJ reduced units; set to the physical value (e.g. `1.380649e-23`) when using SI units |

Raises `KeyError` if any required ACF column is absent and `ValueError` if `volume`/`temperature` are non-positive or `lag_col` is missing.

---

## 3) Exporters

### `write_pgfplots_table`

Writes a delimited data file consumable by pgfplots `\addplot table` in LaTeX. Accepts `pd.DataFrame`, `np.ndarray`, or `dict`.

```python
from lada.exporters import write_pgfplots_table

# From a DataFrame (column names become the header automatically)
write_pgfplots_table(df, "results.dat")

# From a dict of arrays
write_pgfplots_table(
    {"lag_time": t, "G_GK": g_gk, "G_FSR": g_fsr},
    "G_t.dat",
    delimiter=',',
    comment="Stress relaxation, T=1.0, V=500",
)

# From a NumPy array with explicit column names
write_pgfplots_table(
    np.column_stack([time, rg_sq]),
    "rg.dat",
    columns=["t", "Rg2"],
    fmt="%.4f",
)
```

**Usage in LaTeX:**

```latex
\addplot table[x=t, y=Rg2, col sep=space]{rg.dat};
```

Change `col sep` to `comma` or `tab` to match `delimiter=','` or `delimiter='\t'`.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `delimiter` | `' '` | Field separator: `' '`, `'\t'`, or `','` |
| `fmt` | `'%.6g'` | Printf-style format applied to every numeric value |
| `comment` | `None` | Text prepended to the file; each line prefixed with `%` |
| `columns` | `None` | Override or supply column names |

---

## 4) Modifiers

### `rewrite_end_beads`

Reads a LAMMPS data file, identifies the terminal beads of each polymer chain (minimum and maximum atom ID per molecule), and rewrites the topology with those beads assigned a new atom type. Clones mass and pair coefficients from an existing type and updates the header atom-types count automatically.

```python
from lada.modifiers import rewrite_end_beads

rewrite_end_beads(
    input_file="system.data",
    output_file="system_endtype.data",
    new_end_type=3,
    base_type=1,
)
```

---

## License

[MIT](LICENSE)
