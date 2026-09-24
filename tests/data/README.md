# Test Data Directory

This directory contains (or will contain) test data files used by the automated test suite.

## File Size Limits for GitHub

Keep test data files under these limits for optimal CI/CD performance:

| Category | Limit | Reason |
|----------|-------|--------|
| **Individual file** | < 10 MB | Git performance, CI/CD speed |
| **Total test data** | < 50 MB | Repository size, clone speed |
| **Ideal file size** | < 1-5 MB | Fast test execution |
| **Absolute max** | 100 MB | GitHub hard limit per file |

**Recommendation**: Aim for minimal representative samples, preferably < 1 MB total.

---

## Required Test Data Files

Provide these files as reduced/sample versions:

### **1. Dump File Sample**
**File name**: `sample_dump.dump`  
**Purpose**: Test dump file parsing (`test_dump_parsers.py`)  
**Current source**: `Atoms.dump` (158 MB → reduce to < 5 MB)  
**Requirements**:
- Must be valid LAMMPS dump format
- Minimum 2 timesteps (for iteration testing)
- At least 10 atoms per frame
- Include columns: id, type, x, y, z (or similar)

**Size target**: < 1 MB

---

### **2. Log File Sample**
**File name**: `sample_log.lammps`  
**Purpose**: Test log file parsing (`test_log_parsers.py`)  
**Current source**: `log.lammps` (162 MB → reduce to < 5 MB)  
**Requirements**:
- Valid LAMMPS log file format
- Include "Per MPI rank memory allocation" marker
- Include thermo data table with headers
- Include "Loop time" marker

**Size target**: < 500 KB

---

### **3. Data File Sample**
**File name**: `sample.data`  
**Purpose**: Test LAMMPS data file parsing (`test_data_parsers.py`)  
**Current source**: `Vitrimer.data` (4.8 MB → reduce to < 1 MB)  
**Requirements**:
- Valid LAMMPS data file format
- Include Atoms section
- Include Bonds section (if available)
- Proper header with atom/bond counts

**Size target**: < 1 MB

---

### **4. ACF File Sample (Optional)**
**File name**: `sample_acf.txt`  
**Purpose**: Test autocorrelation function parsing  
**Current source**: From your ACF calculations  
**Requirements**:
- Valid format from `fix ave/correlate/long`
- At least 10 time points
- Proper column headers

**Size target**: < 100 KB

**Note**: This is optional. If not provided, that test will be skipped.

---

## How to Provide Reduced Versions

### **Option 1: Subsample Large Files (Recommended)**

```bash
# For dump files: Keep first 2-10 timesteps
# For log files: Keep only one block with minimal thermo data
# For data files: Keep first 100-1000 atoms
```

**Example - Python script to reduce dump file:**
```python
with open("Atoms.dump", "r") as infile:
    with open("sample_dump.dump", "w") as outfile:
        timestep_count = 0
        in_atoms = False
        
        for line in infile:
            if "ITEM: TIMESTEP" in line:
                timestep_count += 1
                if timestep_count > 2:  # Keep only 2 timesteps
                    break
            
            outfile.write(line)
            
            if "ITEM: ATOMS" in line:
                in_atoms = True
            elif in_atoms and "ITEM:" in line:
                in_atoms = False
```

### **Option 2: Create Minimal Valid Files**

For very small test files, manually create minimal valid LAMMPS formats.

---

## Current Files (Don't Commit to Git)

These large files should stay on your local machine but NOT be committed:

- `Atoms.dump` (158 MB)
- `Atoms copy.dump` (1.7 KB) — Can use this!
- `Bonds.dump` (55 MB)
- `Vitrimer.data` (4.8 MB)
- `log.lammps` (162 MB)

**Action**: Once you provide reduced samples, the large files will be excluded via `.gitignore`.

---

## File Placement Instructions

1. Create reduced versions of files (see sizes above)
2. Name them exactly as specified:
   - `sample_dump.dump`
   - `sample_log.lammps`
   - `sample.data`
   - `sample_acf.txt` (optional)
3. Place them in: `/Users/bmagyari/Documents/PhD🔬/Simulations/Python Packages/LaDa/tests/data/`
4. Commit to git

The test suite will automatically use these files.

---

## .gitignore Configuration

Once you provide reduced samples, `.gitignore` will be updated to:

```gitignore
# Exclude large original files (not to be committed)
tests/data/Atoms.dump
tests/data/Atoms copy.dump
tests/data/Bonds.dump
tests/data/Vitrimer.data
tests/data/log.lammps

# Keep test data samples (committed to git)
# sample_dump.dump  ← COMMITTED
# sample_log.lammps ← COMMITTED
# sample.data       ← COMMITTED
# sample_acf.txt    ← COMMITTED
```

---

## Testing With Sample Data

Tests will:
1. ✅ Check if sample files exist
2. ✅ Use them for unit testing
3. ✅ Generate synthetic data for edge cases
4. ⚠️ Skip tests if optional files missing
5. ❌ Fail if required files missing (dump, log, data)

---

## Questions?

If file size or format questions arise, the test runner will tell you exactly what's wrong when you try to run tests.

```bash
pytest tests/ -v
# Shows exactly which files are missing or in wrong format
```

---

## Committed fixtures added in September 2026

Three additional fixtures, generated deterministically from committed scripts (seed
`20260908`), support `lada.analysis.dynamics`, `lada.analysis.rheology`, and
`lada.modifiers.data_modifier`, as well as the `lada-web` app tests in `webapp/tests/`:

### **`sample_trajectory.npz`**
**Purpose**: Small synthetic polymer trajectory for `lada.analysis.dynamics` functions
(`calculate_segment_acf_from_trajectory`, `calculate_rouse_mode_acf_from_trajectory`,
`calculate_isf_from_trajectory`) and the web app's dynamics pages.
**Contents**: key `coords`, shape `(32, 32, 3)`, float64 — 4 chains of 8 beads each,
atom index = `chain * 8 + bead` (chain-major, 0-indexed). Frame 0 is a freely-jointed
chain per molecule; each subsequent frame adds an independent Gaussian kick so the
autocorrelation functions decay within the 32 frames.
**Regenerate**: `python tests/data/generate_sample_trajectory.py`

### **`sample_stress_acf.txt`**
**Purpose**: Stress-autocorrelation fixture for `lada.parsers.read_lammps_acf` and
`lada.analysis.calc_stress_relaxation`, which require the exact column names
`ACF_Sxy, ACF_Sxz, ACF_Syz, ACF_Nxy, ACF_Nxz, ACF_Nyz` (unlike `sample_acf.txt` above,
which uses `acf_x/y/z` and is not compatible with `calc_stress_relaxation`).
**Contents**: two timestep blocks (`t=0` reference, then `t=1000`); `read_lammps_acf`
reads the last block — 100 lag rows at `lag_time = 0, 0.5, …, 49.5`. Each ACF column is
a two-exponential decay (fast + slow relaxation) with small seeded noise, held positive.
**Regenerate**: `python tests/data/generate_sample_stress_acf.py`

### **`sample_molecular.data`**
**Purpose**: Molecular-style LAMMPS data fixture for `lada.modifiers.rewrite_end_beads`
(and `lada.parsers.read_data_file` with `atom_style = molecular`).
**Contents**: 3 linear chains of 4 beads each (atom ids 1–4, 5–8, 9–12), 2 atom types,
with `Masses`, `Pair Coeffs`, `Bond Coeffs`, `Angle Coeffs`, `Bonds`, and `Angles`
sections. Written by hand — there is no generator script for this one.

### **`sample_chains_molecular.data`**
**Purpose**: Chain-topology fixture for `lada.analysis.chains_from_data_file` and the web
app's `mode=data_file`. Deliberately sized to `sample_trajectory.npz` — 4 chains of 8 beads,
32 atoms, atom id = array index + 1 — so a data-file run can be asserted *numerically
identical* to the hand-typed uniform form, which is the anchor test for the whole feature.
**Contents**: `Atoms # molecular` (`id molecule type x y z`, coordinates taken from frame 0
of the trajectory), 28 bonds joining consecutive beads within each molecule, one atom type.
Exercises the molecule-id path with bonds available to confirm the bead order.

### **`sample_chains_bonds_only.data`**
**Purpose**: The same 32 atoms with **no molecule column** (`Atoms # atomic`), so the chains
can only come from the bond graph.
**Contents**: the same 28 bonds, written hostile to a naive implementation — the bond lines
are shuffled, roughly half list the higher atom id first, and chain 4's backbone runs
27-25-29-26-31-28-32-30, so sorting its ids yields the wrong head and tail. A reader that
groups correctly but orders by atom id passes on the other fixture and fails on this one.
Note that `sample.data` cannot do this job: its `Bonds` section is synthetic and broken
(bond 50 duplicates bond 49, and it covers only atoms 1–50 of the 100 present).

Both were written by a one-off script against `sample_trajectory.npz`; as with
`sample_molecular.data`, no generator is kept for them.

Guarding tests: `tests/test_fixtures.py` (byte-reproducibility of the two generated
files from their scripts, and schema checks),
`tests/test_modifiers/test_data_modifier.py` (behavioural tests of `rewrite_end_beads`
against `sample_molecular.data`), and `tests/test_analysis/test_topology.py` plus
`webapp/tests/test_dynamics.py` (the two chain-topology fixtures).
