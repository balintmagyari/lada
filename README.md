# LaDa

**LaDa** (LAMMPS Data Access) is a lightweight Python package for parsing common LAMMPS output formats. The name is, quite intentionally, borrowed from the legendary Soviet car brand **LADA**—because much like its cars, this library aims to be simple, reliable, and able to run just about anywhere without unnecessary luxury features.

In its current state, the library provides straightforward, user-friendly access to LAMMPS dump files, log files, and data files, while keeping external dependencies to a minimum.

Future development will expand the package beyond parsing, adding analysis tools that operate directly on the retrieved simulation data.

Like a classic LADA: it may not come with heated seats or chrome trim, but it will get your data from point A to point B without complaint.

---

## 🚀 Installation

Install using `pip` from PyPI:

```bash
pip install lada
```

> Note: The package is designed to be used with Python 3.12+.

---

## 📦 Package structure

The core parser modules are located in `src/lada/parsers/`:

- `dump_parser.py` - streaming parser for LAMMPS dump files (i.e., output from [`dump`](https://docs.lammps.org/dump.html) command) using `dump_frames()`
- `log_parser.py` - parser for LAMMPS log files using `read_lammps_log()`
- `data_parser.py` - parser for LAMMPS data files (i.e., output from [`write_data`](https://docs.lammps.org/write_data.html) command) using `read_data_file()`

To keep imports simple, the package exports the most common entry point:

```python
from lada import dump_frames, read_lammps_log, read_data_file
```

---

## 🧩 1) Parsing LAMMPS dump files (`dump_parser.py`)

### Iterate through frames from dump file

```python
from lada import dump_frames

for frame in dump_frames("path/to/dump_file.dump"):
    # metadata is a dict of "ITEM:" blocks before main data
    timestep = frame.metadata["TIMESTEP"]          # int
    box_bounds = frame.metadata["BOX BOUNDS pp pp pp"]

    # get numeric columns by name
    ids = frame.get_column("id")
    xs  = frame.get_column("x")

    # convert the main data block into a pandas DataFrame
    df = frame.to_pandas()
```

> This functionality is useful for performing calculations iteratively on 
data belonging to individual timesteps (e.g. radius of gyration). However, 
for calculations that require more than one timestep's data, the `read_dump`
function is more appropriate, the usage of which is [described below](#read-entire-main-data-from-dump-file).

#### Notes on metadata conversion

- `TIMESTEP` is returned as an `int`.
- `NUMBER OF ...` entries (e.g., `NUMBER OF ATOMS`) are converted to `int`.
- `BOX BOUNDS ...` entries are parsed into numeric values:
  - For orthogonal boxes, you get a 2D NumPy array shape `(3, 2)`.
  - For triclinic boxes, you get a 2D NumPy array shape `(3, 3)` where the 3rd column contains tilt factors (xy, xz, yz).

#### Column helpers

These helpers avoid manual index lookups:

```python
ids = frame.get_column("id")
atom_ids = frame.get_column_or("id", default=None)
col_idx = frame.column_index("type")
```

### Read entire main data from dump file

```python
from lada import read_dump

df = read_dump("path/to/dump_file.dump", timestep_col="Timestep") # value of 'timestep_col' determines the column name of the timesteps data
```

#### Notes on usage

While the previously described `dump_frames` function reads not only 
the main data from the dump file but also the metadata written at the 
beginning of each block, the `read_dump` function does not save any of
the metadata information other than the current timestep, whose data
is appended to the final dataframe as a separate column next to the 
bulk data read from the dump file.

---

## 📝 2) Parsing LAMMPS log files (`log_parser.py`)

### Main API

```python
from lada import read_lammps_log

thermo = read_lammps_log("log.lammps")
print(thermo.columns)        # column names (Step, Temp, E_pair, ...)
print(thermo.get("E_pair")) # numpy array of E_pair values

# Convert to pandas DataFrame
df = thermo.to_pandas()
```

**What it parses**
- It extracts the table between the `Per MPI rank memory allocation` marker and the `Loop time` marker.
- The first non-empty line in that section is treated as the header.

---

## 🧬 3) Parsing LAMMPS data files (`data_parser.py`)

### Regular LAMMPS data files

#### Main API

```python
from lada import read_data_file

lammps_data = read_data_file("data.lmp")

# Get a parsed section as a NumPy array
atoms = lammps_data.get("Atoms")

# Convert a section to pandas (smart infer columns for Atoms/Bonds/Velocities)
df_atoms = lammps_data.to_pandas(section="Atoms")
```

#### Notes on atom style detection

- The parser attempts to detect the atom style from the comment in the `Atoms` section header,
  e.g. `Atoms # atomic`.
- If detected, it uses the correct column layout for that style (e.g., `x y z` vs `qx qy qz`).

### Autocorrelation data files

These files refer to files written using LAMMPS' `fix ave/correlate/long` command ([LAMMPS
documentation](https://docs.lammps.org/fix_ave_correlate_long.html)). 

#### Main API

```python
from lada import read_lammps_acf

df = read_lammps_acf('data.txt')
```

The output from the `read_lammps_acf` function is a standard pandas dataframe.

---

## License

[MIT](LICENSE)
