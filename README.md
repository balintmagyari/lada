# lada

lada (LAMMPS Data Access) is a lightweight Python package for parsing common LAMMPS output formats. It provides simple, streaming-friendly access to dump files, log files, and LAMMPS data files, with minimal dependencies.

---

## 🚀 Installation

Install using `pip` from PyPI:

```bash
pip install lada
```

> Note: The package is designed to be used with Python 3.8+.

---

## 📦 Package structure

The core parser modules are located in `src/lada/parsers/`:

- `dump_parser.py` - streaming parser for LAMMPS dump files (i.e., output from [`dump`](https://docs.lammps.org/dump.html) command) using `iter_dump_frames()`
- `log_parser.py` - parser for LAMMPS log files using `read_lammps_log()`
- `data_parser.py` - parser for LAMMPS data files (i.e., output form [`write_data`](https://docs.lammps.org/write_data.html) command) using `read_data_file()`

To keep imports simple, the package exports the most common entry point:

```python
from lada import iter_dump_frames, read_lammps_log, read_data_file
```

---

## 🧩 1) Parsing LAMMPS dump files (`dump_parser.py`)

### Main API

```python
from lada import iter_dump_frames

for frame in iter_dump_frames("path/to/dump_file.dump"):
    # metadata is a dict of "ITEM:" blocks
    timestep = frame.metadata["TIMESTEP"]          # int
    box_bounds = frame.metadata["BOX BOUNDS pp pp pp"]

    # get numeric columns by name
    ids = frame.get_column("id")
    xs  = frame.get_column("x")

    # convert the atom block into a pandas DataFrame
    df = frame.to_dataframe()
```

### Notes on metadata conversion

- `TIMESTEP` is returned as an `int`.
- `NUMBER OF ...` entries (e.g., `NUMBER OF ATOMS`) are converted to `int`.
- `BOX BOUNDS ...` entries are parsed into numeric values:
  - For orthogonal boxes, you get a 2D NumPy array shape `(3, 2)`.
  - For triclinic boxes, you get a 2D NumPy array shape `(3, 3)` where the 3rd column contains tilt factors (xy, xz, yz).

### Column helpers

These helpers avoid manual index lookups:

```python
ids = frame.get_column("id")
atom_ids = frame.get_column_or("id", default=None)
col_idx = frame.column_index("type")
```

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

### Main API

```python
from lada import read_data_file

lammps_data = read_data_file("data.lmp")

# Get a parsed section as a NumPy array
atoms = lammps_data.get("Atoms")

# Convert a section to pandas (smart infer columns for Atoms/Bonds/Velocities)
df_atoms = lammps_data.to_pandas(section="Atoms")
```

### Notes on atom style detection

- The parser attempts to detect the atom style from the comment in the `Atoms` section header,
  e.g. `Atoms # atomic`.
- If detected, it uses the correct column layout for that style (e.g., `x y z` vs `qx qy qz`).

<!-- ---

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest -q
```

---

## 🎯 Tips

- If you want a consistent representation for `BOX BOUNDS`, you can standardize by converting the array to a known shape:

```python
box_bounds = np.asarray(frame.metadata["BOX BOUNDS pp pp pp"])
if box_bounds.shape == (3, 3):
    bounds = box_bounds[:, :2]
    tilt   = box_bounds[:, 2]
```

- Use `frame.to_dataframe()` for easy plotting and analysis with pandas. -->

---

## License

[MIT](LICENSE)
