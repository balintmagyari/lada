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
