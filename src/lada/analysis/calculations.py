import pandas as pd
import numpy as np
from icecream import ic
from tqdm import tqdm
# from typing import Union, List, Optional, Dict

def calculate_avg_rg_sq(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] = ['xu', 'yu', 'zu'],
    molecule_col: str = 'mol',
    timestep_col: str = 'timestep',
    mass_col: str | None = None,
) -> float | dict[float, float]:
    """
    Compute the ensemble-average squared radius of gyration using fast vectorization.
    Dynamically handles both single frames and massive multi-timestep trajectories.
    """
    
    # 1. Standardize input and validate columns
    if isinstance(df, pd.DataFrame):
        cols = df.columns.tolist()
        arr = df.to_numpy()
    elif isinstance(df, np.ndarray):
        if columns is None:
            raise ValueError("Must provide 'columns' list when passing a NumPy array.")
        cols = columns
        arr = df
    else:
        raise TypeError("Data must be a pandas.DataFrame or numpy.ndarray.")
    
    # Extract column indices
    try:
        mol_idx = cols.index(molecule_col)
        x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]
    except ValueError as e:
        raise ValueError(f"Missing a required column for Rg calculation: {e}")
        
    # Extract targeting arrays and force float typing for math
    mols = arr[:, mol_idx]
    coords = arr[:, [x_idx, y_idx, z_idx]].astype(float)
    
    # Safely handle masses
    if mass_col:
        if mass_col not in cols:
            raise ValueError(f"Mass column '{mass_col}' not found in data.")
        masses = arr[:, cols.index(mass_col)].astype(float)
    else:
        masses = np.ones(len(arr), dtype=float)

    # 2. Detect Data Type (Single Frame vs Trajectory)
    has_timesteps = timestep_col in cols
    if has_timesteps:
        timesteps = arr[:, cols.index(timestep_col)]
    else:
        timesteps = np.zeros(len(arr))

    unique_timesteps = np.unique(timesteps)
    rg_sq_by_timestep = {}

    # 3. Compute Rg^2 per frame using tqdm
    for ts in tqdm(unique_timesteps, disable=len(unique_timesteps) <= 1, desc="Calculating Rg^2"):
        # Isolate the frame
        ts_mask = (timesteps == ts)
        ts_mols = mols[ts_mask]
        ts_coords = coords[ts_mask]
        ts_masses = masses[ts_mask]
        
        # Vectorized Rg^2 Calculation
        _, mol_indices = np.unique(ts_mols, return_inverse=True)
        mol_masses = np.bincount(mol_indices, weights=ts_masses)
        mol_masses = np.where(mol_masses < 1e-6, 1.0, mol_masses)
        
        com_x = np.bincount(mol_indices, weights=ts_coords[:, 0] * ts_masses) / mol_masses
        com_y = np.bincount(mol_indices, weights=ts_coords[:, 1] * ts_masses) / mol_masses
        com_z = np.bincount(mol_indices, weights=ts_coords[:, 2] * ts_masses) / mol_masses
        
        atom_com_x = com_x[mol_indices]
        atom_com_y = com_y[mol_indices]
        atom_com_z = com_z[mol_indices]
        
        sq_distances = (ts_coords[:, 0] - atom_com_x)**2 + \
                       (ts_coords[:, 1] - atom_com_y)**2 + \
                       (ts_coords[:, 2] - atom_com_z)**2
                       
        rg_sq_per_mol = np.bincount(mol_indices, weights=ts_masses * sq_distances) / mol_masses
        rg_sq_by_timestep[ts] = float(np.mean(rg_sq_per_mol))

    # 4. Dynamic Return
    if not has_timesteps or len(unique_timesteps) == 1:
        return list(rg_sq_by_timestep.values())[0]
    else:
        return rg_sq_by_timestep