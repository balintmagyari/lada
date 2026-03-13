import pandas as pd
import numpy as np
from icecream import ic
from tqdm import tqdm

# TODO: Run checks on whether the function works as expected
    # Compare results from function to those calculated by LAMMPS
def calculate_avg_rg(df: pd.DataFrame | np.ndarray,
                     columns: list[str] | None = None,
                     coord_cols: list[str] = ['xu', 'yu', 'zu'],
                     molecule_col: str = 'mol',
                     mass_col: str | None = None,
                     ):
    """Compute the ensemble-average radius of gyration for a frame.

    The final return value is the mean of per-molecule Rg values.

    Args:
        df: DataFrame or array containing atomic coordinates and molecule IDs.
        columns: Column names for `df` when passing a NumPy array.
        coord_cols: Names of the x/y/z coordinate columns.
        molecule_col: Name of the molecule ID column.
        mass_col: Name of the mass column (if not provided all atoms are treated as equal mass).

    Returns:
        The mean of per-molecule radius of gyration values.
    """
    if isinstance(df, pd.DataFrame):
        cols = df.columns.tolist()
        arr = df.to_numpy()
    elif isinstance(df, np.ndarray):
        if columns is None:
            raise ValueError("You must provide the 'columns' list when passing a NumPy array.")
        cols = columns
        arr = df
    else:
        raise TypeError("Data must be a pandas.DataFrame or numpy.ndarray.")
    
    try:
        mol_idx = cols.index(molecule_col)
        x_idx = cols.index(coord_cols[0])
        y_idx = cols.index(coord_cols[1])
        z_idx = cols.index(coord_cols[2])
    except ValueError as e:
        raise ValueError(f"Missing a required column for Rg calculation: {e}")
    
    mols = arr[:, mol_idx]
    coords = arr[:, [x_idx, y_idx, z_idx]].astype(float)

    if mass_col:
        if mass_col not in cols:
            raise ValueError(f"Mass column '{mass_col}' not found in data.")
        masses = arr[:, cols.index(mass_col)].astype(float)
    else:
        masses = np.ones(len(arr), dtype=float)

    unique_mols = np.unique(mols)
    rg_squared_values = np.zeros(len(unique_mols))

    for i, mol_id in enumerate(unique_mols):
        mask = (mols == mol_id)
        atom_coords = coords[mask]      # Coordinates of a single molecule's atoms
        atom_masses = masses[mask]      # Masses of a single molecule's individual atoms
        
        total_mass = float(np.sum(atom_masses))
        if total_mass < 1e-6:
            raise ValueError(f"Total mass for molecule {mol_id} is zero.")

        # Calculate Center of Mass for this chain
        com = np.sum(atom_coords * atom_masses[:, np.newaxis], axis=0) / total_mass
        
        # Calculate mass-weighted squared distances from COM
        sq_distances = np.sum((atom_coords - com)**2, axis=1)
        rg_squared = np.sum(atom_masses * sq_distances) / total_mass
        
        rg_squared_values[i] = rg_squared
        
    # 5. Return the ensemble average
    return np.mean(rg_squared_values)

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict

def calculate_avg_rg_sq(df: Union[pd.DataFrame, np.ndarray],
                        columns: Optional[List[str]] = None,
                        coord_cols: List[str] = ['xu', 'yu', 'zu'],
                        molecule_col: str = 'mol',
                        timestep_col: str = 'timestep',
                        mass_col: Optional[str] = None,
                        ) -> Union[float, Dict[float, float]]:
    """Compute the ensemble-average squared radius of gyration.
    
    Dynamically detects if the input is a single frame or a trajectory.
    Returns a float for single frames, or a dict for multiple timesteps.
    """
    # 1. Standardize Input
    if isinstance(df, pd.DataFrame):
        cols = df.columns.tolist()
        arr = df.to_numpy()
    elif isinstance(df, np.ndarray):
        if columns is None:
            raise ValueError("You must provide 'columns' when passing a NumPy array.")
        cols = columns
        arr = df
    else:
        raise TypeError("Data must be a pandas.DataFrame or numpy.ndarray.")
    
    # 2. Extract Essential Columns
    try:
        mol_idx = cols.index(molecule_col)
        x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]
    except ValueError as e:
        raise ValueError(f"Missing a required column: {e}")
    
    mols = arr[:, mol_idx]
    coords = arr[:, [x_idx, y_idx, z_idx]].astype(float)

    if mass_col and mass_col in cols:
        masses = arr[:, cols.index(mass_col)].astype(float)
    else:
        masses = np.ones(len(arr), dtype=float)

    # 3. Detect Data Type (Single Frame vs Trajectory)
    has_timesteps = timestep_col in cols
    if has_timesteps:
        timesteps = arr[:, cols.index(timestep_col)]
    else:
        # Create a dummy array of zeros to force a single loop iteration
        timesteps = np.zeros(len(arr))

    unique_timesteps = np.unique(timesteps)
    rg_sq_by_timestep = {}

    # 4. Compute Rg^2 per frame
    for ts in tqdm(unique_timesteps, disable=len(unique_timesteps) <= 1, desc=r"Calculating $R_g^2$"):
        ts_mask = (timesteps == ts)
        ts_mols = mols[ts_mask]
        ts_coords = coords[ts_mask]
        ts_masses = masses[ts_mask]
        
        unique_mols = np.unique(ts_mols)
        rg_squared_values = np.zeros(len(unique_mols))

        for i, mol_id in enumerate(unique_mols):
            mol_mask = (ts_mols == mol_id)
            atom_coords = ts_coords[mol_mask]      
            atom_masses = ts_masses[mol_mask]      
            
            total_mass = float(np.sum(atom_masses))
            if total_mass < 1e-6:
                raise ValueError(f"Total mass for molecule {mol_id} is zero.")

            # Calculate Center of Mass
            com = np.sum(atom_coords * atom_masses[:, np.newaxis], axis=0) / total_mass
            
            # Calculate mass-weighted squared distances from COM
            sq_distances = np.sum((atom_coords - com)**2, axis=1)
            rg_squared = np.sum(atom_masses * sq_distances) / total_mass
            
            rg_squared_values[i] = rg_squared
            
        rg_sq_by_timestep[ts] = float(np.mean(rg_squared_values))

    # 5. Dynamic Return
    if not has_timesteps or len(unique_timesteps) == 1:
        # If it's a single frame, just return the raw float value
        return list(rg_sq_by_timestep.values())[0]
    else:
        # If it's a trajectory, return the dictionary mapping {timestep: rg2}
        return rg_sq_by_timestep