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

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        Dataframe containing data to be used to calculate squared radius of gyration.
        It can be given either as a pandas dataframe or a numpy array. In case it is a 
        numpy array, the 'columns' argument is necessary to be specified.
    columns: list[str] | None, default=None
        Column header names in case a numpy array was given for 'df' argument.
    coords_cols: list[str], default=['xu', 'yu', 'zu']
        Column headers specifying the coordinate columns. For a reliable calculation
        these coordinates should be in their 'unwrapped' format, which LAMMPS 
        denotes as 'xu', 'yu', and 'zu' for the three coordinates.
    molecule_col: str, default='mol'
        Column header specifying molecular IDs.
    timestep_col: str, default='timestep'
        Column header specifying timestep values.
    mass_col: str | None, default=None
        Column header specifying atom masses. When given, the mass of each atom is 
        accounted for when calculating Rg^2. If None, the mass of all atoms is 
        considered equal.

    Returns
    -------
    float | dict[float, float]
        Returns a singular float value if a single timestep's data was given in df.
        Otherwise it returns a dictionary where the keys specify the timestep and 
        the values the Rg^2 values.
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
    

def calculate_avg_ree_sq(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] = ['xu', 'yu', 'zu'],
    molecule_col: str = 'mol',
    timestep_col: str = 'timestep',
    atom_id_col: str = 'id'  # Crucial for determining the start/end of the chain
) -> float | dict[float, float]:
    """
    Compute the ensemble-average squared end-to-end distance using fast vectorization.
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
        id_idx = cols.index(atom_id_col)
        x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]
    except ValueError as e:
        raise ValueError(f"Missing a required column for Ree calculation: {e}")
        
    # Extract targeting arrays
    mols = arr[:, mol_idx]
    atom_ids = arr[:, id_idx]
    coords = arr[:, [x_idx, y_idx, z_idx]].astype(float)

    # 2. Detect Data Type (Single Frame vs Trajectory)
    has_timesteps = timestep_col in cols
    if has_timesteps:
        timesteps = arr[:, cols.index(timestep_col)]
    else:
        timesteps = np.zeros(len(arr))

    unique_timesteps = np.unique(timesteps)
    ree_sq_by_timestep = {}

    # 3. Compute Ree^2 per frame using tqdm
    for ts in tqdm(unique_timesteps, disable=len(unique_timesteps) <= 1, desc="Calculating Ree^2"):
        # Isolate the frame
        ts_mask = (timesteps == ts)
        ts_mols = mols[ts_mask]
        ts_atom_ids = atom_ids[ts_mask]
        ts_coords = coords[ts_mask]
        
        # Sort atoms first by molecule ID, then by atom ID 
        # This ensures the first and last indices correspond to the chain ends
        sort_keys = np.lexsort((ts_atom_ids, ts_mols))
        sorted_mols = ts_mols[sort_keys]
        sorted_coords = ts_coords[sort_keys]

        # Find the starting index of every new molecule in the sorted array
        _, start_indices = np.unique(sorted_mols, return_index=True)
        
        # The end index of a molecule is the start index of the next one minus 1.
        # The very last molecule ends at the last index of the array.
        end_indices = np.append(start_indices[1:] - 1, len(sorted_mols) - 1)

        # Extract coordinates for the first and last atoms of each molecule
        start_coords = sorted_coords[start_indices]
        end_coords = sorted_coords[end_indices]

        # Calculate squared end-to-end distance per molecule: (x_end - x_start)^2 + ...
        sq_distances = np.sum((end_coords - start_coords)**2, axis=1)
        
        # Store the ensemble average for this timestep
        ree_sq_by_timestep[ts] = float(np.mean(sq_distances))

    # 4. Dynamic Return
    if not has_timesteps or len(unique_timesteps) == 1:
        return list(ree_sq_by_timestep.values())[0]
    else:
        return ree_sq_by_timestep
    

def calculate_ree_vectors(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] = ['xu', 'yu', 'zu'],
    molecule_col: str = 'mol',
    timestep_col: str = 'timestep',
    atom_id_col: str = 'id'
) -> pd.DataFrame:
    """
    Compute the individual end-to-end vectors for each molecule at each timestep.
    Returns a DataFrame containing the timestep, molecule ID, and vector components (dx, dy, dz).
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
        id_idx = cols.index(atom_id_col)
        x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]
    except ValueError as e:
        raise ValueError(f"Missing a required column for Ree vector calculation: {e}")
        
    # Extract targeting arrays
    mols = arr[:, mol_idx]
    atom_ids = arr[:, id_idx]
    coords = arr[:, [x_idx, y_idx, z_idx]].astype(float)

    # 2. Detect Data Type (Single Frame vs Trajectory)
    has_timesteps = timestep_col in cols
    if has_timesteps:
        timesteps = arr[:, cols.index(timestep_col)]
    else:
        timesteps = np.zeros(len(arr))

    unique_timesteps = np.unique(timesteps)
    all_frames_data = []

    # 3. Compute Ree vectors per frame using tqdm
    for ts in tqdm(unique_timesteps, disable=len(unique_timesteps) <= 1, desc="Calculating Ree vectors"):
        # Isolate the frame
        ts_mask = (timesteps == ts)
        ts_mols = mols[ts_mask]
        ts_atom_ids = atom_ids[ts_mask]
        ts_coords = coords[ts_mask]
        
        # Sort atoms first by molecule ID, then by atom ID 
        sort_keys = np.lexsort((ts_atom_ids, ts_mols))
        sorted_mols = ts_mols[sort_keys]
        sorted_coords = ts_coords[sort_keys]

        # Find the starting and ending indices of every molecule
        _, start_indices = np.unique(sorted_mols, return_index=True)
        end_indices = np.append(start_indices[1:] - 1, len(sorted_mols) - 1)

        # Extract coordinates for the first and last atoms
        start_coords = sorted_coords[start_indices]
        end_coords = sorted_coords[end_indices]

        # Calculate the vector components: dx, dy, dz
        ree_vectors = end_coords - start_coords
        
        # Extract the unique molecule IDs corresponding to these vectors
        unique_mol_ids = sorted_mols[start_indices]
        
        # Create an array of the current timestep to match the length of our results
        ts_array = np.full(len(unique_mol_ids), ts)
        
        # Stack everything together into a 2D array for this frame
        frame_data = np.column_stack((ts_array, unique_mol_ids, ree_vectors))
        all_frames_data.append(frame_data)

    # 4. Compile and Return Data
    # Vertically stack all the frame data blocks
    final_array = np.vstack(all_frames_data)
    
    # Convert to a DataFrame for clean downstream handling
    result_df = pd.DataFrame(
        final_array, 
        columns=[timestep_col, molecule_col, 'dx', 'dy', 'dz']
    )
    
    # Cast IDs back to integers (column_stack forces floats if coords are floats)
    result_df[timestep_col] = result_df[timestep_col].astype(int)
    result_df[molecule_col] = result_df[molecule_col].astype(int)

    if not has_timesteps:
        result_df = result_df.drop(columns=timestep_col)
    
    return result_df