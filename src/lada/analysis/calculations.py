import pandas as pd
import numpy as np
from numpy.typing import NDArray
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

def calculate_segment_acf(trajectory_file: str, 
                          segment_pairs: np.ndarray, 
                          time_per_frame: float
                          ) -> NDArray[np.float64]:
    """
    Calculates the normalized end-to-end vector autocorrelation function (ACF) 
    for polymer chains from a compressed NumPy trajectory.

    This function computes the ensemble-averaged, time-correlated memory loss of 
    the polymer chain conformations. It uses a highly vectorized sliding-window 
    approach to calculate the dot product <R(t) . R(0)> across all valid time 
    origins (t0) and averages the result over all specified chains in the system.

    Parameters
    ----------
    trajectory_file : str
        Path to the compressed NumPy archive (.npz) containing the simulation 
        trajectory. The archive must contain an array accessed via the 'coords' 
        keyword, structured with shape (n_frames, n_atoms, 3).
    end_pairs : numpy.ndarray
        A 2D array of integers of shape (n_chains, 2) containing the 0-indexed 
        indices of the head and tail beads for every polymer chain in the melt. 
        Column 0 corresponds to the head indices, and Column 1 corresponds to 
        the tail indices.
    time_per_frame : float
        The physical simulation time elapsed between consecutive saved frames 
        in the trajectory (e.g., MD_timestep * steps_between_saves).

    Returns
    -------
    numpy.ndarray
        A 2D array of floats of shape (n_frames, 2) containing the final data:
        - Column 0: The physical lag time (delta t) for the correlation window.
        - Column 1: The normalized autocorrelation value, C(t). The value at 
                    C(0) is strictly normalized to 1.0.

    Notes
    -----
    Statistical reliability decreases linearly as the lag time approaches the 
    total trajectory length due to the decreasing number of available time 
    origins for the ensemble average. It is standard practice to discard or 
    visually ignore the final 10% to 20% of the returned array when extracting 
    terminal relaxation times.
    """
    data = np.load(trajectory_file, allow_pickle=False)
    coords = data['coords']  # Shape: (n_frames, n_atoms, 3)
    
    # Extract the head and tail indices
    heads = segment_pairs[:, 0]
    tails = segment_pairs[:, 1]
    
    # Calculate the end-to-end vector for ALL chains at ALL frames
    R = coords[:, tails, :] - coords[:, heads, :]
    
    n_frames, n_chains, _ = R.shape
    acf = np.zeros(n_frames)
    
    # Slide the time window (lag) across the trajectory
    for lag in tqdm(range(n_frames), "Calculating autocorrelation"):
        R_start = R[:n_frames - lag]
        R_lag = R[lag:]
        
        # Calculate the dot product and average
        dot_products = np.sum(R_start * R_lag, axis=2)
        acf[lag] = np.mean(dot_products)
        
    # Normalize the function
    acf_normalized = acf / acf[0]
    
    # --- Generate Time Axis and Stack ---
    # Create an array of physical time values starting at 0
    time_array = np.arange(n_frames) * time_per_frame
    
    # Stack the time and ACF arrays into a 2D matrix of shape (n_frames, 2)
    final_output = np.column_stack((time_array, acf_normalized))
    
    return final_output

def calculate_rouse_mode_acf(trajectory_file: str, 
                             chain_indices: np.ndarray, 
                             p: int, 
                             time_per_frame: float
                             ) -> NDArray[np.float64]:
    """
    Calculates the normalized autocorrelation function (ACF) for a specific 
    Rouse mode of polymer chains from a compressed NumPy trajectory.

    This function isolates independent, orthogonal harmonic motions (modes) of 
    a polymer chain using a discrete cosine transform. It projects the 3D bead 
    coordinates into mode space to find the mode amplitude X_p(t), and then 
    computes the ensemble-averaged time-correlation <X_p(t) . X_p(0)> using a 
    vectorized sliding-window approach.

    Parameters
    ----------
    trajectory_file : str
        Path to the compressed NumPy archive (.npz) containing the simulation 
        trajectory. The archive must contain an array accessed via the 'coords' 
        keyword, structured with shape (n_frames, n_atoms, 3).
    chain_indices : numpy.ndarray
        A 2D array of integers of shape (n_chains, beads_per_chain) containing 
        the 0-indexed indices of every bead in every polymer chain. The beads 
        must be sequentially ordered from one end of the chain to the other 
        along the backbone.
    p : int
        The Rouse mode number to calculate. Must be an integer bounded by 
        0 <= p < N, where N is the number of beads per chain. 
        - p=0 returns the center-of-mass translation (does not decay to zero).
        - p=1 returns the fundamental mode (whole chain relaxation).
        - Higher p values isolate increasingly localized segmental vibrations.
    time_per_frame : float
        The physical simulation time elapsed between consecutive saved frames 
        in the trajectory.

    Returns
    -------
    numpy.ndarray
        A 2D array of floats of shape (n_frames, 2) containing the final data:
        - Column 0: The physical lag time (delta t) for the correlation window.
        - Column 1: The normalized autocorrelation value, C(t). The value at 
                    C(0) is strictly normalized to 1.0.

    Raises
    ------
    ValueError
        If the requested mode `p` is negative or greater than or equal to the 
        number of beads per chain (N), violating the Nyquist limit for discrete 
        polymer representations.

    Notes
    -----
    Statistical reliability decreases linearly as the lag time approaches the 
    total trajectory length. It is standard practice to discard the terminal 
    10% to 20% of the returned array when extracting relaxation times (tau_p).
    """
    data = np.load(trajectory_file, allow_pickle=False)
    coords = data['coords']  # Shape: (n_frames, n_atoms, 3)
    
    # 1. Isolate the chains. 
    # chain_coords shape becomes: (n_frames, n_chains, beads_per_chain, 3)
    chain_coords = coords[:, chain_indices, :] 
    
    n_frames, n_chains, N, _ = chain_coords.shape

    # ---------------------------------------------------------
    # VALIDATION CHECK: Enforce the physical limits of p
    # ---------------------------------------------------------
    if not (0 <= p < N):
        raise ValueError(
            f"Mathematical constraint violated! You requested mode p={p}. "
            f"For a chain containing N={N} beads, the Rouse mode number 'p' "
            f"must be an integer strictly between 0 and {N-1}."
        )
    # ---------------------------------------------------------
    
    # 2. Build the cosine weights
    # We use 1-based indexing for the math formula, so n goes from 1 to N
    n_array = np.arange(1, N + 1)
    
    # Calculate the cosine term for each bead
    weights = np.cos(p * np.pi * (n_array - 0.5) / N)
    
    # Reshape weights to (1, 1, N, 1) so it broadcasts perfectly against chain_coords
    weights = weights.reshape(1, 1, N, 1)
    
    # 3. Calculate the Rouse mode amplitudes X_p(t)
    # Multiply the coordinates by the weights, then sum across the bead axis (axis=2)
    # Resulting X_p shape: (n_frames, n_chains, 3)
    X_p = np.sum(chain_coords * weights, axis=2) 
    
    acf = np.zeros(n_frames)
    
    # 4. Sliding time window (identical to the end-to-end ACF)
    for lag in tqdm(range(n_frames), desc=f"Mode {p} ACF"):
        X_start = X_p[:n_frames - lag]
        X_lag = X_p[lag:]
        
        dot_products = np.sum(X_start * X_lag, axis=2)
        acf[lag] = np.mean(dot_products)
        
    # 5. Normalize and Format
    acf_normalized = acf / acf[0]
    time_array = np.arange(n_frames) * time_per_frame
    
    return np.column_stack((time_array, acf_normalized))

def _generate_q_vectors(q_magnitude: float, n_vectors: int = 50) -> np.ndarray:
    """
    Generates an array of 3D scattering vectors uniformly distributed on the 
    surface of a sphere of radius |q| using a Fibonacci lattice.

    Parameters
    ----------
    q_magnitude : float
        The desired magnitude |q| of the scattering vectors. This defines the 
        physical length scale being probed (q = 2*pi / d).
    n_vectors : int, optional
        The number of vectors to generate for isotropic averaging. Default is 50.
        Higher numbers give smoother averages but linearly increase the 
        computational time of the ISF calculation.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_vectors, 3) containing the [qx, qy, qz] components.
    """
    # The golden angle in radians
    phi = np.pi * (3.0 - np.sqrt(5.0)) 
    
    # Array of indices from 0 to n_vectors - 1
    indices = np.arange(n_vectors)
    
    # Calculate the Z coordinates (evenly spaced from 1 down to -1)
    # We use n_vectors - 1 to ensure we hit the exact poles if n > 1
    z = 1.0 - (indices / float(n_vectors - 1)) * 2.0
    
    # Calculate the radius of the slice at each Z coordinate
    radius = np.sqrt(1.0 - z * z)
    
    # Calculate the angle theta for each point using the golden angle
    theta = phi * indices
    
    # Calculate X and Y coordinates
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    
    # Stack them together into a (N, 3) array
    unit_vectors = np.column_stack((x, y, z))
    
    # Scale the unit vectors by the requested physical magnitude
    return unit_vectors * q_magnitude

def calculate_isf(trajectory_file: str, 
                  time_per_frame: float,
                  q_magnitude: float, 
                  n_vectors: int = 50
                  ) -> NDArray[np.float64]:
    """
    Calculates the coherent Intermediate Scattering Function, F(q,t), 
    for a specific scattering vector magnitude |q|.

    This function uses the density fluctuation autocorrelation method to achieve 
    O(N) scaling. It evaluates the time-dependent memory loss of spatial 
    density waves across the simulation box.

    Parameters
    ----------
    trajectory_file : str
        Path to the compressed NumPy archive (.npz) containing the coordinates.
    time_per_frame : float
        The physical simulation time elapsed between saved frames.
    q_magnitude : float
        The desired magnitude |q| of the scattering vectors. This defines the 
        physical length scale being probed (q = 2*pi / d).
    n_vectors : int, default=50
        The number of vectors to dynamically generate for isotropic averaging 
        using a Fibonacci lattice. Default is 50. Higher numbers give smoother 
        averages but linearly increase computational time.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_frames, 2):
        - Column 0: Lag time (delta t).
        - Column 1: The normalized ISF value, F(q,t) / F(q,0).
    """
    data = np.load(trajectory_file, allow_pickle=False)
    coords = data['coords']  # Shape: (n_frames, n_atoms, 3)

    q_vectors = _generate_q_vectors(q_magnitude, n_vectors)
    
    n_frames, n_atoms, _ = coords.shape
    n_q = q_vectors.shape[0]
    
    # We will accumulate the ACF for all q-vectors to average at the end
    avg_acf = np.zeros(n_frames)
    
    # Loop over each q-vector in the provided list
    for q_idx in tqdm(range(n_q), "Calculating ISF for vectors"):
        q_vec = q_vectors[q_idx]  # Shape: (3,)
        
        # 1. Calculate the phase for every atom at every frame: q . r(t)
        # coords shape: (n_frames, n_atoms, 3) dot (3,) -> (n_frames, n_atoms)
        phases = np.dot(coords, q_vec)
        
        # 2. Calculate the density fluctuation rho(q, t)
        # Sum the complex exponentials across all atoms (axis=1)
        # Resulting rho shape: (n_frames,)
        rho = np.sum(np.exp(-1j * phases), axis=1)
        
        # 3. Calculate the autocorrelation of rho using the sliding window
        acf_q = np.zeros(n_frames, dtype=np.complex128)
        
        for lag in range(n_frames):
            rho_start = rho[:n_frames - lag]
            rho_lag = rho[lag:]
            
            # Multiply rho(t+lag) by the complex conjugate of rho(t)
            correlations = rho_lag * np.conj(rho_start)
            acf_q[lag] = np.mean(correlations)
            
        # The true physical ISF should be purely real. Any imaginary component 
        # is numerical noise that vanishes upon averaging. We take the real part.
        avg_acf += np.real(acf_q) / n_atoms
        
    # Average across all the q-vectors provided
    avg_acf /= n_q
    
    # Normalize the function so F(q, 0) = 1.0
    isf_normalized = avg_acf / avg_acf[0]
    
    # Generate time axis and format output
    time_array = np.arange(n_frames) * time_per_frame
    final_output = np.column_stack((time_array, isf_normalized))
    
    return final_output

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
 
_SHEAR_COLS = ("ACF_Sxy", "ACF_Sxz", "ACF_Syz")
 
# Accept both the correct name and the known typo from the LAMMPS script
_NORMAL_COLS_CANONICAL = ("ACF_Nxy", "ACF_Nxz", "ACF_Nyz")
_NORMAL_COLS_TYPO      = ("ACF_Nxy", "ACF_Nxz", "ACF_yz")
 
 
def _resolve_normal_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Return the actual normal-stress-difference column names present in *df*,
    tolerating the known 'ACF_yz' typo for 'ACF_Nyz'.
 
    Raises
    ------
    KeyError
        If neither the canonical nor the typo variant of any column is found.
    """
    resolved = []
    for canonical, typo in zip(_NORMAL_COLS_CANONICAL, _NORMAL_COLS_TYPO):
        if canonical in df.columns:
            resolved.append(canonical)
        elif typo in df.columns:
            resolved.append(typo)
        else:
            raise KeyError(
                f"Could not find normal-stress ACF column '{canonical}' "
                f"(also tried '{typo}') in the DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
    return tuple(resolved)
 
def _validate_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following expected columns are missing from the DataFrame: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def calc_stress_relaxation(df: pd.DataFrame, 
                           volume: float, 
                           temperature: float, 
                           lag_col: str = "lag_time",
                           kB: float = 1.0) -> pd.DataFrame:
    """
    Calculate the stress relaxation modulus G(t) from a LAMMPS ACF DataFrame
    using the Green-Kubo (GK) and Full Stress Relaxation (FSR) methods.
 
    Assumes LJ units (k_B = 1), so the prefactor is simply V / T.
 
    Parameters
    ----------
    df : pd.DataFrame
        ACF DataFrame as returned by ``read_lammps_acf``.  Must contain
        columns for the shear ACFs (ACF_Sxy, ACF_Sxz, ACF_Syz) and the
        normal stress difference ACFs (ACF_Nxy, ACF_Nxz, ACF_Nyz — or the
        known typo variant ACF_yz for the last one).
    volume : float
        System volume in LJ units (length^3).
    temperature : float
        System temperature in LJ units (energy).
    lag_col : str, optional
        Name of the lag-time column in *df* (default: ``'lag_time'``).
 
    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns:
          - ``lag_time``  : float - physical lag time (copied from *df*)
          - ``G_GK``      : float - Green-Kubo relaxation modulus
          - ``G_FSR``     : float - Full stress relaxation modulus
 
    Raises
    ------
    KeyError
        If any required ACF column is absent from *df*.
    ValueError
        If *volume* or *temperature* are non-positive, or if *lag_col* is
        not found in *df*.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if volume <= 0:
        raise ValueError(f"volume must be positive, got {volume}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if lag_col not in df.columns:
        raise ValueError(
            f"Lag-time column '{lag_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
 
    _validate_columns(df, _SHEAR_COLS)
    n_xy, n_xz, n_yz = _resolve_normal_cols(df)
 
    # ------------------------------------------------------------------
    # Prefactor  (V / k_B T),  k_B = 1 in LJ units
    # ------------------------------------------------------------------
    prefactor = volume / (temperature * kB)
 
    # ------------------------------------------------------------------
    # Green-Kubo
    # G_GK(t) = (V/T*kB) * (1/3) * ( ACF_Sxy + ACF_Sxz + ACF_Syz )
    # ------------------------------------------------------------------
    shear_mean = (
        df["ACF_Sxy"] + df["ACF_Sxz"] + df["ACF_Syz"]
    ) / 3.0
 
    G_GK = prefactor * shear_mean
 
    # ------------------------------------------------------------------
    # Full Stress Relaxation
    # G_FSR(t) = (V/T) * (1/5) * [   ACF_Sxy  + ACF_Sxz  + ACF_Syz
    #                               + (1/2)*ACF_Nxy
    #                               + (1/2)*ACF_Nxz
    #                               + (1/2)*ACF_Nyz ]
    # ------------------------------------------------------------------
    fsr_sum = (
          df["ACF_Sxy"]
        + df["ACF_Sxz"]
        + df["ACF_Syz"]
        + 0.5 * df[n_xy]
        + 0.5 * df[n_xz]
        + 0.5 * df[n_yz]
    )
 
    G_FSR = prefactor * fsr_sum / 5.0
 
    # ------------------------------------------------------------------
    # Assemble output DataFrame
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "lag_time" : df[lag_col].values,
        "G_GK"     : G_GK.values,
        "G_FSR"    : G_FSR.values,
    })
 
    return result