from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


def _require_columns(cols: list, required: list[str], calculation: str) -> None:
    """Raise a ValueError naming every required column the data lacks."""
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"Missing a required column for {calculation} calculation: {', '.join(missing)}. "
            f"The data has: {', '.join(str(c) for c in cols)}."
        )


def _chain_mask(frame_mols: np.ndarray) -> np.ndarray:
    """Atoms of one frame that belong to a real chain.

    Molecule ID 0 is what LAMMPS gives atoms that belong to no molecule (solvent, ions,
    walls), and a one-atom molecule has no conformation to speak of: its Rg is 0 and its two
    "ends" are the same atom. Counting either would drag the chain average towards zero, so
    only molecules with a non-zero ID and at least two atoms are kept.
    """
    unique_mols, inverse, counts = np.unique(frame_mols, return_inverse=True, return_counts=True)
    keep = (unique_mols != 0) & (counts >= 2)
    return keep[inverse]


def _no_chains_error(ts, has_timesteps: bool, molecule_col: str) -> ValueError:
    where = f" at timestep {ts}" if has_timesteps else ""
    return ValueError(
        f"No polymer chains were found{where}: every atom has molecule ID 0 or is the only "
        f"atom of its molecule in column '{molecule_col}'. Rg and Ree are computed only for "
        "molecules with a non-zero ID and at least two atoms."
    )


def calculate_avg_rg_sq(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] | None = None,
    molecule_col: str = "mol",
    timestep_col: str = "timestep",
    mass_col: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> float | dict[float, float]:
    """
    Compute the ensemble-average squared radius of gyration using fast vectorization.
    Dynamically handles both single frames and massive multi-timestep trajectories.

    Only real chains are averaged: atoms with molecule ID 0 (no molecule) and molecules
    of a single atom are left out. A ValueError is raised if a frame has no chain left.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        Dataframe containing data to be used to calculate squared radius of gyration.
        It can be given either as a pandas dataframe or a numpy array. In case it is a
        numpy array, the 'columns' argument is necessary to be specified.
    columns: list[str] | None, default=None
        Column header names in case a numpy array was given for 'df' argument.
    coord_cols: list[str] | None, default=None
        Column headers specifying the coordinate columns. Defaults to
        ``['xu', 'yu', 'zu']`` when not provided. For a reliable calculation
        these should be the unwrapped coordinates, which LAMMPS denotes as
        ``xu``, ``yu``, and ``zu``.
    molecule_col: str, default='mol'
        Column header specifying molecular IDs.
    timestep_col: str, default='timestep'
        Column header specifying timestep values.
    mass_col: str | None, default=None
        Column header specifying atom masses. When given, the mass of each atom is
        accounted for when calculating Rg^2. If None, the mass of all atoms is
        considered equal.
    progress_callback: Callable[[int, int], None] | None, default=None
        If given, called as ``progress_callback(current, total)`` once per
        timestep processed, with ``current`` 1-based, so the final call is
        always ``(total, total)``. Default is None (no progress reporting).

    Returns
    -------
    float | dict[float, float]
        Returns a singular float value if a single timestep's data was given in df.
        Otherwise it returns a dictionary where the keys specify the timestep and
        the values the Rg^2 values.
    """

    if coord_cols is None:
        coord_cols = ["xu", "yu", "zu"]

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
    _require_columns(cols, [molecule_col, *coord_cols], "Rg")
    mol_idx = cols.index(molecule_col)
    x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]

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
    total_timesteps = len(unique_timesteps)
    rg_sq_by_timestep = {}

    # 3. Compute Rg^2 per frame using tqdm
    for i, ts in enumerate(
        tqdm(unique_timesteps, disable=total_timesteps <= 1, desc="Calculating Rg^2")
    ):
        # Isolate the frame
        ts_mask = timesteps == ts
        ts_mols = mols[ts_mask]
        ts_coords = coords[ts_mask]
        ts_masses = masses[ts_mask]

        # Keep only real chains (mol != 0, at least two atoms)
        chains = _chain_mask(ts_mols)
        if not chains.any():
            raise _no_chains_error(ts, has_timesteps, molecule_col)
        ts_mols = ts_mols[chains]
        ts_coords = ts_coords[chains]
        ts_masses = ts_masses[chains]

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

        sq_distances = (
            (ts_coords[:, 0] - atom_com_x) ** 2
            + (ts_coords[:, 1] - atom_com_y) ** 2
            + (ts_coords[:, 2] - atom_com_z) ** 2
        )

        rg_sq_per_mol = np.bincount(mol_indices, weights=ts_masses * sq_distances) / mol_masses
        rg_sq_by_timestep[ts] = float(np.mean(rg_sq_per_mol))
        if progress_callback is not None:
            progress_callback(i + 1, total_timesteps)

    # 4. Dynamic Return
    # Keyed by timestep whenever the data carried a timestep column, even for a single frame:
    # collapsing that case to a bare float threw away which timestep it was, and callers that
    # rebuild a table then had to invent one (they used 0).
    if not has_timesteps:
        return list(rg_sq_by_timestep.values())[0]
    else:
        return rg_sq_by_timestep


def calculate_avg_ree_sq(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] | None = None,
    molecule_col: str = "mol",
    timestep_col: str = "timestep",
    atom_id_col: str = "id",
    progress_callback: Callable[[int, int], None] | None = None,
) -> float | dict[float, float]:
    """
    Compute the ensemble-average squared end-to-end distance using fast vectorization.
    Dynamically handles both single frames and multi-timestep trajectories.

    Only real chains are averaged: atoms with molecule ID 0 (no molecule) and molecules
    of a single atom are left out. A ValueError is raised if a frame has no chain left.

    Chain ends are identified as the minimum and maximum atom ID within each molecule,
    so unwrapped coordinates (``xu``, ``yu``, ``zu``) should be used to avoid
    periodic-boundary artifacts.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        Data for the calculation. Can be a pandas DataFrame or a numpy array.
        When a numpy array is passed, ``columns`` must also be provided.
    columns : list[str] | None, default=None
        Column header names required when ``df`` is a numpy array.
    coord_cols : list[str] | None, default=None
        Column headers for the coordinate columns. Defaults to
        ``['xu', 'yu', 'zu']`` when not provided.
    molecule_col : str, default='mol'
        Column header specifying molecule IDs.
    timestep_col : str, default='timestep'
        Column header specifying timestep values.
    atom_id_col : str, default='id'
        Column header specifying atom IDs. Used to identify the first and last
        atom in each chain (by minimum and maximum ID within a molecule).
    progress_callback : Callable[[int, int], None] | None, default=None
        If given, called as ``progress_callback(current, total)`` once per
        timestep processed, with ``current`` 1-based, so the final call is
        always ``(total, total)``. Default is None (no progress reporting).

    Returns
    -------
    float | dict[float, float]
        A single float for single-timestep input, or a dict mapping each
        timestep to its ensemble-average Ree² value for trajectory input.
    """

    if coord_cols is None:
        coord_cols = ["xu", "yu", "zu"]

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
    _require_columns(cols, [molecule_col, atom_id_col, *coord_cols], "Ree")
    mol_idx = cols.index(molecule_col)
    id_idx = cols.index(atom_id_col)
    x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]

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
    total_timesteps = len(unique_timesteps)
    ree_sq_by_timestep = {}

    # 3. Compute Ree^2 per frame using tqdm
    for i, ts in enumerate(
        tqdm(unique_timesteps, disable=total_timesteps <= 1, desc="Calculating Ree^2")
    ):
        # Isolate the frame
        ts_mask = timesteps == ts
        ts_mols = mols[ts_mask]
        ts_atom_ids = atom_ids[ts_mask]
        ts_coords = coords[ts_mask]

        # Keep only real chains (mol != 0, at least two atoms)
        chains = _chain_mask(ts_mols)
        if not chains.any():
            raise _no_chains_error(ts, has_timesteps, molecule_col)
        ts_mols = ts_mols[chains]
        ts_atom_ids = ts_atom_ids[chains]
        ts_coords = ts_coords[chains]

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
        sq_distances = np.sum((end_coords - start_coords) ** 2, axis=1)

        # Store the ensemble average for this timestep
        ree_sq_by_timestep[ts] = float(np.mean(sq_distances))
        if progress_callback is not None:
            progress_callback(i + 1, total_timesteps)

    # 4. Dynamic Return (keyed by timestep whenever one was present; see calculate_avg_rg_sq)
    if not has_timesteps:
        return list(ree_sq_by_timestep.values())[0]
    else:
        return ree_sq_by_timestep


def calculate_ree_vectors(
    df: pd.DataFrame | np.ndarray,
    columns: list[str] | None = None,
    coord_cols: list[str] | None = None,
    molecule_col: str = "mol",
    timestep_col: str = "timestep",
    atom_id_col: str = "id",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Compute the end-to-end vector for each molecule at each timestep.

    Only real chains are averaged: atoms with molecule ID 0 (no molecule) and molecules
    of a single atom are left out. A ValueError is raised if a frame has no chain left.

    Parameters
    ----------
    df : pd.DataFrame | np.ndarray
        Data for the calculation. Can be a pandas DataFrame or a numpy array.
        When a numpy array is passed, ``columns`` must also be provided.
    columns : list[str] | None, default=None
        Column header names required when ``df`` is a numpy array.
    coord_cols : list[str] | None, default=None
        Column headers for the coordinate columns. Defaults to
        ``['xu', 'yu', 'zu']`` when not provided.
    molecule_col : str, default='mol'
        Column header specifying molecule IDs.
    timestep_col : str, default='timestep'
        Column header specifying timestep values.
    atom_id_col : str, default='id'
        Column header specifying atom IDs. Used to identify the first and last
        atom in each chain (by minimum and maximum ID within a molecule).
    progress_callback : Callable[[int, int], None] | None, default=None
        If given, called as ``progress_callback(current, total)`` once per
        timestep processed, with ``current`` 1-based, so the final call is
        always ``(total, total)``. Default is None (no progress reporting).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[mol, dx, dy, dz]`` for single-frame input,
        or ``[timestep, mol, dx, dy, dz]`` for trajectory input. Each row
        contains the end-to-end vector for one molecule at one timestep.
    """

    if coord_cols is None:
        coord_cols = ["xu", "yu", "zu"]

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
    _require_columns(cols, [molecule_col, atom_id_col, *coord_cols], "Ree vector")
    mol_idx = cols.index(molecule_col)
    id_idx = cols.index(atom_id_col)
    x_idx, y_idx, z_idx = [cols.index(c) for c in coord_cols]

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
    total_timesteps = len(unique_timesteps)
    all_frames_data = []

    # 3. Compute Ree vectors per frame using tqdm
    for i, ts in enumerate(
        tqdm(unique_timesteps, disable=total_timesteps <= 1, desc="Calculating Ree vectors")
    ):
        # Isolate the frame
        ts_mask = timesteps == ts
        ts_mols = mols[ts_mask]
        ts_atom_ids = atom_ids[ts_mask]
        ts_coords = coords[ts_mask]

        # Keep only real chains (mol != 0, at least two atoms)
        chains = _chain_mask(ts_mols)
        if not chains.any():
            raise _no_chains_error(ts, has_timesteps, molecule_col)
        ts_mols = ts_mols[chains]
        ts_atom_ids = ts_atom_ids[chains]
        ts_coords = ts_coords[chains]

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
        if progress_callback is not None:
            progress_callback(i + 1, total_timesteps)

    # 4. Compile and Return Data
    # Vertically stack all the frame data blocks
    final_array = np.vstack(all_frames_data)

    # Convert to a DataFrame for clean downstream handling
    result_df = pd.DataFrame(final_array, columns=[timestep_col, molecule_col, "dx", "dy", "dz"])

    # Cast IDs back to integers (column_stack forces floats if coords are floats)
    result_df[timestep_col] = result_df[timestep_col].astype(int)
    result_df[molecule_col] = result_df[molecule_col].astype(int)

    if not has_timesteps:
        result_df = result_df.drop(columns=timestep_col)

    return result_df
