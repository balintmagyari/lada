import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def calculate_segment_acf_from_trajectory(
    trajectory_file: str, segment_pairs: np.ndarray, time_per_frame: float
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
    segment_pairs : numpy.ndarray
        A 2D array of integers of shape (n_chains, 2) containing the 0-indexed
        indices of the head and tail beads for every polymer chain segment that
        the user want to perform autocorrelation for.
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
    coords = data["coords"]  # Shape: (n_frames, n_atoms, 3)

    # Extract the head and tail indices
    heads = segment_pairs[:, 0]
    tails = segment_pairs[:, 1]

    # Calculate the end-to-end vector for ALL chains at ALL frames
    R = coords[:, tails, :] - coords[:, heads, :]

    n_frames, n_chains, _ = R.shape
    acf = np.zeros(n_frames)

    # Slide the time window (lag) across the trajectory
    for lag in tqdm(range(n_frames), "Calculating autocorrelation"):
        R_start = R[: n_frames - lag]
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


def calculate_rouse_mode_acf_from_trajectory(
    trajectory_file: str, chain_indices: np.ndarray, p: int, time_per_frame: float
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
    coords = data["coords"]  # Shape: (n_frames, n_atoms, 3)

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
            f"must be an integer strictly between 0 and {N - 1}."
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
        X_start = X_p[: n_frames - lag]
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


def calculate_isf_from_trajectory(
    trajectory_file: str, time_per_frame: float, q_magnitude: float, n_vectors: int = 50
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
    coords = data["coords"]  # Shape: (n_frames, n_atoms, 3)

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
            rho_start = rho[: n_frames - lag]
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
# Deprecated aliases (pre-1.2.0 names) — removed in lada 2.0.0
# ---------------------------------------------------------------------------


def calculate_segment_acf(
    trajectory_file: str, segment_pairs: np.ndarray, time_per_frame: float
) -> NDArray[np.float64]:
    """Calculate the segment end-to-end ACF from a compressed NumPy trajectory.

    .. deprecated:: 1.2.0
       `calculate_segment_acf` will be removed in lada 2.0.0.
       Please use `calculate_segment_acf_from_trajectory` instead.
    """
    warnings.warn(
        "Call to deprecated function 'calculate_segment_acf'. "
        "This function will be removed in version 2.0.0. "
        "Use 'calculate_segment_acf_from_trajectory' instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return calculate_segment_acf_from_trajectory(trajectory_file, segment_pairs, time_per_frame)


def calculate_rouse_mode_acf(
    trajectory_file: str, chain_indices: np.ndarray, p: int, time_per_frame: float
) -> NDArray[np.float64]:
    """Calculate the Rouse mode ACF from a compressed NumPy trajectory.

    .. deprecated:: 1.2.0
       `calculate_rouse_mode_acf` will be removed in lada 2.0.0.
       Please use `calculate_rouse_mode_acf_from_trajectory` instead.
    """
    warnings.warn(
        "Call to deprecated function 'calculate_rouse_mode_acf'. "
        "This function will be removed in version 2.0.0. "
        "Use 'calculate_rouse_mode_acf_from_trajectory' instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return calculate_rouse_mode_acf_from_trajectory(
        trajectory_file, chain_indices, p, time_per_frame
    )


def calculate_isf(
    trajectory_file: str, time_per_frame: float, q_magnitude: float, n_vectors: int = 50
) -> NDArray[np.float64]:
    """Calculate the coherent intermediate scattering function from a compressed NumPy trajectory.

    .. deprecated:: 1.2.0
       `calculate_isf` will be removed in lada 2.0.0.
       Please use `calculate_isf_from_trajectory` instead.
    """
    warnings.warn(
        "Call to deprecated function 'calculate_isf'. "
        "This function will be removed in version 2.0.0. "
        "Use 'calculate_isf_from_trajectory' instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return calculate_isf_from_trajectory(trajectory_file, time_per_frame, q_magnitude, n_vectors)
