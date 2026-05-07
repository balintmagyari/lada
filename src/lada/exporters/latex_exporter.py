from typing import Literal

import numpy as np
import pandas as pd


def write_pgfplots_table(
    data: pd.DataFrame | np.ndarray | dict,
    filepath: str,
    columns: list[str] | None = None,
    delimiter: Literal[" ", "\t", ","] = " ",
    fmt: str = "%.6g",
    comment: str | None = None,
) -> None:
    """Write data to a delimited file for use with pgfplots \\addplot table in LaTeX.

    The output file can be read directly by pgfplots using:

        \\addplot table[x=col1, y=col2, col sep=space]{filename.dat};

    Change ``col sep`` to ``tab`` or ``comma`` if you use those delimiters.

    Parameters
    ----------
    data : pd.DataFrame | np.ndarray | dict
        Data to export. Accepted forms:
        - ``pd.DataFrame``: column names are used as the header automatically.
        - ``dict``: keys become the header; values must be equal-length 1-D arrays.
        - ``np.ndarray``: 1-D arrays are treated as a single column; 2-D arrays
          are written as-is. Supply ``columns`` to include a header row.
    filepath : str
        Destination file path (e.g. ``'results/rg_data.dat'``).
    columns : list[str] | None, optional
        Column names written as the header row.  Required when ``data`` is a
        NumPy array and you want named columns.  When ``data`` is a DataFrame
        or dict this argument overrides the automatic names.
    delimiter : {' ', '\\t', ','}, default ' '
        Field separator written between values. Space produces ``.dat`` files
        readable by pgfplots with ``col sep=space`` (the default pgfplots
        setting). Use ``','`` for CSV or ``'\\t'`` for TSV.
    fmt : str, default '%.6g'
        Printf-style format string applied to every numeric value.  ``'%.6g'``
        gives up to 6 significant figures and suppresses trailing zeros.
        Use ``'%.10e'`` for full scientific notation, ``'%.4f'`` for fixed
        decimal places, etc.
    comment : str | None, optional
        Optional comment text prepended to the file.  Each line is prefixed
        with ``%`` (the LaTeX/pgfplots comment character), so pgfplots ignores
        it automatically.  Useful for recording simulation metadata alongside
        the data.

    Raises
    ------
    TypeError
        If ``data`` is not a DataFrame, ndarray, or dict.
    ValueError
        If ``columns`` length does not match the number of data columns, or if
        dict values have unequal lengths.

    Examples
    --------
    Export a DataFrame directly:

    >>> write_pgfplots_table(df, 'rg.dat', comment='Rg^2 vs time, T=1.0')

    Export named NumPy columns:

    >>> write_pgfplots_table(
    ...     np.column_stack([time, rg_sq]),
    ...     'rg.dat',
    ...     columns=['t', 'Rg2'],
    ... )

    Export a dict of arrays:

    >>> write_pgfplots_table(
    ...     {'lag_time': t, 'G_GK': g_gk, 'G_FSR': g_fsr},
    ...     'stress.dat',
    ...     delimiter=',',
    ... )
    """
    # --- Normalise input to (header, 2-D array) ---
    if isinstance(data, pd.DataFrame):
        header = list(data.columns) if columns is None else columns
        arr = data.to_numpy()

    elif isinstance(data, dict):
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All arrays in the dict must have the same length; "
                f"got lengths {dict(zip(data.keys(), lengths, strict=False))}."
            )
        header = list(data.keys()) if columns is None else columns
        arr = np.column_stack(list(data.values()))

    elif isinstance(data, np.ndarray):
        arr = data if data.ndim == 2 else data.reshape(-1, 1)
        header = columns  # may be None — written without header in that case

    else:
        raise TypeError(
            f"'data' must be a pd.DataFrame, np.ndarray, or dict; got {type(data).__name__}."
        )

    # --- Validate column count ---
    if header is not None and len(header) != arr.shape[1]:
        raise ValueError(
            f"'columns' has {len(header)} entries but data has {arr.shape[1]} column(s)."
        )

    # --- Write file ---
    with open(filepath, "w") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"% {line}\n")

        if header is not None:
            f.write(delimiter.join(header) + "\n")

        for row in arr:
            f.write(delimiter.join(fmt % v for v in row) + "\n")
