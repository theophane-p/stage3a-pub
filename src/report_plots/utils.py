from typing import Optional
import numpy as np


def format_float_tex(
        value: float, 
        precision: Optional[int] = 2
) -> str:
    r"""Format a float to a string in LaTeX format."""
    if np.isnan(value):
        return "NaN"
    elif np.isinf(value):
        return r"$\infty$"
    elif value == 0:
        return "0"
    else:
        e = np.floor(np.log10(np.abs(value)))
        m = value / 10**e
        return f"${m:.{precision}f} \\times 10^{{{int(e)}}}$"


def format_array1d_tex(
        array: np.ndarray,
        precision: Optional[int] = 2
) -> str:
    r"""Format a 1D array to a string in LaTeX format."""
    return " & ".join([format_float_tex(value) for value in array])