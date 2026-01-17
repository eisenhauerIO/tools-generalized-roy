"""
Minimal local polynomial kernel regression for MTE estimation.

Extracted from kernreg package (https://github.com/segsell/kernreg)
with Python 3.8+ compatibility fix (typing.TypedDict instead of mypy_extensions).

Based on Wand & Jones (1995) and their R package KernSmooth.
"""

import math
from typing import Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd


class Result(TypedDict):
    """Result dict for func locpoly."""

    gridpoints: np.ndarray
    curvest: np.ndarray
    bandwidth: float


def locpoly(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    derivative: int = 0,
    degree: Optional[int] = None,
    gridsize: Optional[int] = None,
    bandwidth: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    binned: bool = False,
    truncate: bool = True,
) -> Result:
    """Estimate regression function or derivatives using local polynomials.

    Args:
        x: Array of x data. Must be sorted.
        y: Array of y data. Same length as x. Must be presorted by x.
        derivative: Order of derivative to estimate (0 = function itself).
        degree: Degree of local polynomial. Default: derivative + 1.
        gridsize: Number of equally-spaced grid points.
        bandwidth: Kernel bandwidth smoothing parameter.
        a: Start point of grid.
        b: End point of grid.
        binned: If True, x and y are bin counts rather than raw data.
        truncate: If True, trim endpoints.

    Returns:
        Result dict with gridpoints, curvest (curve estimate), and bandwidth.
    """
    x, y, degree, gridsize, a, b = _process_inputs(x, y, derivative, degree, gridsize, a, b)

    if _is_sorted(x) is False:
        raise ValueError("Input arrays x and y must be sorted by x before estimation!")

    if degree not in [derivative + 1, derivative + 3, derivative + 5, derivative + 7]:
        raise ValueError("The degree of the polynomial must be equal to derivative v + 1, v + 3, v + 5, or v + 7.")

    if bandwidth is None:
        raise ValueError("Bandwidth must be provided.")

    binwidth = (b - a) / (gridsize - 1)

    if binned is False:
        xcounts, ycounts = _linear_binning(x, y, gridsize, a, binwidth, truncate)
    else:
        xcounts, ycounts = x, y

    weights = _get_kernelweights(bandwidth, binwidth)

    x_weighted, y_weighted = _combine_bincounts_kernelweights(
        xcounts, ycounts, weights, degree, gridsize, bandwidth, binwidth
    )

    curvest = _get_curve_estimator(x_weighted, y_weighted, degree, derivative, gridsize)

    gridpoints = np.linspace(a, b, gridsize)

    return Result(gridpoints=gridpoints, curvest=curvest, bandwidth=bandwidth)


def _get_curve_estimator(
    x_weighted: np.ndarray,
    y_weighted: np.ndarray,
    degree: int,
    derivative: int,
    gridsize: int,
) -> np.ndarray:
    """Solve locally weighted least-squares regression problem."""
    coly = degree + 1
    xmat = np.zeros((coly, coly))
    yvec = np.zeros(coly)
    curvest = np.zeros(gridsize)

    for g in range(gridsize):
        for row in range(0, coly):
            for column in range(0, coly):
                colindex = row + column
                xmat[row, column] = x_weighted[g, colindex]
                yvec[row] = y_weighted[g, row]

        beta = np.linalg.solve(xmat, yvec)
        curvest[g] = beta[derivative]

    curvest = math.gamma(derivative + 1) * curvest

    return curvest


def _process_inputs(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    derivative: int,
    degree: Optional[int],
    gridsize: Optional[int],
    a: Optional[float],
    b: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Process input arguments."""
    if isinstance(x, pd.Series):
        x, y = np.asarray(x), np.asarray(y)

    if degree is None:
        degree = derivative + 1

    if gridsize is None:
        gridsize = 401 if len(x) > 400 else len(x)

    if a is None:
        a = float(min(x))

    if b is None:
        b = float(max(x))

    return x, y, degree, gridsize, a, b


def _is_sorted(a: np.ndarray) -> bool:
    """Check if array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def _linear_binning(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int,
    a: float,
    binwidth: float,
    truncate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply linear binning to x and y."""
    N = len(x)

    xcounts = np.zeros(gridsize)
    ycounts = np.zeros(gridsize)
    xgrid = np.zeros(N)
    binweights = np.zeros(N)
    bincenters = [0] * N

    for i in range(N):
        xgrid[i] = ((x[i] - a) / binwidth) + 1
        bincenters[i] = int(xgrid[i])
        binweights[i] = xgrid[i] - bincenters[i]

    for point in range(gridsize):
        for index, value in enumerate(bincenters):
            if value == point:
                xcounts[point - 1] += 1 - binweights[index]
                xcounts[point] += binweights[index]

                ycounts[point - 1] += (1 - binweights[index]) * y[index]
                ycounts[point] += binweights[index] * y[index]

    if truncate is False:
        xcounts, ycounts = _include_weights_from_endpoints(xcounts, ycounts, y, xgrid, gridsize)

    return xcounts, ycounts


def _include_weights_from_endpoints(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    y: np.ndarray,
    xgrid: np.ndarray,
    gridsize: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attach weight from values outside the grid to corresponding endpoints."""
    for index, value in enumerate(xgrid):
        if value < 1:
            xcounts[0] += 1
            ycounts[0] += y[index]
        elif value >= gridsize:
            xcounts[gridsize - 1] += 1
            ycounts[gridsize - 1] += y[index]

    return xcounts, ycounts


def _get_kernelweights(bandwidth: float, delta: float) -> np.ndarray:
    """Compute approximated weights for the Gaussian kernel."""
    tau = 4
    L = math.floor(tau * bandwidth / delta)
    length = 2 * L + 1

    weights = np.zeros(length)
    mid = L + 1

    for j in range(L + 1):
        weights[mid - 1 + j] = math.exp(-((delta * j / bandwidth) ** 2) / 2)
        weights[mid - 1 - j] = weights[mid - 1 + j]

    return weights


def _combine_bincounts_kernelweights(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    weights: np.ndarray,
    degree: int,
    gridsize: int,
    bandwidth: float,
    binwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine bin counts and bin averages with kernel weights."""
    tau = 4
    L = math.floor(tau * bandwidth / binwidth)
    length = 2 * L + 1
    mid = L + 1

    colx = 2 * degree + 1
    coly = degree + 1

    x_weighted = np.zeros((gridsize, colx))
    y_weighted = np.zeros((gridsize, coly))

    for g in range(gridsize):
        if xcounts[g] != 0:
            for i in range(max(0, g - L - 1), min(gridsize, g + L)):
                if 0 <= i <= gridsize - 1 and 0 <= g - i + mid - 1 <= length - 1:
                    fac_ = 1.0

                    x_weighted[i, 0] += xcounts[g] * weights[g - i + mid - 1]
                    y_weighted[i, 0] += ycounts[g] * weights[g - i + mid - 1]

                    for j in range(1, colx):
                        fac_ *= binwidth * (g - i)

                        x_weighted[i, j] += xcounts[g] * weights[g - i + mid - 1] * fac_

                        if j < coly:
                            y_weighted[i, j] += ycounts[g] * weights[g - i + mid - 1] * fac_

    return x_weighted, y_weighted
