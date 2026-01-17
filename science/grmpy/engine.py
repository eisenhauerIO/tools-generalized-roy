"""
Main engine for grmpy package.

Public API:
- estimate(): Estimate MTE from data
- simulate(): Generate synthetic Roy model data
- plot(): Visualize marginal treatment effects
"""

from typing import Optional

import pandas as pd

from grmpy.core.contracts import Config, EstimationResult


def estimate(config: Config, data: pd.DataFrame) -> EstimationResult:
    """
    Estimate Marginal Treatment Effect from data.

    Args:
        config: Configuration object with estimation settings.
        data: DataFrame with outcome, treatment, and covariates.

    Returns:
        EstimationResult with MTE and coefficients.

    Example:
        >>> config = grmpy.process_config("analysis.yml")
        >>> result = grmpy.estimate(config, data)
        >>> print(result.mte)
    """
    from grmpy.estimators import estimate as run_estimation

    return run_estimation(config, data)


def simulate(config: Config) -> pd.DataFrame:
    """
    Generate synthetic data according to the generalized Roy model.

    Args:
        config: Configuration object with simulation settings.

    Returns:
        DataFrame with simulated data (Y, Y_1, Y_0, D, U_1, U_0, V).

    Example:
        >>> config = grmpy.process_config("simulation.yml")
        >>> data = grmpy.simulate(config)
        >>> print(data.head())
    """
    from grmpy.simulators import simulate as run_simulation

    return run_simulation(config)


def plot(
    result: EstimationResult,
    output_file: Optional[str] = None,
    show_confidence: bool = True,
) -> None:
    """
    Plot Marginal Treatment Effect curve.

    Args:
        result: EstimationResult from estimate()
        output_file: Optional path to save figure
        show_confidence: Whether to show confidence bands

    Example:
        >>> result = grmpy.estimate(config, data)
        >>> grmpy.plot(result, output_file="mte.png")
    """
    from grmpy.visualization.mte_plot import plot_mte_curve

    plot_mte_curve(
        result=result,
        output_file=output_file,
        show_confidence=show_confidence,
    )
