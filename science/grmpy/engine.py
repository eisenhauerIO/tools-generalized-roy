"""
Main engine for grmpy package.

Public API:
- fit(): Estimate MTE from data
- simulate(): Generate synthetic Roy model data
- plot_mte(): Visualize marginal treatment effects
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from grmpy.core.contracts import EstimationResult
from grmpy.config import process_config


def fit(config_path: Union[str, Path]) -> EstimationResult:
    """
    Estimate Marginal Treatment Effect from data.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        EstimationResult with MTE and coefficients.

    Example:
        >>> result = grmpy.fit("analysis.grmpy.yml")
        >>> print(result.mte)
    """
    from grmpy.estimators import fit as run_estimation

    config = process_config(str(config_path))
    return run_estimation(config)


def simulate(config_path: Union[str, Path]) -> pd.DataFrame:
    """
    Generate synthetic data according to the generalized Roy model.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        DataFrame with simulated data (Y, Y_1, Y_0, D, U_1, U_0, V).

    Example:
        >>> data = grmpy.simulate("simulation.grmpy.yml")
        >>> print(data.head())
    """
    from grmpy.simulators import simulate as run_simulation

    config = process_config(str(config_path))
    return run_simulation(config)


def plot_mte(
    result: EstimationResult,
    config_path: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    show_confidence: bool = True,
) -> None:
    """
    Plot Marginal Treatment Effect curve.

    Args:
        result: EstimationResult from fit()
        config_path: Optional path to config for plot settings
        output_file: Optional path to save figure
        show_confidence: Whether to show confidence bands

    Example:
        >>> result = grmpy.fit("analysis.grmpy.yml")
        >>> grmpy.plot_mte(result, output_file="mte_plot.png")
    """
    from grmpy.visualization.mte_plot import plot_mte_curve

    plot_mte_curve(
        result=result,
        config_path=config_path,
        output_file=output_file,
        show_confidence=show_confidence,
    )
