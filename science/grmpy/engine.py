"""
Main engine orchestrator for grmpy package.

Design Decision: This module provides the primary public API functions
that coordinate the complete estimation and simulation workflows.

Public API:
- fit(): Main entry point for estimation
- simulate(): Main entry point for simulation
- plot_mte(): Visualize marginal treatment effects
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from grmpy.core.contracts import EstimationResult
from grmpy.config import process_config


def fit(config_path: Union[str, Path]) -> EstimationResult:
    """
    Main entry point for estimation.

    Orchestrates the complete estimation workflow:
    1. Load and validate configuration
    2. Create appropriate estimator
    3. Load and validate data
    4. Execute estimation
    5. Return standardized results

    Args:
        config_path: Path to YAML configuration file

    Returns:
        EstimationResult with MTE and related quantities

    Raises:
        ConfigurationError: If configuration is invalid
        DataValidationError: If data fails validation
        EstimationError: If estimation fails

    Example:
        >>> result = grmpy.fit("analysis.grmpy.yml")
        >>> print(result.mte)
    """
    from grmpy.estimators.factory import create_estimator_manager

    # Create configured manager
    manager = create_estimator_manager(str(config_path))
    manager.connect()

    # Execute estimation
    result = manager.fit()

    return result


def simulate(config_path: Union[str, Path]) -> pd.DataFrame:
    """
    Main entry point for simulation.

    Generates synthetic data according to the generalized Roy model
    with parameters specified in the configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        DataFrame with simulated data including:
        - Y: Observed outcome
        - Y_1, Y_0: Potential outcomes
        - D: Treatment indicator
        - U_1, U_0, V: Unobservables

    Raises:
        ConfigurationError: If configuration is invalid
        SimulationError: If simulation fails

    Example:
        >>> data = grmpy.simulate("simulation.grmpy.yml")
        >>> print(data.head())
    """
    from grmpy.simulators.factory import create_simulator_manager

    # Create configured manager
    manager = create_simulator_manager(str(config_path))
    manager.connect()

    # Execute simulation
    return manager.simulate()


def plot_mte(
    result: EstimationResult,
    config_path: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    show_confidence: bool = True,
) -> None:
    """
    Plot Marginal Treatment Effect curve.

    Visualizes the MTE across the distribution of unobserved resistance
    with optional confidence intervals.

    Args:
        result: EstimationResult from fit()
        config_path: Optional path to config for additional plot settings
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
