"""
Generalized Roy Model simulator.

Provides a straightforward simulate() function for generating synthetic
data according to the generalized Roy model specification.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from grmpy.core.contracts import Config, SimulationConfig
from grmpy.core.exceptions import GrmpyError


def simulate(config: Config) -> pd.DataFrame:
    """
    Generate synthetic data according to the generalized Roy model.

    Executes simulation by:
    1. Validating configuration
    2. Setting random seed for reproducibility
    3. Simulating unobservables from multivariate normal
    4. Simulating covariates
    5. Computing outcomes and treatment decisions

    Args:
        config: Configuration object containing simulation parameters.

    Returns:
        DataFrame with simulated data including:
        - Y: Observed outcome
        - Y_1, Y_0: Potential outcomes
        - D: Treatment indicator
        - U_1, U_0, V: Unobservables

    Raises:
        ConfigurationError: If simulation configuration is missing.
        SimulationError: If legacy module import fails or simulation errors.
    """
    if config.simulation is None:
        raise GrmpyError(
            "No simulation configuration found in config file. "
            "Please add a SIMULATION section."
        )

    sim_config = config.simulation

    # Validate basic requirements
    if sim_config.agents <= 0:
        raise GrmpyError("Number of agents must be positive")

    # Set seed if provided for reproducibility
    if sim_config.seed is not None:
        np.random.seed(sim_config.seed)

    # Import legacy simulation functions
    try:
        from grmpy.simulate.simulate_auxiliary import (
            simulate_covariates,
            simulate_outcomes,
            simulate_unobservables,
            write_output,
        )
    except ImportError as e:
        raise GrmpyError(f"Failed to import legacy simulation module: {e}")

    # Build legacy dict format for compatibility
    legacy_dict = _build_legacy_dict(sim_config)

    # Execute simulation steps
    U = simulate_unobservables(legacy_dict)
    X = simulate_covariates(legacy_dict)
    df = simulate_outcomes(legacy_dict, X, U)

    # Optionally write output
    if sim_config.output_file:
        df = write_output(legacy_dict, df)

    return df


def _build_legacy_dict(sim_config: SimulationConfig) -> Dict[str, Any]:
    """
    Build legacy dictionary format from SimulationConfig.

    This bridge function enables using the validated legacy simulation
    algorithms with the new typed configuration system.

    Args:
        sim_config: SimulationConfig with simulation parameters.

    Returns:
        Dictionary in legacy format expected by simulate_* functions.
    """
    # Determine number of covariates from coefficients
    n_treated = len(sim_config.coefficients_treated)
    n_untreated = len(sim_config.coefficients_untreated)
    n_choice = len(sim_config.coefficients_choice)

    legacy_dict = {
        "SIMULATION": {
            "agents": sim_config.agents,
            "seed": sim_config.seed,
            "source": sim_config.source,
            "output_file": sim_config.output_file,
        },
        "TREATED": {
            "coeff": sim_config.coefficients_treated,
            "order": [f"X{i}" for i in range(n_treated)],
        },
        "UNTREATED": {
            "coeff": sim_config.coefficients_untreated,
            "order": [f"X{i}" for i in range(n_untreated)],
        },
        "CHOICE": {
            "coeff": sim_config.coefficients_choice,
            "order": [f"Z{i}" for i in range(n_choice)],
        },
        "DIST": {
            "coeff": sim_config.covariance,
        },
        "DETERMINISTIC": False,
        "AUX": {},
    }

    return legacy_dict
