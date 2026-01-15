"""
Generalized Roy Model simulator adapter.

Design Decision: This adapter wraps the existing simulate() logic while
conforming to the Simulator interface for consistent orchestration. This
enables using the validated simulation algorithms while providing a modern,
extensible interface.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from grmpy.core.contracts import SimulationConfig, SimulationResult
from grmpy.core.exceptions import SimulationError
from grmpy.simulators.base import Simulator


class RoyModelSimulator(Simulator):
    """
    Generalized Roy Model data simulator.

    Generates synthetic data according to the generalized Roy model
    with configurable parameters for outcomes and selection.

    Design Decision: Wraps legacy simulation functions to leverage
    tested code while providing modern interface for extension.
    """

    def __init__(self):
        self.config: Optional[SimulationConfig] = None

    def connect(self, config: SimulationConfig) -> None:
        """
        Initialize simulator with configuration.

        Args:
            config: SimulationConfig containing number of agents, seed,
                coefficients, and covariance matrix parameters.
        """
        self.config = config

    def validate_connection(self) -> bool:
        """
        Verify simulator is properly configured.

        Checks that config is set and agents count is positive.

        Returns:
            True if config is valid, False otherwise.
        """
        if self.config is None:
            return False
        if self.config.agents <= 0:
            return False
        return True

    def simulate(self, **kwargs) -> SimulationResult:
        """
        Execute simulation and return results.

        Generates synthetic Roy model data by:
        1. Setting random seed for reproducibility
        2. Simulating unobservables from multivariate normal
        3. Simulating covariates
        4. Computing outcomes and treatment decisions

        Design Decision: Wraps legacy simulate_* functions rather than
        reimplementing to avoid introducing bugs in statistical code.

        Args:
            **kwargs: Additional arguments (for interface compatibility).

        Returns:
            SimulationResult containing:
            - data: DataFrame with Y, Y_1, Y_0, D, and unobservables
            - parameters: Dict of simulation parameters used
            - metadata: Source information

        Raises:
            SimulationError: If not connected or legacy module import fails.
        """
        if self.config is None:
            raise SimulationError("Simulator not connected")

        # Set seed if provided for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Import and use legacy simulation
        # Design Decision: Keep core simulation in legacy module during
        # migration to avoid introducing bugs in validated statistical code
        try:
            from grmpy.simulate.simulate_auxiliary import (
                simulate_covariates,
                simulate_outcomes,
                simulate_unobservables,
                write_output,
            )
        except ImportError as e:
            raise SimulationError(
                f"Failed to import legacy simulation module: {e}"
            )

        # Build legacy dict format
        legacy_dict = self._build_legacy_dict()

        # Execute simulation steps
        U = simulate_unobservables(legacy_dict)
        X = simulate_covariates(legacy_dict)
        df = simulate_outcomes(legacy_dict, X, U)

        # Optionally write output
        if self.config.output_file:
            df = write_output(legacy_dict, df)

        # Build parameters dict for result
        parameters = {
            "agents": self.config.agents,
            "seed": self.config.seed,
            "coefficients_treated": self.config.coefficients_treated,
            "coefficients_untreated": self.config.coefficients_untreated,
            "coefficients_choice": self.config.coefficients_choice,
            "covariance": self.config.covariance,
        }

        return SimulationResult(
            data=df,
            parameters=parameters,
            metadata={"source": "roy_model"},
        )

    def _build_legacy_dict(self) -> Dict[str, Any]:
        """
        Build legacy dictionary format from new config.

        Design Decision: This bridge function enables gradual migration
        from the old dict-based configuration to typed dataclasses without
        rewriting the core simulation algorithms.

        Returns:
            Dictionary in legacy format expected by simulate_* functions.

        Raises:
            SimulationError: If simulator not connected.
        """
        if self.config is None:
            raise SimulationError("Simulator not connected")

        # Determine number of covariates from coefficients
        n_treated = len(self.config.coefficients_treated)
        n_untreated = len(self.config.coefficients_untreated)
        n_choice = len(self.config.coefficients_choice)

        legacy_dict = {
            "SIMULATION": {
                "agents": self.config.agents,
                "seed": self.config.seed,
                "source": self.config.source,
                "output_file": self.config.output_file,
            },
            "TREATED": {
                "coeff": self.config.coefficients_treated,
                "order": [f"X{i}" for i in range(n_treated)],
            },
            "UNTREATED": {
                "coeff": self.config.coefficients_untreated,
                "order": [f"X{i}" for i in range(n_untreated)],
            },
            "CHOICE": {
                "coeff": self.config.coefficients_choice,
                "order": [f"Z{i}" for i in range(n_choice)],
            },
            "DIST": {
                "coeff": self.config.covariance,
            },
            "DETERMINISTIC": False,
            "AUX": {},
        }

        return legacy_dict
