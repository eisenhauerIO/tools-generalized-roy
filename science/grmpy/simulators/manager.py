"""
Simulator manager for orchestrating simulation workflow.

Design Decision: Separates workflow coordination from simulation
logic, consistent with estimator manager design.
"""

from typing import Optional

import pandas as pd

from grmpy.core.contracts import Config, SimulationResult
from grmpy.core.exceptions import SimulationError
from grmpy.simulators.base import Simulator


class SimulatorManager:
    """
    Orchestrates simulation workflow.

    Responsibilities:
    - Initialize and validate simulator connection
    - Coordinate simulation execution
    - Format and return results
    """

    def __init__(self, simulator: Simulator, config: Config):
        """
        Initialize manager with simulator and configuration.

        Args:
            simulator: Simulator instance to use
            config: Configuration object with simulation settings
        """
        self.simulator = simulator
        self.config = config
        self._is_connected = False

    def connect(self) -> "SimulatorManager":
        """
        Initialize simulator connection.

        Returns:
            Self for method chaining

        Raises:
            SimulationError: If connection validation fails
        """
        if self.config.simulation is None:
            raise SimulationError("No simulation configuration provided")

        self.simulator.connect(self.config.simulation)

        if not self.simulator.validate_connection():
            raise SimulationError(
                "Simulator failed connection validation. "
                "Check configuration settings."
            )

        self._is_connected = True
        return self

    def simulate(self) -> pd.DataFrame:
        """
        Execute simulation and return generated data.

        Returns:
            DataFrame with simulated data

        Raises:
            SimulationError: If not connected or simulation fails
        """
        if not self._is_connected:
            raise SimulationError(
                "Must call connect() before simulate(). "
                "Use: manager.connect().simulate()"
            )

        try:
            result = self.simulator.simulate()
            return result.data
        except Exception as e:
            raise SimulationError(f"Simulation failed: {e}") from e

    @property
    def is_connected(self) -> bool:
        """Check if manager is connected and ready."""
        return self._is_connected
