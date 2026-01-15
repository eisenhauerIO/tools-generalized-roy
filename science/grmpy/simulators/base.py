"""
Abstract base interface for all simulation methods.

Design Decision: Follows the same pattern as estimators for consistency.
Separates configuration from simulation logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from grmpy.core.contracts import SimulationConfig, SimulationResult


class Simulator(ABC):
    """
    Abstract interface for all simulation methods.

    Lifecycle:
    1. connect() - Initialize with configuration
    2. validate_connection() - Verify setup is complete
    3. simulate() - Execute simulation
    """

    @abstractmethod
    def connect(self, config: SimulationConfig) -> None:
        """
        Initialize simulator with configuration.

        Args:
            config: Simulation configuration object
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Verify simulator is properly configured.

        Returns:
            True if configuration is valid and complete
        """
        pass

    @abstractmethod
    def simulate(self, **kwargs) -> SimulationResult:
        """
        Execute simulation and return results.

        Args:
            **kwargs: Additional simulator-specific options

        Returns:
            SimulationResult containing generated data and parameters
        """
        pass
