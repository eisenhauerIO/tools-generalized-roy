"""
Core module containing contracts, exceptions, and shared utilities.
"""

from grmpy.core.contracts import (
    Config,
    DataSchema,
    EstimationConfig,
    EstimationDataSchema,
    EstimationResult,
    SimulationConfig,
    SimulationDataSchema,
    SimulationResult,
)
from grmpy.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    EstimationError,
    GrmpyError,
    SimulationError,
)

__all__ = [
    # Contracts
    "Config",
    "DataSchema",
    "EstimationConfig",
    "EstimationDataSchema",
    "EstimationResult",
    "SimulationConfig",
    "SimulationDataSchema",
    "SimulationResult",
    # Exceptions
    "GrmpyError",
    "ConfigurationError",
    "DataValidationError",
    "EstimationError",
    "SimulationError",
]
