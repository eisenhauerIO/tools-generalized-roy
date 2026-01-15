"""
Custom exceptions for grmpy package.

Design Decision: Custom exceptions enable precise error handling and
provide context-rich messages that guide users to fix issues.
"""


class GrmpyError(Exception):
    """Base exception for all grmpy errors."""

    pass


class ConfigurationError(GrmpyError):
    """
    Raised when configuration is invalid or incomplete.

    Includes context about what is missing/invalid and available options.
    """

    pass


class DataValidationError(GrmpyError):
    """
    Raised when input data fails validation.

    Includes information about missing columns or invalid types.
    """

    pass


class EstimationError(GrmpyError):
    """
    Raised when estimation fails.

    Includes information about the failure mode (convergence, numerical issues).
    """

    pass


class SimulationError(GrmpyError):
    """
    Raised when simulation fails.

    Includes information about parameter issues or generation failures.
    """

    pass
