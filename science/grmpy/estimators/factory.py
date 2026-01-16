"""
Factory and registry for estimator creation.

Design Decision: Registry pattern enables extension without modifying
core code. Users can add custom estimators at runtime.

This module provides:
- ESTIMATOR_REGISTRY: Mapping of method names to classes
- register_estimator(): Runtime registration of custom estimators
- get_estimator(): Factory function for instantiation
- create_estimator_manager(): High-level factory with configuration
"""

from typing import Dict, Type

from grmpy.estimators.base import Estimator


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

# Registry mapping method names to implementations
# Populated by adapter imports at module load time
ESTIMATOR_REGISTRY: Dict[str, Type[Estimator]] = {}


# -----------------------------------------------------------------------------
# Registration Functions
# -----------------------------------------------------------------------------


def register_estimator(name: str, estimator_class: Type[Estimator]) -> None:
    """
    Register a custom estimator at runtime.

    Design Decision: Enables extension without modifying core code.
    Users can add custom estimators for specialized use cases.

    Args:
        name: Identifier for the estimator (e.g., "parametric")
        estimator_class: Class implementing Estimator interface

    Raises:
        TypeError: If class doesn't implement Estimator interface

    Example:
        >>> class MyEstimator(Estimator):
        ...     # implementation
        >>> register_estimator("my_method", MyEstimator)
    """
    if not issubclass(estimator_class, Estimator):
        raise TypeError(
            f"Estimator class must inherit from Estimator base class. "
            f"Got: {estimator_class} with bases {estimator_class.__bases__}"
        )
    ESTIMATOR_REGISTRY[name] = estimator_class


def unregister_estimator(name: str) -> None:
    """
    Remove an estimator from the registry.

    Args:
        name: Identifier of estimator to remove

    Raises:
        ValueError: If estimator not found in registry
    """
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Estimator '{name}' not found in registry")
    del ESTIMATOR_REGISTRY[name]


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def get_estimator(method: str) -> Estimator:
    """
    Factory function to instantiate estimator by method name.

    Args:
        method: Name of estimation method

    Returns:
        Instantiated estimator object

    Raises:
        ValueError: If method not found, lists available options
    """
    if method not in ESTIMATOR_REGISTRY:
        available = list(ESTIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown estimation method: '{method}'. "
            f"Available methods: {available}"
        )
    return ESTIMATOR_REGISTRY[method]()


def list_estimators() -> Dict[str, str]:
    """
    List all registered estimators with their docstrings.

    Returns:
        Dictionary mapping method names to descriptions
    """
    return {
        name: (cls.__doc__ or "No description").split("\n")[0].strip()
        for name, cls in ESTIMATOR_REGISTRY.items()
    }


def create_estimator_manager(config_path: str) -> "EstimatorManager":
    """
    High-level factory creating configured manager.

    Args:
        config_path: Path to YAML configuration

    Returns:
        Fully configured EstimatorManager ready for fitting
    """
    from grmpy.config import process_config
    from grmpy.estimators.manager import EstimatorManager

    config = process_config(config_path)

    if config.estimation is None:
        from grmpy.core.exceptions import ConfigurationError
        raise ConfigurationError(
            "No estimation configuration found in config file. "
            "Please add an ESTIMATION section."
        )

    estimator = get_estimator(config.estimation.function)
    return EstimatorManager(estimator, config)


# -----------------------------------------------------------------------------
# Auto-register built-in estimators
# -----------------------------------------------------------------------------

def _register_builtin_estimators() -> None:
    """Register built-in estimators on module import."""
    # Import here to avoid circular imports
    from grmpy.estimators.adapter_parametric import ParametricEstimator
    from grmpy.estimators.adapter_semiparametric import SemiparametricEstimator

    register_estimator("parametric", ParametricEstimator)
    register_estimator("semiparametric", SemiparametricEstimator)


# Register on import - deferred to avoid import errors during development
# _register_builtin_estimators()
