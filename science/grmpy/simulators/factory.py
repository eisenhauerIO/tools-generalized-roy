"""
Factory and registry for simulator creation.

Design Decision: Registry pattern enables extension without modifying
core code, consistent with estimator factory design.
"""

from typing import Dict, Type

from grmpy.simulators.base import Simulator


# Registry mapping simulator names to implementations
SIMULATOR_REGISTRY: Dict[str, Type[Simulator]] = {}


def register_simulator(name: str, simulator_class: Type[Simulator]) -> None:
    """
    Register a custom simulator at runtime.

    Args:
        name: Identifier for the simulator
        simulator_class: Class implementing Simulator interface

    Raises:
        TypeError: If class doesn't implement Simulator interface
    """
    if not issubclass(simulator_class, Simulator):
        raise TypeError(
            f"Simulator class must inherit from Simulator base class. "
            f"Got: {simulator_class} with bases {simulator_class.__bases__}"
        )
    SIMULATOR_REGISTRY[name] = simulator_class


def get_simulator(name: str) -> Simulator:
    """
    Factory function to instantiate simulator by name.

    Args:
        name: Name of simulator

    Returns:
        Instantiated simulator object

    Raises:
        ValueError: If simulator not found, lists available options
    """
    if name not in SIMULATOR_REGISTRY:
        available = list(SIMULATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown simulator: '{name}'. "
            f"Available simulators: {available}"
        )
    return SIMULATOR_REGISTRY[name]()


def create_simulator_manager(config_path: str) -> "SimulatorManager":
    """
    High-level factory creating configured manager.

    Args:
        config_path: Path to YAML configuration

    Returns:
        Fully configured SimulatorManager ready for simulation
    """
    from grmpy.config import process_config
    from grmpy.simulators.manager import SimulatorManager

    config = process_config(config_path)

    if config.simulation is None:
        from grmpy.core.exceptions import ConfigurationError
        raise ConfigurationError(
            "No simulation configuration found in config file. "
            "Please add a SIMULATION section."
        )

    # Default to roy_model simulator
    simulator = get_simulator("roy_model")
    return SimulatorManager(simulator, config)
