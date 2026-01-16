"""
Configuration management for grmpy package.

Design Decision: Single entry point for configuration processing ensures
consistent validation and default handling across all operations.

This module provides:
- Single process_config() function as public API
- Deep merge of user config with defaults
- Validation at load time with context-rich errors
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from grmpy.core.contracts import Config
from grmpy.core.exceptions import ConfigurationError


# -----------------------------------------------------------------------------
# Default Configuration
# Design Decision: FUNCTION + PARAMS pattern matches impact-engine for
# consistency across the organization's tooling.
# -----------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "ESTIMATION": {
        "FUNCTION": "parametric",
        "PARAMS": {
            "optimizer": "BFGS",
            "max_iterations": 10000,
            "tolerance": 1e-6,
            "gridsize": 500,
            "ps_range": [0.005, 0.995],
            # Minimum sample size for semiparametric estimation
            # Design Decision: 100 is conservative based on simulation studies
            # showing unreliable local polynomial estimates with fewer points
            "min_sample_size": 100,
        },
    },
    "SIMULATION": {
        "FUNCTION": "roy_model",
        "PARAMS": {
            "agents": 1000,
            "source": "sim",
            "output_file": "data.grmpy.pkl",
        },
    },
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def process_config(path: Union[str, Path]) -> Config:
    """
    Single public API for configuration processing.

    Design Decision: All configuration loading goes through this function
    to ensure consistent validation and default handling.

    Args:
        path: Path to user YAML configuration file

    Returns:
        Fully validated, merged Config object

    Raises:
        ConfigurationError: If file not found or validation fails
    """
    path = Path(path)

    # Validate file exists
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: '{path}'. "
            f"Please provide a valid .yml or .yaml file."
        )

    # Load user configuration
    user_config = _load_yaml(path)

    # Merge with defaults
    merged = _deep_merge(DEFAULTS.copy(), user_config)

    # Convert to typed Config object (validates during construction)
    config = Config.from_dict(merged)

    # Run additional validation
    _validate_config(config)

    return config


def load_raw_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration as raw dictionary without processing.

    Useful for accessing raw values before type conversion.

    Args:
        path: Path to YAML configuration file

    Returns:
        Raw configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: '{path}'")
    return _deep_merge(DEFAULTS.copy(), _load_yaml(path))


# -----------------------------------------------------------------------------
# Internal Functions
# -----------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML file and return dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        ConfigurationError: If YAML parsing fails
    """
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {path}\n"
            f"Error: {e}"
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dict into base dict.

    Design Decision: Deep merge preserves nested structures while
    allowing selective overrides at any level.

    Args:
        base: Base dictionary with defaults
        override: Override dictionary from user

    Returns:
        Merged dictionary with override values taking precedence
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _validate_config(config: Config) -> None:
    """
    Perform additional validation on configuration.

    Args:
        config: Config object to validate

    Raises:
        ConfigurationError: If validation fails
    """
    if config.estimation is not None:
        config.estimation.validate()

        # Check data file exists for estimation
        if config.estimation.file:
            data_path = Path(config.estimation.file)
            if not data_path.exists():
                raise ConfigurationError(
                    f"Estimation data file not found: '{data_path}'. "
                    f"Please provide a valid data file path."
                )

    if config.simulation is not None:
        config.simulation.validate()
