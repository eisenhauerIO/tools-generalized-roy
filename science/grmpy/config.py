"""
Configuration management for grmpy package.

Design Decision: Single entry point for configuration processing ensures
consistent validation and default handling across all operations.

This module provides:
- Single process_config() function as public API
- Deep merge of user config with defaults from YAML
- Validation at load time with context-rich errors

Design Decision: Defaults stored in config_defaults.yaml per coding standards,
enabling easy schema extraction and user-friendly reference.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from grmpy.core.contracts import Config
from grmpy.core.exceptions import GrmpyError

# -----------------------------------------------------------------------------
# Default Configuration
# Design Decision: Defaults loaded from YAML file for consistency with
# coding standards and to enable schema extraction.
# -----------------------------------------------------------------------------

_DEFAULTS_FILE = Path(__file__).parent / "config_defaults.yaml"


def _load_defaults() -> Dict[str, Any]:
    """Load default configuration from YAML file."""
    if not _DEFAULTS_FILE.exists():
        raise GrmpyError(f"Default configuration file not found: {_DEFAULTS_FILE}")
    with open(_DEFAULTS_FILE, "r") as f:
        return yaml.safe_load(f) or {}


# Load defaults at module import time
DEFAULTS: Dict[str, Any] = _load_defaults()


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
        raise GrmpyError(f"Configuration file not found: '{path}'. Please provide a valid .yml or .yaml file.")

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
        raise GrmpyError(f"Configuration file not found: '{path}'")
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
        raise GrmpyError(f"Failed to parse YAML configuration: {path}\nError: {e}")


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
                raise GrmpyError(
                    f"Estimation data file not found: '{data_path}'. Please provide a valid data file path."
                )

    if config.simulation is not None:
        config.simulation.validate()
