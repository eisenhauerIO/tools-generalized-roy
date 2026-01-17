"""
Pytest configuration and fixtures for grmpy tests.

Design Decision: Centralized fixtures ensure consistent test setup
and enable isolated, reproducible test execution.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Markers
# -----------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line(
        "markers", "regression: marks regression tests against vault"
    )


# -----------------------------------------------------------------------------
# Fixtures: Environment
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_resources_dir() -> Path:
    """Path to test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """
    Provide isolated temporary directory for each test.

    Design Decision: Function-scoped (not module) to ensure complete
    test isolation. Slower but prevents cross-test contamination.
    """
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        yield Path(tmpdir)
        os.chdir(original_dir)


# -----------------------------------------------------------------------------
# Fixtures: Reproducibility
# -----------------------------------------------------------------------------


@pytest.fixture(scope="function")
def seeded_rng() -> np.random.Generator:
    """
    Provide seeded random generator for reproducible tests.

    Design Decision: Uses numpy's new Generator API instead of
    global seed for better isolation between tests.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="function")
def deterministic_seed():
    """Set global numpy seed for legacy code compatibility."""
    np.random.seed(1223)
    yield
    # No cleanup needed - each test gets fresh seed


# -----------------------------------------------------------------------------
# Fixtures: Test Data Factories
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_estimation_data(seeded_rng) -> pd.DataFrame:
    """
    Generate minimal valid estimation dataset.

    Returns:
        DataFrame with Y, D, Z columns meeting schema requirements
    """
    n = 100
    z = seeded_rng.normal(0, 1, n)
    d = (z + seeded_rng.normal(0, 0.5, n) > 0).astype(int)
    y = 1.0 + 0.5 * d + seeded_rng.normal(0, 1, n)

    return pd.DataFrame({"Y": y, "D": d, "Z": z})


@pytest.fixture
def sample_config_dict() -> dict:
    """Minimal valid configuration dictionary."""
    return {
        "ESTIMATION": {
            "method": "parametric",
            "file": "data.pkl",
            "dependent": "Y",
            "treatment": "D",
        },
        "SIMULATION": {
            "agents": 1000,
            "seed": 42,
        },
    }


@pytest.fixture
def sample_simulation_config(temp_directory) -> Path:
    """Create minimal valid simulation config file."""
    import yaml

    config = {
        "SIMULATION": {
            "agents": 100,
            "seed": 42,
            "output_file": "test_data.pkl",
        },
        "TREATED": {"coeff": [1.0, 0.5]},
        "UNTREATED": {"coeff": [0.5, 0.3]},
        "CHOICE": {"coeff": [0.0, 1.0]},
        "DIST": {"coeff": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    }

    config_path = temp_directory / "test_simulation.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def invalid_config_missing_field(temp_directory) -> Path:
    """Config file missing required field for error testing."""
    import yaml

    config = {"SIMULATION": {"agents": 100}}  # Missing other required fields

    config_path = temp_directory / "invalid.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


# -----------------------------------------------------------------------------
# Fixtures: Mock Objects
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_estimator():
    """
    Provide mock estimator for testing manager/factory without real computation.

    Design Decision: Uses protocol-based mock to verify interface compliance
    without coupling to specific implementations.
    """
    from unittest.mock import MagicMock

    from grmpy.estimators.base import Estimator

    mock = MagicMock(spec=Estimator)
    mock.validate_connection.return_value = True
    mock.get_required_columns.return_value = ["Y", "D", "Z"]
    mock.fit.return_value = {
        "mte": np.zeros(10),
        "mte_x": np.zeros((100, 10)),
        "mte_u": np.zeros(10),
        "quantiles": np.linspace(0, 1, 10),
        "b0": np.array([1.0]),
        "b1": np.array([0.5]),
    }
    return mock


# -----------------------------------------------------------------------------
# Fixtures: Regression Vault
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def regression_vault(test_resources_dir) -> dict:
    """
    Load regression test expected values.

    Design Decision: Session-scoped to load vault once. JSON format
    for human-readable diffs in version control.
    """
    import json

    vault_dir = test_resources_dir.parent / "regression" / "vault"
    vault = {}

    if vault_dir.exists():
        for vault_file in vault_dir.glob("*.json"):
            with open(vault_file) as f:
                vault[vault_file.stem] = json.load(f)

    return vault
