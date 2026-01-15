# grmpy Architecture Upgrade Plan

## Purpose and Scope

This document outlines the plan to modernize grmpy's architecture by adopting patterns from `tools-impact-engine` and adhering to the `support-llm-instruction` coding standards. The goal is to transform grmpy from a procedural/functional codebase into a maintainable, extensible, enterprise-grade package while preserving its core econometric functionality.

### What This Plan Covers
- Directory restructuring
- Configuration management overhaul
- Interface and contract definitions
- Factory and registry pattern implementation
- Adapter-based module refactoring
- Documentation and coding standards alignment

### What This Plan Does NOT Cover
- Changes to the underlying econometric algorithms
- New statistical methods or models

### Breaking Changes Accepted
- Public API will change to align with new architecture
- No backward compatibility layer required
- Clean slate approach for simpler, maintainable codebase

---

## Phase 1: Project Structure Reorganization

### Current Structure
```
grmpy/
├── __init__.py
├── __version__.py
├── grmpy_config.py
├── simulate/
├── estimate/
├── check/
├── read/
├── plot/
└── test/
```

### Target Structure
```
science/
└── grmpy/
    ├── __init__.py
    ├── __version__.py
    ├── config.py                    # Central configuration management
    ├── engine.py                    # Main orchestrator
    ├── core/
    │   ├── __init__.py
    │   ├── contracts.py             # Data schemas and contracts
    │   ├── config_bridge.py         # Config transformation layer
    │   └── exceptions.py            # Custom exceptions
    ├── estimators/                  # Renamed from estimate/
    │   ├── __init__.py
    │   ├── base.py                  # Abstract estimator interface
    │   ├── factory.py               # Estimator registry and factory
    │   ├── manager.py               # Estimator orchestration
    │   ├── adapter_parametric.py    # Parametric normal model
    │   └── adapter_semiparametric.py # LIV semiparametric model
    ├── simulators/                  # Renamed from simulate/
    │   ├── __init__.py
    │   ├── base.py                  # Abstract simulator interface
    │   ├── factory.py               # Simulator registry and factory
    │   ├── manager.py               # Simulation orchestration
    │   └── adapter_roy_model.py     # Generalized Roy model simulator
    ├── visualization/               # Renamed from plot/
    │   ├── __init__.py
    │   ├── base.py                  # Abstract visualizer interface
    │   ├── factory.py               # Visualizer registry
    │   └── adapter_mte_plot.py      # MTE plotting adapter
    └── tests/
        ├── __init__.py
        ├── conftest.py
        ├── unit/
        ├── integration/
        └── resources/
docs/
development/
deployment/
```

### Rationale
- Aligns with impact-engine's `science/` directory convention
- Separates concerns into distinct layers (core, estimators, simulators, visualization)
- Enables independent testing and development of each layer
- Follows single-responsibility principle per module

### Migration Steps
1. Create new directory structure
2. Move files with git mv to preserve history
3. Update all import statements
4. Update pyproject.toml build paths
5. Verify all tests pass

---

## Phase 2: Configuration Management System

### Current Approach
- YAML files read directly via `read/read.py`
- No default values management
- Validation scattered across `check/` module
- Configuration dictionaries passed as raw dicts

### Target Approach

#### 2.1 Single Entry Point
```python
# science/grmpy/config.py

def process_config(path: str) -> Config:
    """
    Single public API for configuration processing.

    Args:
        path: Path to user YAML configuration file

    Returns:
        Fully validated, merged Config object

    Raises:
        ConfigurationError: If validation fails with context-rich message
    """
    user_config = _load_yaml(path)
    defaults = _load_defaults()
    merged = _deep_merge(defaults, user_config)
    validated = _validate_config(merged)
    return Config.from_dict(validated)
```

#### 2.2 Default Values in YAML
```yaml
# science/grmpy/defaults/estimation_defaults.yml
estimation:
  method: parametric
  optimizer: BFGS
  max_iterations: 10000
  tolerance: 1e-6

# science/grmpy/defaults/simulation_defaults.yml
simulation:
  agents: 1000
  seed: null
  output_format: pickle
```

#### 2.3 Configuration Dataclass
```python
# science/grmpy/core/contracts.py

@dataclass
class EstimationConfig:
    """Configuration contract for estimation operations."""
    method: str
    file: str
    optimizer: str = "BFGS"
    max_iterations: int = 10000
    tolerance: float = 1e-6

    def validate(self) -> None:
        """Validate configuration with context-rich errors."""
        valid_methods = ["parametric", "semiparametric"]
        if self.method not in valid_methods:
            raise ConfigurationError(
                f"Invalid estimation method: '{self.method}'. "
                f"Available options: {valid_methods}"
            )
```

### Rationale
- Configuration-driven behavior (not hardcoded)
- Single source of truth for defaults
- Validation at load time with helpful messages
- Type-safe configuration objects downstream

---

## Phase 3: Base Interfaces and Contracts

### 3.1 Estimator Interface
```python
# science/grmpy/estimators/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class Estimator(ABC):
    """
    Abstract interface for all estimation methods.

    Design Decision: Separates data transformation (outbound/inbound) from
    fitting logic to enable different data formats without modifying core
    estimation algorithms.
    """

    @abstractmethod
    def connect(self, config: EstimationConfig) -> None:
        """Initialize estimator with configuration."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Verify estimator is properly configured."""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data meets estimator requirements.

        Raises:
            DataValidationError: With specific column/type issues
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        pass

    @abstractmethod
    def transform_outbound(self, data: pd.DataFrame) -> Any:
        """Transform standard format to estimator-specific format."""
        pass

    @abstractmethod
    def fit(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute estimation and return results.

        Returns:
            Dictionary containing: mte, quantiles, coefficients, etc.
        """
        pass

    @abstractmethod
    def transform_inbound(self, results: Any) -> Dict[str, Any]:
        """Transform estimator results to standard output format."""
        pass
```

### 3.2 Data Contracts
```python
# science/grmpy/core/contracts.py

@dataclass
class DataSchema:
    """
    Base schema for data validation and transformation.

    Design Decision: Explicit field mappings enable integration with
    different data sources without modifying core logic.
    """
    required_fields: list[str]
    optional_fields: list[str] = field(default_factory=list)
    field_mappings: Dict[str, str] = field(default_factory=dict)

    def validate(self, df: pd.DataFrame) -> None:
        """Validate DataFrame contains required columns."""
        missing = set(self.required_fields) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def from_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply field mappings from external format."""
        return df.rename(columns=self.field_mappings)

    def to_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply reverse mappings to external format."""
        reverse = {v: k for k, v in self.field_mappings.items()}
        return df.rename(columns=reverse)


@dataclass
class EstimationDataSchema(DataSchema):
    """Schema for estimation input data."""
    required_fields: list[str] = field(default_factory=lambda: [
        "Y", "D", "Z"  # Outcome, Treatment, Instrument
    ])
    optional_fields: list[str] = field(default_factory=lambda: [
        "X"  # Covariates
    ])


@dataclass
class SimulationDataSchema(DataSchema):
    """Schema for simulation output data."""
    required_fields: list[str] = field(default_factory=lambda: [
        "Y", "Y_1", "Y_0", "D", "U_1", "U_0", "V"
    ])
```

### 3.3 Result Contracts
```python
@dataclass
class EstimationResult:
    """
    Standardized estimation output contract.

    Constraint: All estimators must return results conforming to this
    schema, enabling consistent downstream processing.
    """
    mte: np.ndarray
    mte_x: np.ndarray
    mte_u: np.ndarray
    quantiles: np.ndarray
    coefficients: Dict[str, np.ndarray]
    standard_errors: Optional[Dict[str, np.ndarray]] = None
    confidence_intervals: Optional[Dict[str, tuple]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Rationale
- Explicit interfaces enable polymorphic behavior
- Data contracts prevent integration errors
- Separation of transformation from core logic
- Consistent output format regardless of method

---

## Phase 4: Factory and Registry Pattern

### 4.1 Estimator Registry
```python
# science/grmpy/estimators/factory.py

from typing import Type, Dict
from grmpy.estimators.base import Estimator
from grmpy.estimators.adapter_parametric import ParametricEstimator
from grmpy.estimators.adapter_semiparametric import SemiparametricEstimator

# Registry mapping method names to implementations
ESTIMATOR_REGISTRY: Dict[str, Type[Estimator]] = {
    "parametric": ParametricEstimator,
    "semiparametric": SemiparametricEstimator,
}


def register_estimator(name: str, estimator_class: Type[Estimator]) -> None:
    """
    Register a custom estimator at runtime.

    Design Decision: Enables extension without modifying core code.
    Users can add custom estimators for specialized use cases.

    Args:
        name: Identifier for the estimator
        estimator_class: Class implementing Estimator interface

    Raises:
        TypeError: If class doesn't implement Estimator interface
    """
    if not issubclass(estimator_class, Estimator):
        raise TypeError(
            f"Estimator class must inherit from Estimator base class. "
            f"Got: {estimator_class.__bases__}"
        )
    ESTIMATOR_REGISTRY[name] = estimator_class


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
    estimator = get_estimator(config.estimation.method)
    return EstimatorManager(estimator, config)
```

### 4.2 Estimator Manager
```python
# science/grmpy/estimators/manager.py

class EstimatorManager:
    """
    Orchestrates estimation workflow.

    Design Decision: Separates workflow coordination from estimation
    logic. Manager handles data loading, validation, and result
    formatting while delegating statistical work to adapters.
    """

    def __init__(self, estimator: Estimator, config: Config):
        self.estimator = estimator
        self.config = config
        self._is_connected = False

    def connect(self) -> "EstimatorManager":
        """Initialize estimator connection."""
        self.estimator.connect(self.config.estimation)
        if not self.estimator.validate_connection():
            raise ConnectionError("Estimator failed validation")
        self._is_connected = True
        return self

    def fit(self, data: pd.DataFrame) -> EstimationResult:
        """
        Execute full estimation pipeline.

        Pipeline:
        1. Validate input data
        2. Transform to estimator format
        3. Fit model
        4. Transform results to standard format
        """
        if not self._is_connected:
            raise RuntimeError("Must call connect() before fit()")

        self.estimator.validate_data(data)
        transformed = self.estimator.transform_outbound(data)
        raw_results = self.estimator.fit(transformed)
        return self.estimator.transform_inbound(raw_results)
```

### Rationale
- Open/closed principle: extend without modifying
- Single responsibility: factory creates, manager coordinates
- Dependency inversion: depend on abstractions (Estimator)
- Runtime extensibility for custom research methods

---

## Phase 5: Adapter Implementations

### 5.1 Parametric Estimator Adapter
```python
# science/grmpy/estimators/adapter_parametric.py

class ParametricEstimator(Estimator):
    """
    Parametric normal model estimator using maximum likelihood.

    Assumption: Joint normality of unobservables (U_1, U_0, V).

    This adapter wraps the existing par_fit() logic while conforming
    to the Estimator interface for consistent orchestration.
    """

    def __init__(self):
        self.config: Optional[EstimationConfig] = None
        self._optimizer_options: Dict[str, Any] = {}

    def connect(self, config: EstimationConfig) -> None:
        self.config = config
        self._optimizer_options = {
            "method": config.optimizer,
            "maxiter": config.max_iterations,
            "gtol": config.tolerance,
        }

    def validate_connection(self) -> bool:
        return self.config is not None

    def get_required_columns(self) -> list[str]:
        return ["Y", "D", "Z"]  # Plus covariates from config

    def validate_data(self, data: pd.DataFrame) -> None:
        schema = EstimationDataSchema()
        schema.validate(data)
        # Additional parametric-specific validation
        self._validate_covariance_structure(data)

    def transform_outbound(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform DataFrame to numpy arrays for optimization."""
        return {
            "Y": data["Y"].values,
            "D": data["D"].values,
            "Z": data["Z"].values,
            "X": data[self.config.covariates].values if self.config.covariates else None,
        }

    def fit(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Execute maximum likelihood estimation."""
        # Core estimation logic (migrated from estimate_par.py)
        # ... optimization code ...
        return raw_results

    def transform_inbound(self, results: Dict[str, Any]) -> EstimationResult:
        """Convert raw optimization results to standard format."""
        return EstimationResult(
            mte=results["mte"],
            mte_x=results["mte_x"],
            mte_u=results["mte_u"],
            quantiles=results["quantiles"],
            coefficients={"b0": results["b0"], "b1": results["b1"]},
            metadata={"method": "parametric", "optimizer": self.config.optimizer}
        )
```

### 5.2 Semiparametric Estimator Adapter
```python
# science/grmpy/estimators/adapter_semiparametric.py

class SemiparametricEstimator(Estimator):
    """
    Local Instrumental Variables (LIV) semiparametric estimator.

    Assumption: No distributional assumptions on unobservables.

    Design Decision: Uses local polynomial regression for flexibility
    at the cost of requiring larger sample sizes.
    """

    # Implementation follows same pattern as ParametricEstimator
    # with LIV-specific logic in fit() method
```

### Rationale
- Preserves existing statistical algorithms
- Wraps legacy code in modern interface
- Enables gradual migration without breaking changes
- Clear separation between interface and implementation

---

## Phase 6: Main Engine Orchestrator

```python
# science/grmpy/engine.py

def fit(config_path: str) -> EstimationResult:
    """
    Main entry point for estimation.

    This function serves as the primary public API, orchestrating
    the complete estimation workflow.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        EstimationResult with MTE and related quantities

    Example:
        >>> result = grmpy.fit("analysis.grmpy.yml")
        >>> print(result.mte)
    """
    from grmpy.estimators.factory import create_estimator_manager
    from grmpy.core.contracts import EstimationDataSchema

    # Create configured manager
    manager = create_estimator_manager(config_path)
    manager.connect()

    # Load and validate data
    config = process_config(config_path)
    data = pd.read_pickle(config.estimation.file)

    # Execute estimation
    result = manager.fit(data)

    return result


def simulate(config_path: str) -> pd.DataFrame:
    """
    Main entry point for simulation.

    Generates synthetic data according to the generalized Roy model.
    """
    from grmpy.simulators.factory import create_simulator_manager

    manager = create_simulator_manager(config_path)
    manager.connect()

    return manager.simulate()
```

---

## Phase 7: Coding Standards Alignment

### 7.1 Formatting Configuration
```toml
# pyproject.toml updates

[tool.black]
line-length = 120
target-version = ['py311']

[tool.ruff]
line-length = 120
target-version = "py311"
select = ["E", "W", "F", "I"]

[tool.ruff.isort]
known-first-party = ["grmpy"]
```

### 7.2 Type Hints (Required)
```python
# All function signatures must have type hints
def fit(
    config_path: str,
    *,
    validate: bool = True,
    verbose: bool = False,
) -> EstimationResult:
    ...
```

### 7.3 Docstring Format
```python
def calculate_mte(
    coefficients: Dict[str, np.ndarray],
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Calculate Marginal Treatment Effect across quantiles.

    Args:
        coefficients: Dictionary with 'b0' and 'b1' coefficient arrays
        quantiles: Array of quantile points for evaluation

    Returns:
        Array of MTE values at each quantile point

    Raises:
        ValueError: If coefficient arrays have mismatched dimensions
    """
```

### 7.4 Import Organization
```python
# Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

# Third-party
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Local
from grmpy.core.contracts import EstimationConfig, EstimationResult
from grmpy.core.exceptions import ConfigurationError
```

---

## Phase 8: Documentation Updates

### 8.1 Architecture Documentation
Create `docs/source/architecture.md` explaining:
- Why the layered architecture exists
- Design decisions and trade-offs
- Extension points and how to use them
- Constraints and assumptions

### 8.2 Module-Level Docstrings
Each module must have a docstring explaining:
- Purpose and scope
- What it does NOT do
- Key design decisions
- Dependencies and assumptions

### 8.3 API Reference
Update Sphinx configuration to auto-generate API docs from docstrings.

---

## Implementation Sequence

| Phase | Description | Dependencies | Estimated Complexity |
|-------|-------------|--------------|---------------------|
| 1 | Directory restructure | None | Low |
| 2 | Configuration system | Phase 1 | Medium |
| 3 | Interfaces and contracts | Phase 1 | Medium |
| 4 | Factory and registry | Phase 3 | Low |
| 5 | Adapter implementations | Phases 3, 4 | High |
| 6 | Engine orchestrator | Phases 2, 4, 5 | Medium |
| 7 | Coding standards | All phases | Low |
| 8 | Documentation | All phases | Medium |
| 9 | Testing strategy | Phases 1-6 | High |

---

## Phase 9: Testing Strategy

### Current State Analysis

| Issue | Impact |
|-------|--------|
| All 25 tests currently **SKIPPED** | No CI protection |
| No mocking - full integration only | Slow test execution |
| No parametrization | Repetitive test code |
| No coverage enforcement | Unknown gaps |
| Regression vault in JSON/PKL | Hard to review changes |

### Target Test Architecture

```
science/grmpy/tests/
├── __init__.py
├── conftest.py                    # Shared fixtures, markers
├── unit/
│   ├── __init__.py
│   ├── test_config.py             # Configuration parsing/validation
│   ├── test_contracts.py          # Schema validation
│   ├── test_factory.py            # Registry and factory functions
│   ├── estimators/
│   │   ├── test_base.py           # Interface compliance
│   │   ├── test_parametric.py     # Parametric adapter unit tests
│   │   └── test_semiparametric.py # Semiparametric adapter unit tests
│   ├── simulators/
│   │   └── test_roy_model.py      # Simulator unit tests
│   └── visualization/
│       └── test_mte_plot.py       # Plot generation tests
├── integration/
│   ├── __init__.py
│   ├── test_estimation_pipeline.py    # End-to-end estimation
│   ├── test_simulation_pipeline.py    # End-to-end simulation
│   └── test_replication.py            # Carneiro et al. replication
├── regression/
│   ├── __init__.py
│   ├── test_numerical_accuracy.py     # Regression tests against vault
│   └── vault/                          # Expected results (version controlled)
│       ├── parametric_mte.json
│       ├── semiparametric_mte.json
│       └── simulation_outputs.json
└── resources/
    ├── configs/                   # Test configuration files
    └── data/                      # Test datasets
```

### 9.1 Test Configuration (conftest.py)

```python
# science/grmpy/tests/conftest.py

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
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "regression: marks regression tests against vault")


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
def sample_simulation_config(test_resources_dir) -> Path:
    """Path to minimal valid simulation config."""
    return test_resources_dir / "configs" / "minimal_simulation.yml"


@pytest.fixture
def invalid_config_missing_field(temp_directory) -> Path:
    """Config file missing required field for error testing."""
    config_path = temp_directory / "invalid.yml"
    config_path.write_text("simulation:\n  agents: 100\n")  # Missing required fields
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
    mock.fit.return_value = {"mte": np.zeros(10), "quantiles": np.linspace(0, 1, 10)}
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
    for vault_file in vault_dir.glob("*.json"):
        with open(vault_file) as f:
            vault[vault_file.stem] = json.load(f)
    return vault
```

### 9.2 Unit Test Patterns

#### Contract Testing
```python
# science/grmpy/tests/unit/test_contracts.py

import pytest
import pandas as pd
import numpy as np
from grmpy.core.contracts import (
    EstimationDataSchema,
    SimulationDataSchema,
    EstimationResult,
    DataValidationError,
)


class TestEstimationDataSchema:
    """Tests for estimation input data validation."""

    def test_validate_accepts_valid_dataframe(self, sample_estimation_data):
        """Schema accepts DataFrame with all required columns."""
        schema = EstimationDataSchema()
        schema.validate(sample_estimation_data)  # Should not raise

    def test_validate_rejects_missing_columns(self):
        """Schema raises DataValidationError listing missing columns."""
        schema = EstimationDataSchema()
        incomplete_df = pd.DataFrame({"Y": [1, 2], "D": [0, 1]})  # Missing Z

        with pytest.raises(DataValidationError) as exc_info:
            schema.validate(incomplete_df)

        assert "Z" in str(exc_info.value)
        assert "Missing required columns" in str(exc_info.value)

    @pytest.mark.parametrize("missing_col", ["Y", "D", "Z"])
    def test_validate_identifies_each_missing_column(self, missing_col):
        """Schema correctly identifies which specific column is missing."""
        schema = EstimationDataSchema()
        cols = {"Y": [1], "D": [1], "Z": [1]}
        del cols[missing_col]

        with pytest.raises(DataValidationError) as exc_info:
            schema.validate(pd.DataFrame(cols))

        assert missing_col in str(exc_info.value)

    def test_from_external_applies_field_mappings(self):
        """Schema transforms external field names to standard names."""
        schema = EstimationDataSchema()
        schema.field_mappings = {"outcome": "Y", "treatment": "D", "instrument": "Z"}

        external_df = pd.DataFrame({
            "outcome": [1, 2],
            "treatment": [0, 1],
            "instrument": [0.5, -0.5]
        })

        result = schema.from_external(external_df)

        assert list(result.columns) == ["Y", "D", "Z"]


class TestEstimationResult:
    """Tests for estimation result dataclass."""

    def test_result_stores_all_required_fields(self):
        """Result object stores MTE and related quantities."""
        result = EstimationResult(
            mte=np.array([0.1, 0.2]),
            mte_x=np.array([0.1]),
            mte_u=np.array([0.1]),
            quantiles=np.array([0.25, 0.75]),
            coefficients={"b0": np.array([1.0]), "b1": np.array([0.5])},
        )

        assert len(result.mte) == 2
        assert result.quantiles[0] == 0.25

    def test_result_metadata_defaults_to_empty_dict(self):
        """Metadata field defaults to empty dict if not provided."""
        result = EstimationResult(
            mte=np.zeros(1),
            mte_x=np.zeros(1),
            mte_u=np.zeros(1),
            quantiles=np.zeros(1),
            coefficients={},
        )

        assert result.metadata == {}
```

#### Factory Testing
```python
# science/grmpy/tests/unit/test_factory.py

import pytest
from grmpy.estimators.base import Estimator
from grmpy.estimators.factory import (
    ESTIMATOR_REGISTRY,
    register_estimator,
    get_estimator,
)


class TestEstimatorRegistry:
    """Tests for estimator registration and retrieval."""

    def test_get_estimator_returns_parametric(self):
        """Factory returns ParametricEstimator for 'parametric' key."""
        estimator = get_estimator("parametric")
        assert isinstance(estimator, Estimator)

    def test_get_estimator_returns_semiparametric(self):
        """Factory returns SemiparametricEstimator for 'semiparametric' key."""
        estimator = get_estimator("semiparametric")
        assert isinstance(estimator, Estimator)

    def test_get_estimator_raises_for_unknown_method(self):
        """Factory raises ValueError with available options for unknown method."""
        with pytest.raises(ValueError) as exc_info:
            get_estimator("unknown_method")

        assert "unknown_method" in str(exc_info.value)
        assert "parametric" in str(exc_info.value)  # Lists available

    def test_register_estimator_adds_to_registry(self):
        """Custom estimator can be registered at runtime."""
        class CustomEstimator(Estimator):
            # Minimal implementation for test
            pass

        register_estimator("custom", CustomEstimator)

        assert "custom" in ESTIMATOR_REGISTRY
        # Cleanup
        del ESTIMATOR_REGISTRY["custom"]

    def test_register_estimator_rejects_non_estimator_class(self):
        """Registration fails for classes not implementing Estimator."""
        class NotAnEstimator:
            pass

        with pytest.raises(TypeError) as exc_info:
            register_estimator("invalid", NotAnEstimator)

        assert "Estimator base class" in str(exc_info.value)
```

#### Interface Compliance Testing
```python
# science/grmpy/tests/unit/estimators/test_base.py

import pytest
from abc import ABC
from grmpy.estimators.base import Estimator
from grmpy.estimators.adapter_parametric import ParametricEstimator
from grmpy.estimators.adapter_semiparametric import SemiparametricEstimator


class TestEstimatorInterface:
    """Verify all estimator implementations satisfy the interface contract."""

    @pytest.mark.parametrize("estimator_class", [
        ParametricEstimator,
        SemiparametricEstimator,
    ])
    def test_implements_all_abstract_methods(self, estimator_class):
        """Estimator subclass implements all required abstract methods."""
        # If any abstract method is missing, instantiation will raise TypeError
        instance = estimator_class()

        # Verify key methods exist and are callable
        assert callable(instance.connect)
        assert callable(instance.validate_connection)
        assert callable(instance.validate_data)
        assert callable(instance.get_required_columns)
        assert callable(instance.transform_outbound)
        assert callable(instance.fit)
        assert callable(instance.transform_inbound)

    @pytest.mark.parametrize("estimator_class", [
        ParametricEstimator,
        SemiparametricEstimator,
    ])
    def test_get_required_columns_returns_list(self, estimator_class):
        """get_required_columns returns a list of strings."""
        instance = estimator_class()
        columns = instance.get_required_columns()

        assert isinstance(columns, list)
        assert all(isinstance(col, str) for col in columns)
```

### 9.3 Integration Test Patterns

```python
# science/grmpy/tests/integration/test_estimation_pipeline.py

import pytest
import pandas as pd
import numpy as np
from grmpy.engine import fit, simulate
from grmpy.core.contracts import EstimationResult


@pytest.mark.integration
@pytest.mark.slow
class TestEstimationPipeline:
    """End-to-end tests for complete estimation workflow."""

    def test_parametric_estimation_produces_valid_result(
        self, temp_directory, test_resources_dir
    ):
        """
        Full parametric estimation pipeline returns valid EstimationResult.

        Pipeline: config → simulate → fit → result
        """
        config_path = test_resources_dir / "configs" / "parametric_test.yml"

        # Run simulation to generate data
        simulated_data = simulate(str(config_path))
        assert isinstance(simulated_data, pd.DataFrame)
        assert len(simulated_data) > 0

        # Run estimation
        result = fit(str(config_path))

        # Verify result contract
        assert isinstance(result, EstimationResult)
        assert result.mte is not None
        assert len(result.mte) == len(result.quantiles)
        assert "b0" in result.coefficients
        assert "b1" in result.coefficients

    def test_semiparametric_estimation_produces_valid_result(
        self, temp_directory, test_resources_dir
    ):
        """
        Full semiparametric estimation pipeline returns valid EstimationResult.
        """
        config_path = test_resources_dir / "configs" / "semiparametric_test.yml"

        result = fit(str(config_path))

        assert isinstance(result, EstimationResult)
        assert result.metadata.get("method") == "semiparametric"

    def test_estimation_fails_gracefully_with_invalid_config(
        self, invalid_config_missing_field
    ):
        """
        Estimation raises ConfigurationError with helpful message for invalid config.
        """
        from grmpy.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            fit(str(invalid_config_missing_field))

        # Error message should guide user to fix the issue
        assert "missing" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()
```

### 9.4 Regression Test Patterns

```python
# science/grmpy/tests/regression/test_numerical_accuracy.py

import pytest
import numpy as np
from grmpy.engine import fit


@pytest.mark.regression
class TestNumericalAccuracy:
    """
    Regression tests ensuring numerical results match expected values.

    Design Decision: Uses JSON vault for expected values to enable
    human-readable diffs in pull request reviews. Tolerances set based
    on statistical precision requirements.
    """

    DECIMAL_PRECISION = 6  # Match to 6 decimal places

    def test_parametric_mte_matches_vault(
        self, test_resources_dir, regression_vault, deterministic_seed
    ):
        """
        Parametric MTE computation matches regression vault values.

        Constraint: Any change to MTE values must be intentional and
        reviewed. Vault update requires explicit justification.
        """
        config_path = test_resources_dir / "configs" / "regression_parametric.yml"

        result = fit(str(config_path))
        expected = regression_vault["parametric_mte"]

        np.testing.assert_array_almost_equal(
            result.mte,
            np.array(expected["mte"]),
            decimal=self.DECIMAL_PRECISION,
            err_msg="MTE values changed from regression vault"
        )

    def test_semiparametric_mte_matches_vault(
        self, test_resources_dir, regression_vault, deterministic_seed
    ):
        """Semiparametric MTE computation matches regression vault values."""
        config_path = test_resources_dir / "configs" / "regression_semiparametric.yml"

        result = fit(str(config_path))
        expected = regression_vault["semiparametric_mte"]

        np.testing.assert_array_almost_equal(
            result.mte,
            np.array(expected["mte"]),
            decimal=self.DECIMAL_PRECISION,
        )

    def test_carneiro_replication_matches_published_results(
        self, test_resources_dir, regression_vault, deterministic_seed
    ):
        """
        Carneiro et al. (2011) replication matches published MTE curve.

        Reference: Heckman, Urzua, Vytlacil (2006) / Carneiro et al. (2011)
        This test validates the core econometric implementation.
        """
        config_path = test_resources_dir / "configs" / "carneiro_replication.yml"

        result = fit(str(config_path))
        expected = regression_vault["carneiro_replication"]

        np.testing.assert_array_almost_equal(
            result.mte,
            np.array(expected["mte"]),
            decimal=4,  # Published precision
            err_msg="Carneiro replication diverged from published results"
        )
```

### 9.5 Test Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["science/grmpy/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
    "regression: marks regression tests against vault",
]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["science/grmpy"]
branch = true
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true
```

### 9.6 CI/CD Pipeline Updates

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-unit:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install hatch
      - name: Run unit tests
        run: hatch run test:unit

  test-integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install hatch
      - name: Run integration tests
        run: hatch run test:integration

  test-regression:
    name: Regression Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install hatch
      - name: Run regression tests
        run: hatch run test:regression

  coverage:
    name: Coverage Report
    runs-on: ubuntu-latest
    needs: [test-unit]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install hatch
      - name: Generate coverage
        run: hatch run test:coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
          fail_ci_if_error: true

  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install hatch
      - name: Run linter
        run: hatch run lint
      - name: Check formatting
        run: hatch run format --check
```

### 9.7 Hatch Environment Scripts

```toml
# pyproject.toml additions

[tool.hatch.envs.test]
dependencies = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",  # Parallel execution
]

[tool.hatch.envs.test.scripts]
unit = "pytest science/grmpy/tests/unit -v"
integration = "pytest science/grmpy/tests/integration -v -m integration"
regression = "pytest science/grmpy/tests/regression -v -m regression"
all = "pytest science/grmpy/tests -v"
fast = "pytest science/grmpy/tests -v -m 'not slow'"
coverage = "pytest science/grmpy/tests --cov=science/grmpy --cov-report=xml --cov-report=term-missing"
parallel = "pytest science/grmpy/tests -v -n auto"  # Use all CPUs
```

### 9.8 Testing Migration Strategy

| Step | Action | Validates |
|------|--------|-----------|
| 1 | Create new test directory structure | Directory layout |
| 2 | Write conftest.py with fixtures | Test infrastructure |
| 3 | Write contract tests first | Schema definitions |
| 4 | Write factory/registry tests | Extension mechanism |
| 5 | Migrate unit tests with mocking | Component isolation |
| 6 | Migrate integration tests | Pipeline correctness |
| 7 | Convert regression vault to JSON | Reviewable expected values |
| 8 | Enable coverage enforcement | Quality gate |
| 9 | Remove skip markers from old tests | Full CI protection |

---

## Success Criteria

1. **Tests enabled and passing** - All tests run in CI (no skipped tests)
2. **Coverage ≥ 80%** - Enforced in CI pipeline
3. **Regression vault in JSON** - Human-readable diffs in PRs
4. **Test isolation** - Each test runs independently
5. **Fast feedback** - Unit tests complete in < 30 seconds
6. **New estimators testable** - Interface compliance tests auto-include new adapters
7. **Configuration defaults externalized** - YAML files for defaults
8. **All functions have type hints** - Enforced by ruff
9. **Documentation explains why** - Design rationale captured
10. **Code passes ruff and black** - 120 char lines, consistent style

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Numerical precision drift | High | Regression vault with strict tolerances |
| Slow test suite | Medium | Parallel execution, fast/slow markers |
| Flaky tests from randomness | Medium | Seeded RNG fixtures, deterministic mode |
| Coverage gaming | Low | Review coverage reports, require meaningful tests |
| Breaking changes undetected | High | Integration tests cover full pipeline |

---

## References

- impact-engine architecture: `eisenhauerIO/tools-impact-engine`
- Coding standards: `eisenhauerIO/support-llm-instruction/coding/coding-standards.md`
- Documentation guidelines: `eisenhauerIO/support-llm-instruction/coding/coding-docs.md`
