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
- User-facing API breaking changes (backward compatibility maintained)

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

---

## Backward Compatibility Strategy

### Maintained APIs
```python
# These public APIs remain unchanged
grmpy.fit(init_file)           # Works as before
grmpy.simulate(init_file)      # Works as before
grmpy.plot_mte(rslt, init_file) # Works as before
```

### Deprecation Path
```python
# Old internal imports (deprecated with warnings)
from grmpy.estimate.estimate import fit  # DeprecationWarning

# New internal imports (recommended)
from grmpy.estimators.factory import create_estimator_manager
```

---

## Success Criteria

1. All existing tests pass without modification
2. Public API (`fit`, `simulate`, `plot_mte`) unchanged
3. New estimators can be registered without modifying core code
4. Configuration defaults are externalized to YAML
5. All functions have type hints
6. Documentation explains design decisions (why, not what)
7. Code passes ruff and black with specified configuration

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing user code | High | Maintain backward-compatible wrapper functions |
| Over-engineering for academic use case | Medium | Keep abstractions minimal, document extension points |
| Test coverage gaps during migration | High | Migrate tests alongside code, maintain CI/CD |
| Performance regression | Low | Benchmark before/after, profile critical paths |

---

## References

- impact-engine architecture: `eisenhauerIO/tools-impact-engine`
- Coding standards: `eisenhauerIO/support-llm-instruction/coding/coding-standards.md`
- Documentation guidelines: `eisenhauerIO/support-llm-instruction/coding/coding-docs.md`
