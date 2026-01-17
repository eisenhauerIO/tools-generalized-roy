"""
Data contracts for grmpy package.

Design Decision: Explicit contracts enable consistent data handling across
modules and provide clear validation at system boundaries.

Design Decision: FUNCTION + PARAMS pattern matches impact-engine for
consistency across the organization's tooling.

Contracts define:
- Required and optional fields for data structures
- Field mappings for external data sources
- Result schemas for consistent output format
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grmpy.core.exceptions import GrmpyError

# -----------------------------------------------------------------------------
# Data Schemas
# -----------------------------------------------------------------------------


@dataclass
class DataSchema:
    """
    Base schema for data validation and transformation.

    Design Decision: Explicit field mappings enable integration with
    different data sources without modifying core logic.
    """

    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_mappings: Dict[str, str] = field(default_factory=dict)

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame contains required columns.

        Args:
            df: DataFrame to validate

        Raises:
            DataValidationError: If required columns are missing
        """
        missing = set(self.required_fields) - set(df.columns)
        if missing:
            raise GrmpyError(
                f"Missing required columns: {sorted(missing)}. " f"Available columns: {sorted(df.columns)}"
            )

    def from_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply field mappings from external format to standard format.

        Args:
            df: DataFrame with external column names

        Returns:
            DataFrame with standard column names
        """
        return df.rename(columns=self.field_mappings)

    def to_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reverse mappings to external format.

        Args:
            df: DataFrame with standard column names

        Returns:
            DataFrame with external column names
        """
        reverse = {v: k for k, v in self.field_mappings.items()}
        return df.rename(columns=reverse)

    def all_fields(self) -> List[str]:
        """Return all defined fields (required + optional)."""
        return self.required_fields + self.optional_fields


@dataclass
class EstimationDataSchema(DataSchema):
    """
    Schema for estimation input data.

    Required columns:
    - Y: Outcome variable
    - D: Treatment indicator (binary)
    - Z: Instrument(s)

    Optional columns:
    - X: Covariates
    """

    required_fields: List[str] = field(default_factory=lambda: ["Y", "D", "Z"])
    optional_fields: List[str] = field(default_factory=lambda: ["X"])


@dataclass
class SimulationDataSchema(DataSchema):
    """
    Schema for simulation output data.

    Contains both observed and latent variables for analysis.
    """

    required_fields: List[str] = field(default_factory=lambda: ["Y", "Y_1", "Y_0", "D", "U_1", "U_0", "V"])


# -----------------------------------------------------------------------------
# Configuration Contracts
# Design Decision: FUNCTION specifies which implementation to use,
# PARAMS contains parameters for that implementation.
# -----------------------------------------------------------------------------


@dataclass
class EstimationConfig:
    """
    Configuration contract for estimation operations.

    Design Decision: Dataclass provides type safety and validation
    at configuration load time. FUNCTION + PARAMS pattern enables
    data-driven selection of estimation method.
    """

    function: str  # Which estimator to use: "parametric", "semiparametric"
    file: str = ""
    dependent: str = "Y"
    treatment: str = "D"
    instrument: str = "Z"
    covariates_treated: List[str] = field(default_factory=list)
    covariates_untreated: List[str] = field(default_factory=list)
    covariates_choice: List[str] = field(default_factory=list)
    optimizer: str = "BFGS"
    max_iterations: int = 10000
    tolerance: float = 1e-6
    start_values: Optional[Dict[str, Any]] = None
    # Semiparametric options
    bandwidth: Optional[float] = None
    gridsize: int = 500
    ps_range: tuple = (0.005, 0.995)
    # Minimum sample size for semiparametric estimation
    # Design Decision: Semiparametric LIV requires sufficient observations
    # for local polynomial regression. 100 is a conservative minimum based
    # on simulation studies showing unreliable estimates with fewer points.
    min_sample_size: int = 100

    def validate(self) -> None:
        """
        Validate configuration with context-rich errors.

        Design Decision: Dynamically imports AVAILABLE_FUNCTIONS from estimators
        module to derive valid options rather than hard-coding.

        Raises:
            GrmpyError: If configuration is invalid.
        """
        from grmpy.core.exceptions import GrmpyError

        # Lazy import to avoid circular dependency
        from grmpy.estimators import AVAILABLE_FUNCTIONS

        if self.function not in AVAILABLE_FUNCTIONS:
            raise GrmpyError(
                f"Invalid estimation FUNCTION: '{self.function}'. " f"Available options: {AVAILABLE_FUNCTIONS}"
            )

        valid_optimizers = ["BFGS", "POWELL", "L-BFGS-B"]
        if self.optimizer not in valid_optimizers:
            raise GrmpyError(f"Invalid optimizer: '{self.optimizer}'. " f"Available options: {valid_optimizers}")


@dataclass
class SimulationConfig:
    """
    Configuration contract for simulation operations.

    Design Decision: FUNCTION + PARAMS pattern enables future
    extension with different simulation models.
    """

    function: str  # Which simulator to use: "roy_model"
    agents: int = 1000
    seed: Optional[int] = None
    source: str = "sim"
    output_file: str = "data.grmpy.pkl"
    # Model parameters
    coefficients_treated: List[float] = field(default_factory=list)
    coefficients_untreated: List[float] = field(default_factory=list)
    coefficients_choice: List[float] = field(default_factory=list)
    covariance: List[List[float]] = field(default_factory=list)

    def validate(self) -> None:
        """
        Validate simulation configuration.

        Design Decision: Dynamically imports AVAILABLE_FUNCTIONS from simulators
        module to derive valid options rather than hard-coding.

        Raises:
            GrmpyError: If configuration is invalid.
        """
        from grmpy.core.exceptions import GrmpyError

        # Lazy import to avoid circular dependency
        from grmpy.simulators import AVAILABLE_FUNCTIONS

        if self.function not in AVAILABLE_FUNCTIONS:
            raise GrmpyError(
                f"Invalid simulation FUNCTION: '{self.function}'. " f"Available options: {AVAILABLE_FUNCTIONS}"
            )

        if self.agents <= 0:
            raise GrmpyError(f"Number of agents must be positive, got: {self.agents}")


@dataclass
class Config:
    """
    Root configuration object containing all sub-configurations.
    """

    estimation: Optional[EstimationConfig] = None
    simulation: Optional[SimulationConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Parses FUNCTION + PARAMS structure:
            ESTIMATION:
              FUNCTION: parametric
              PARAMS:
                file: data.csv
                optimizer: BFGS

        Args:
            data: Configuration dictionary

        Returns:
            Config object with validated sub-configurations
        """
        estimation = None
        simulation = None

        if "ESTIMATION" in data:
            est_data = data["ESTIMATION"]
            function = est_data.get("FUNCTION", "parametric")
            params = est_data.get("PARAMS", {})

            estimation = EstimationConfig(
                function=function,
                file=params.get("file", ""),
                dependent=params.get("dependent", "Y"),
                treatment=params.get("treatment", "D"),
                instrument=params.get("instrument", "Z"),
                covariates_treated=params.get("covariates_treated", []),
                covariates_untreated=params.get("covariates_untreated", []),
                covariates_choice=params.get("covariates_choice", []),
                optimizer=params.get("optimizer", "BFGS"),
                max_iterations=params.get("max_iterations", 10000),
                tolerance=params.get("tolerance", 1e-6),
                start_values=params.get("start_values"),
                bandwidth=params.get("bandwidth"),
                gridsize=params.get("gridsize", 500),
                ps_range=tuple(params.get("ps_range", [0.005, 0.995])),
                min_sample_size=params.get("min_sample_size", 100),
            )

        if "SIMULATION" in data:
            sim_data = data["SIMULATION"]
            function = sim_data.get("FUNCTION", "roy_model")
            params = sim_data.get("PARAMS", {})

            simulation = SimulationConfig(
                function=function,
                agents=params.get("agents", 1000),
                seed=params.get("seed"),
                source=params.get("source", "sim"),
                output_file=params.get("output_file", "data.grmpy.pkl"),
                coefficients_treated=params.get("coefficients_treated", []),
                coefficients_untreated=params.get("coefficients_untreated", []),
                coefficients_choice=params.get("coefficients_choice", []),
                covariance=params.get("covariance", []),
            )

        return cls(estimation=estimation, simulation=simulation)


# -----------------------------------------------------------------------------
# Result Contracts
# -----------------------------------------------------------------------------


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "mte": self.mte.tolist() if isinstance(self.mte, np.ndarray) else self.mte,
            "mte_x": self.mte_x.tolist() if isinstance(self.mte_x, np.ndarray) else self.mte_x,
            "mte_u": self.mte_u.tolist() if isinstance(self.mte_u, np.ndarray) else self.mte_u,
            "quantiles": self.quantiles.tolist() if isinstance(self.quantiles, np.ndarray) else self.quantiles,
            "coefficients": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.coefficients.items()},
            "metadata": self.metadata,
        }


@dataclass
class SimulationResult:
    """
    Standardized simulation output contract.
    """

    data: pd.DataFrame
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
