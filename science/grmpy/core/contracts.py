"""
Data contracts for grmpy package.

Design Decision: Explicit contracts enable consistent data handling across
modules and provide clear validation at system boundaries.

Contracts define:
- Required and optional fields for data structures
- Field mappings for external data sources
- Result schemas for consistent output format
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grmpy.core.exceptions import DataValidationError


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
            raise DataValidationError(
                f"Missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(df.columns)}"
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

    required_fields: List[str] = field(
        default_factory=lambda: ["Y", "D", "Z"]
    )
    optional_fields: List[str] = field(default_factory=lambda: ["X"])


@dataclass
class SimulationDataSchema(DataSchema):
    """
    Schema for simulation output data.

    Contains both observed and latent variables for analysis.
    """

    required_fields: List[str] = field(
        default_factory=lambda: ["Y", "Y_1", "Y_0", "D", "U_1", "U_0", "V"]
    )


# -----------------------------------------------------------------------------
# Configuration Contracts
# -----------------------------------------------------------------------------


@dataclass
class EstimationConfig:
    """
    Configuration contract for estimation operations.

    Design Decision: Dataclass provides type safety and validation
    at configuration load time.
    """

    method: str
    file: str
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

    def validate(self) -> None:
        """
        Validate configuration with context-rich errors.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        from grmpy.core.exceptions import ConfigurationError

        valid_methods = ["parametric", "semiparametric"]
        if self.method not in valid_methods:
            raise ConfigurationError(
                f"Invalid estimation method: '{self.method}'. "
                f"Available options: {valid_methods}"
            )

        valid_optimizers = ["BFGS", "POWELL", "L-BFGS-B"]
        if self.optimizer not in valid_optimizers:
            raise ConfigurationError(
                f"Invalid optimizer: '{self.optimizer}'. "
                f"Available options: {valid_optimizers}"
            )


@dataclass
class SimulationConfig:
    """
    Configuration contract for simulation operations.
    """

    agents: int
    seed: Optional[int] = None
    source: str = "sim"
    output_file: str = "data.grmpy.pkl"
    # Model parameters
    coefficients_treated: List[float] = field(default_factory=list)
    coefficients_untreated: List[float] = field(default_factory=list)
    coefficients_choice: List[float] = field(default_factory=list)
    covariance: List[List[float]] = field(default_factory=list)

    def validate(self) -> None:
        """Validate simulation configuration."""
        from grmpy.core.exceptions import ConfigurationError

        if self.agents <= 0:
            raise ConfigurationError(
                f"Number of agents must be positive, got: {self.agents}"
            )


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

        Args:
            data: Configuration dictionary

        Returns:
            Config object with validated sub-configurations
        """
        estimation = None
        simulation = None

        if "ESTIMATION" in data:
            est_data = data["ESTIMATION"]
            estimation = EstimationConfig(
                method=est_data.get("method", "parametric"),
                file=est_data.get("file", ""),
                dependent=est_data.get("dependent", "Y"),
                treatment=est_data.get("treatment", "D"),
                instrument=est_data.get("instrument", "Z"),
                covariates_treated=est_data.get("covariates_treated", []),
                covariates_untreated=est_data.get("covariates_untreated", []),
                covariates_choice=est_data.get("covariates_choice", []),
                optimizer=est_data.get("optimizer", "BFGS"),
                max_iterations=est_data.get("max_iterations", 10000),
                tolerance=est_data.get("tolerance", 1e-6),
                start_values=est_data.get("start_values"),
                bandwidth=est_data.get("bandwidth"),
                gridsize=est_data.get("gridsize", 500),
                ps_range=tuple(est_data.get("ps_range", [0.005, 0.995])),
            )

        if "SIMULATION" in data:
            sim_data = data["SIMULATION"]
            simulation = SimulationConfig(
                agents=sim_data.get("agents", 1000),
                seed=sim_data.get("seed"),
                source=sim_data.get("source", "sim"),
                output_file=sim_data.get("output_file", "data.grmpy.pkl"),
                coefficients_treated=sim_data.get("coefficients_treated", []),
                coefficients_untreated=sim_data.get("coefficients_untreated", []),
                coefficients_choice=sim_data.get("coefficients_choice", []),
                covariance=sim_data.get("covariance", []),
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
            "coefficients": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.coefficients.items()
            },
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
