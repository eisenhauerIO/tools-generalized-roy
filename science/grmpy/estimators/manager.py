"""
Estimator manager for orchestrating estimation workflow.

Design Decision: Separates workflow coordination from estimation
logic. Manager handles data loading, validation, and result
formatting while delegating statistical work to adapters.
"""

from typing import Optional

import pandas as pd

from grmpy.core.contracts import Config, EstimationResult
from grmpy.core.exceptions import EstimationError
from grmpy.estimators.base import Estimator


class EstimatorManager:
    """
    Orchestrates estimation workflow.

    Responsibilities:
    - Initialize and validate estimator connection
    - Load and validate input data
    - Coordinate transformation and fitting
    - Format and return results

    Does NOT:
    - Implement statistical algorithms (delegated to adapters)
    - Handle configuration parsing (delegated to config module)
    """

    def __init__(self, estimator: Estimator, config: Config):
        """
        Initialize manager with estimator and configuration.

        Args:
            estimator: Estimator instance to use for fitting
            config: Configuration object with estimation settings
        """
        self.estimator = estimator
        self.config = config
        self._is_connected = False

    def connect(self) -> "EstimatorManager":
        """
        Initialize estimator connection.

        Returns:
            Self for method chaining

        Raises:
            EstimationError: If connection validation fails
        """
        if self.config.estimation is None:
            raise EstimationError("No estimation configuration provided")

        self.estimator.connect(self.config.estimation)

        if not self.estimator.validate_connection():
            raise EstimationError(
                "Estimator failed connection validation. "
                "Check configuration settings."
            )

        self._is_connected = True
        return self

    def fit(self, data: Optional[pd.DataFrame] = None) -> EstimationResult:
        """
        Execute full estimation pipeline.

        Pipeline:
        1. Validate connection
        2. Load data if not provided
        3. Validate input data
        4. Transform to estimator format
        5. Fit model
        6. Transform results to standard format

        Args:
            data: Optional DataFrame. If not provided, loads from config.

        Returns:
            EstimationResult with MTE and related quantities

        Raises:
            EstimationError: If not connected or fitting fails
        """
        if not self._is_connected:
            raise EstimationError(
                "Must call connect() before fit(). "
                "Use: manager.connect().fit()"
            )

        # Load data if not provided
        if data is None:
            data = self._load_data()

        # Validate input data
        self.estimator.validate_data(data)

        # Transform to estimator format
        transformed = self.estimator.transform_outbound(data)

        # Execute estimation
        try:
            raw_results = self.estimator.fit(transformed)
        except Exception as e:
            raise EstimationError(f"Estimation failed: {e}") from e

        # Transform to standard format
        return self.estimator.transform_inbound(raw_results)

    def _load_data(self) -> pd.DataFrame:
        """
        Load data from path specified in configuration.

        Returns:
            DataFrame loaded from file

        Raises:
            EstimationError: If data loading fails
        """
        if self.config.estimation is None:
            raise EstimationError("No estimation configuration")

        file_path = self.config.estimation.file
        if not file_path:
            raise EstimationError(
                "No data file specified in configuration. "
                "Either provide data directly or set 'file' in ESTIMATION config."
            )

        try:
            if file_path.endswith(".pkl"):
                return pd.read_pickle(file_path)
            elif file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith(".dta"):
                return pd.read_stata(file_path)
            elif file_path.endswith(".txt"):
                return pd.read_csv(file_path, sep="\t")
            else:
                # Try pickle as default
                return pd.read_pickle(file_path)
        except Exception as e:
            raise EstimationError(f"Failed to load data from '{file_path}': {e}") from e

    @property
    def is_connected(self) -> bool:
        """Check if manager is connected and ready for fitting."""
        return self._is_connected
