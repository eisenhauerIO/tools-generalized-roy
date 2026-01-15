"""
Abstract base interface for all estimation methods.

Design Decision: Separates data transformation (outbound/inbound) from
fitting logic to enable different data formats without modifying core
estimation algorithms.

All estimator implementations must inherit from Estimator and implement
all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from grmpy.core.contracts import EstimationConfig, EstimationResult


class Estimator(ABC):
    """
    Abstract interface for all estimation methods.

    Lifecycle:
    1. connect() - Initialize with configuration
    2. validate_connection() - Verify setup is complete
    3. validate_data() - Check input data meets requirements
    4. transform_outbound() - Convert data to estimator format
    5. fit() - Execute estimation
    6. transform_inbound() - Convert results to standard format
    """

    @abstractmethod
    def connect(self, config: EstimationConfig) -> None:
        """
        Initialize estimator with configuration.

        Args:
            config: Estimation configuration object
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Verify estimator is properly configured.

        Returns:
            True if configuration is valid and complete
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data meets estimator requirements.

        Args:
            data: Input DataFrame to validate

        Raises:
            DataValidationError: With specific column/type issues
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Return list of required DataFrame columns.

        Returns:
            List of column names that must be present in input data
        """
        pass

    @abstractmethod
    def transform_outbound(self, data: pd.DataFrame) -> Any:
        """
        Transform standard format to estimator-specific format.

        Args:
            data: DataFrame in standard format

        Returns:
            Data in format required by fit() method
        """
        pass

    @abstractmethod
    def fit(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute estimation and return raw results.

        Args:
            data: Data in estimator-specific format
            **kwargs: Additional estimator-specific options

        Returns:
            Dictionary containing raw estimation results
        """
        pass

    @abstractmethod
    def transform_inbound(self, results: Dict[str, Any]) -> EstimationResult:
        """
        Transform estimator results to standard output format.

        Args:
            results: Raw results from fit()

        Returns:
            Standardized EstimationResult object
        """
        pass
