"""
Parametric normal model estimator adapter.

Design Decision: This adapter wraps the existing par_fit() logic while
conforming to the Estimator interface for consistent orchestration.
This enables using the battle-tested legacy algorithms while providing
a modern, extensible interface.

Assumption: Joint normality of unobservables (U_1, U_0, V).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grmpy.core.contracts import EstimationConfig, EstimationResult
from grmpy.core.exceptions import DataValidationError, EstimationError
from grmpy.estimators.base import Estimator


class ParametricEstimator(Estimator):
    """
    Parametric normal model estimator using maximum likelihood.

    Assumes joint normality of unobservables (U_1, U_0, V) and estimates
    the Marginal Treatment Effect (MTE) via maximum likelihood.

    Design Decision: Optimizer options are derived from EstimationConfig
    rather than hardcoded to allow user customization while maintaining
    sensible defaults defined in config.py DEFAULTS.
    """

    def __init__(self):
        self.config: Optional[EstimationConfig] = None
        self._optimizer_options: Dict[str, Any] = {}

    def connect(self, config: EstimationConfig) -> None:
        """
        Initialize estimator with configuration.

        Design Decision: Optimizer options are constructed here rather than
        at fit() time to enable early validation and consistent behavior.

        Args:
            config: EstimationConfig containing method parameters, file paths,
                and optimizer settings.
        """
        self.config = config
        self._optimizer_options = {
            "maxiter": config.max_iterations,
            "gtol": config.tolerance,
            "disp": False,
        }

    def validate_connection(self) -> bool:
        """
        Verify estimator is properly configured.

        Returns:
            True if config has been set via connect(), False otherwise.
        """
        return self.config is not None

    def get_required_columns(self) -> List[str]:
        """
        Return list of required DataFrame columns.

        Returns:
            List of column names that must be present in input data.
            Returns configured column names if connected, defaults otherwise.
        """
        if self.config is None:
            return ["Y", "D"]
        return [self.config.dependent, self.config.treatment]

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data meets estimator requirements.

        Checks for required columns and validates treatment indicator
        is binary (0/1) as required by the Roy model framework.

        Args:
            data: DataFrame to validate.

        Raises:
            DataValidationError: If required columns are missing (lists
                available columns) or treatment is not binary (shows
                found values).
        """
        required = self.get_required_columns()
        missing = set(required) - set(data.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(data.columns)}"
            )

        # Check treatment is binary
        if self.config:
            treatment_col = self.config.treatment
            unique_vals = data[treatment_col].unique()
            if not set(unique_vals).issubset({0, 1}):
                raise DataValidationError(
                    f"Treatment column '{treatment_col}' must be binary (0/1). "
                    f"Found values: {sorted(unique_vals)}"
                )

    def transform_outbound(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform DataFrame to format for estimation.

        Design Decision: Returns a dictionary wrapper rather than transforming
        data directly, allowing fit() to access both data and config for
        building the legacy format.

        Args:
            data: DataFrame in standard format with required columns.

        Returns:
            Dictionary containing 'data' and 'config' keys for fit() method.

        Raises:
            EstimationError: If estimator not connected.
        """
        if self.config is None:
            raise EstimationError("Estimator not connected")

        return {
            "data": data,
            "config": self.config,
        }

    def fit(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute maximum likelihood estimation.

        Wraps the legacy par_fit() function which implements the parametric
        normal model estimation via scipy.optimize.minimize.

        Args:
            data: Dictionary from transform_outbound() containing DataFrame
                and config.
            **kwargs: Additional arguments (currently unused, for interface
                compatibility).

        Returns:
            Dictionary containing raw estimation results including:
            - mte: Marginal treatment effect values
            - mte_x, mte_u: Observable and unobservable MTE components
            - quantiles: Evaluation points
            - b0, b1: Estimated coefficients
            - opt_rslt: Optimization result details

        Raises:
            EstimationError: If legacy module import fails or estimation
                encounters numerical issues.
        """
        df = data["data"]
        config = data["config"]

        # Import legacy estimation function
        # Design Decision: Keep core algorithms in legacy module during
        # migration to avoid introducing bugs in tested statistical code
        try:
            from grmpy.estimate.estimate_par import par_fit
        except ImportError as e:
            raise EstimationError(
                f"Failed to import legacy estimation module: {e}"
            )

        # Build legacy dict format from config
        legacy_dict = self._build_legacy_dict(config, df)

        # Call legacy estimation
        result = par_fit(legacy_dict, df)

        return result

    def transform_inbound(self, results: Dict[str, Any]) -> EstimationResult:
        """
        Convert raw optimization results to standard format.

        Transforms legacy result dictionary to EstimationResult dataclass
        for consistent downstream processing.

        Args:
            results: Dictionary from fit() containing raw estimation output.

        Returns:
            EstimationResult with standardized fields for MTE, coefficients,
            and metadata.
        """
        return EstimationResult(
            mte=np.array(results.get("mte", [])),
            mte_x=np.array(results.get("mte_x", [])),
            mte_u=np.array(results.get("mte_u", [])),
            quantiles=np.array(results.get("quantiles", [])),
            coefficients={
                "b0": np.array(results.get("b0", [])),
                "b1": np.array(results.get("b1", [])),
            },
            metadata={
                "method": "parametric",
                "optimizer": self.config.optimizer if self.config else "BFGS",
            },
        )

    def _build_legacy_dict(
        self, config: EstimationConfig, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build legacy dictionary format from new config.

        Design Decision: This bridge function enables gradual migration
        from the old dict-based configuration to typed dataclasses without
        rewriting the core estimation algorithms.

        Args:
            config: Typed EstimationConfig object.
            data: DataFrame to extract column information from.

        Returns:
            Dictionary in legacy format expected by par_fit().
        """
        all_cols = list(data.columns)
        outcome_col = config.dependent
        treatment_col = config.treatment

        # Remove outcome and treatment from potential covariates
        potential_covs = [c for c in all_cols if c not in [outcome_col, treatment_col]]

        # Use configured covariates or infer from data
        covs_treated = config.covariates_treated or potential_covs
        covs_untreated = config.covariates_untreated or potential_covs
        covs_choice = config.covariates_choice or potential_covs

        legacy_dict = {
            "ESTIMATION": {
                "file": config.file,
                "dependent": outcome_col,
                "indicator": treatment_col,
                "start": config.start_values or "auto",
                "maxiter": config.max_iterations,
                "optimizer": config.optimizer,
                "gtol": config.tolerance,
            },
            "TREATED": {
                "order": covs_treated,
            },
            "UNTREATED": {
                "order": covs_untreated,
            },
            "CHOICE": {
                "order": covs_choice,
            },
            "AUX": {
                "criteria": None,
            },
        }

        return legacy_dict
