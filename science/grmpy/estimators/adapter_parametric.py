"""
Parametric normal model estimator adapter.

Design Decision: This adapter wraps the existing par_fit() logic while
conforming to the Estimator interface for consistent orchestration.

Assumption: Joint normality of unobservables (U_1, U_0, V).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from grmpy.core.contracts import EstimationConfig, EstimationResult
from grmpy.core.exceptions import DataValidationError, EstimationError
from grmpy.estimators.base import Estimator


class ParametricEstimator(Estimator):
    """
    Parametric normal model estimator using maximum likelihood.

    Assumes joint normality of unobservables (U_1, U_0, V) and estimates
    the Marginal Treatment Effect (MTE) via maximum likelihood.
    """

    def __init__(self):
        self.config: Optional[EstimationConfig] = None
        self._optimizer_options: Dict[str, Any] = {}

    def connect(self, config: EstimationConfig) -> None:
        """Initialize estimator with configuration."""
        self.config = config
        self._optimizer_options = {
            "maxiter": config.max_iterations,
            "gtol": config.tolerance,
            "disp": False,
        }

    def validate_connection(self) -> bool:
        """Verify estimator is properly configured."""
        return self.config is not None

    def get_required_columns(self) -> List[str]:
        """Return list of required DataFrame columns."""
        if self.config is None:
            return ["Y", "D"]
        # Include configured column names
        cols = [self.config.dependent, self.config.treatment]
        return cols

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data meets estimator requirements."""
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
        """Transform DataFrame to format for estimation."""
        if self.config is None:
            raise EstimationError("Estimator not connected")

        # Build the dictionary structure expected by legacy code
        return {
            "data": data,
            "config": self.config,
        }

    def fit(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute maximum likelihood estimation.

        This method wraps the legacy par_fit functionality.
        """
        df = data["data"]
        config = data["config"]

        # Import legacy estimation function
        # We keep the core logic in the legacy module for now
        try:
            from grmpy.estimate.estimate_par import par_fit
            from grmpy.read.read import read
        except ImportError as e:
            raise EstimationError(
                f"Failed to import legacy estimation module: {e}"
            )

        # Build legacy dict format from config
        # This bridges new config to old format
        legacy_dict = self._build_legacy_dict(config, df)

        # Call legacy estimation
        result = par_fit(legacy_dict, df)

        return result

    def transform_inbound(self, results: Dict[str, Any]) -> EstimationResult:
        """Convert raw optimization results to standard format."""
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

        This is a bridge to support the existing estimation code
        during migration.
        """
        # Determine covariates from config or infer from data
        all_cols = list(data.columns)
        outcome_col = config.dependent
        treatment_col = config.treatment

        # Remove outcome and treatment from potential covariates
        potential_covs = [c for c in all_cols if c not in [outcome_col, treatment_col]]

        # Use configured covariates or infer
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
