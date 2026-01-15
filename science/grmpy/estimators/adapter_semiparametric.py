"""
Semiparametric LIV estimator adapter.

Design Decision: This adapter wraps the existing semipar_fit() logic while
conforming to the Estimator interface for consistent orchestration. This
enables using the robust LIV algorithms while providing modern extensibility.

Assumption: No distributional assumptions on unobservables. Uses Local
Instrumental Variables (LIV) for flexible MTE estimation at the cost of
requiring larger sample sizes than parametric methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grmpy.core.contracts import EstimationConfig, EstimationResult
from grmpy.core.exceptions import DataValidationError, EstimationError
from grmpy.estimators.base import Estimator


class SemiparametricEstimator(Estimator):
    """
    Local Instrumental Variables (LIV) semiparametric estimator.

    Makes no distributional assumptions on unobservables and uses
    local polynomial regression for MTE estimation.

    Design Decision: Uses local polynomial regression for flexibility
    at the cost of requiring larger sample sizes. The minimum sample
    size is configurable via EstimationConfig.min_sample_size to allow
    users to adjust based on their specific use case.
    """

    def __init__(self):
        self.config: Optional[EstimationConfig] = None

    def connect(self, config: EstimationConfig) -> None:
        """
        Initialize estimator with configuration.

        Args:
            config: EstimationConfig containing method parameters including
                bandwidth, gridsize, and minimum sample size settings.
        """
        self.config = config

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
        """
        if self.config is None:
            return ["Y", "D"]
        return [self.config.dependent, self.config.treatment]

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data meets estimator requirements.

        Checks for required columns, validates treatment is binary, and
        ensures sufficient sample size for local polynomial regression.

        Design Decision: Minimum sample size is read from config rather
        than hardcoded, allowing users to override the default (100) when
        they have domain knowledge justifying a different threshold.

        Args:
            data: DataFrame to validate.

        Raises:
            DataValidationError: If required columns are missing, treatment
                is not binary, or sample size is below minimum.
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

        # Semiparametric requires larger samples for reliable local polynomial regression
        min_sample_size = self.config.min_sample_size if self.config else 100
        if len(data) < min_sample_size:
            raise DataValidationError(
                f"Semiparametric estimation requires at least {min_sample_size} "
                f"observations for reliable local polynomial regression. "
                f"Got: {len(data)}. Adjust min_sample_size in config if needed."
            )

    def transform_outbound(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform DataFrame to format for estimation.

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
        Execute semiparametric LIV estimation.

        Wraps the legacy semipar_fit() function which implements Local
        Instrumental Variables estimation using local polynomial regression.

        Args:
            data: Dictionary from transform_outbound() containing DataFrame
                and config.
            **kwargs: Additional arguments (for interface compatibility).

        Returns:
            Dictionary containing raw estimation results including:
            - mte: Marginal treatment effect values
            - mte_x, mte_u: Observable and unobservable MTE components
            - quantiles: Evaluation points
            - b0, b1: Estimated coefficients

        Raises:
            EstimationError: If legacy module import fails (kernreg package
                required) or estimation encounters issues.
        """
        df = data["data"]
        config = data["config"]

        # Import legacy estimation function
        # Design Decision: Keep core algorithms in legacy module to avoid
        # introducing bugs in complex statistical code during migration
        try:
            from grmpy.estimate.estimate_semipar import semipar_fit
        except ImportError as e:
            raise EstimationError(
                f"Failed to import legacy semiparametric module: {e}. "
                "Note: Semiparametric estimation requires the 'kernreg' package."
            )

        # Build legacy dict format from config
        legacy_dict = self._build_legacy_dict(config, df)

        # Call legacy estimation
        try:
            result = semipar_fit(legacy_dict, df)
        except ImportError as e:
            raise EstimationError(
                f"Semiparametric estimation failed: {e}. "
                "The 'kernreg' package may not be installed or compatible."
            )

        return result

    def transform_inbound(self, results: Dict[str, Any]) -> EstimationResult:
        """
        Convert raw results to standard format.

        Transforms legacy result dictionary to EstimationResult dataclass
        for consistent downstream processing.

        Args:
            results: Dictionary from fit() containing raw estimation output.

        Returns:
            EstimationResult with standardized fields for MTE, coefficients,
            and metadata including bandwidth and gridsize used.
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
                "method": "semiparametric",
                "bandwidth": self.config.bandwidth if self.config else None,
                "gridsize": self.config.gridsize if self.config else 500,
            },
        )

    def _build_legacy_dict(
        self, config: EstimationConfig, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build legacy dictionary format from new config.

        Design Decision: This bridge function enables gradual migration
        from the old dict-based configuration to typed dataclasses without
        rewriting the core LIV estimation algorithms.

        Args:
            config: Typed EstimationConfig object.
            data: DataFrame to extract column information from.

        Returns:
            Dictionary in legacy format expected by semipar_fit().
        """
        all_cols = list(data.columns)
        outcome_col = config.dependent
        treatment_col = config.treatment

        potential_covs = [c for c in all_cols if c not in [outcome_col, treatment_col]]

        covs_treated = config.covariates_treated or potential_covs
        covs_untreated = config.covariates_untreated or potential_covs
        covs_choice = config.covariates_choice or potential_covs

        ps_start, ps_end = config.ps_range

        legacy_dict = {
            "ESTIMATION": {
                "file": config.file,
                "dependent": outcome_col,
                "indicator": treatment_col,
                "bandwidth": config.bandwidth,
                "gridsize": config.gridsize,
                "ps_range": [ps_start, ps_end],
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
        }

        return legacy_dict
