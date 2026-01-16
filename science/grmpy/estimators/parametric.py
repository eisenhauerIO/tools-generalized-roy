"""
Parametric normal model estimator.

Assumption: Joint normality of unobservables (U_1, U_0, V).
Estimates MTE via maximum likelihood.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from grmpy.core.contracts import Config, EstimationConfig, EstimationResult
from grmpy.core.exceptions import DataValidationError, EstimationError


def estimate(config: Config, data: pd.DataFrame) -> EstimationResult:
    """
    Estimate MTE using parametric normal model via maximum likelihood.

    Args:
        config: Configuration object with estimation settings.
        data: DataFrame with outcome, treatment, and covariates.

    Returns:
        EstimationResult with MTE and coefficients.

    Raises:
        EstimationError: If estimation fails.
        DataValidationError: If data validation fails.
    """
    est_config = config.estimation
    if est_config is None:
        raise EstimationError("No estimation configuration provided")

    # Validate data
    _validate_data(data, est_config)

    # Import legacy estimation function
    try:
        from grmpy.estimate.estimate_par import par_fit
    except ImportError as e:
        raise EstimationError(f"Failed to import legacy estimation module: {e}")

    # Build legacy dict format and run estimation
    legacy_dict = _build_legacy_dict(est_config, data)
    raw_result = par_fit(legacy_dict, data)

    # Convert to standard result format
    return EstimationResult(
        mte=np.array(raw_result.get("mte", [])),
        mte_x=np.array(raw_result.get("mte_x", [])),
        mte_u=np.array(raw_result.get("mte_u", [])),
        quantiles=np.array(raw_result.get("quantiles", [])),
        coefficients={
            "b0": np.array(raw_result.get("b0", [])),
            "b1": np.array(raw_result.get("b1", [])),
        },
        metadata={
            "function": "parametric",
            "optimizer": est_config.optimizer,
        },
    )


def _validate_data(data: pd.DataFrame, config: EstimationConfig) -> None:
    """Validate input data meets requirements."""
    required = [config.dependent, config.treatment]
    missing = set(required) - set(data.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(data.columns)}"
        )

    # Check treatment is binary
    unique_vals = data[config.treatment].unique()
    if not set(unique_vals).issubset({0, 1}):
        raise DataValidationError(
            f"Treatment column '{config.treatment}' must be binary (0/1). "
            f"Found values: {sorted(unique_vals)}"
        )


def _build_legacy_dict(config: EstimationConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """Build legacy dictionary format from config."""
    all_cols = list(data.columns)
    outcome_col = config.dependent
    treatment_col = config.treatment

    potential_covs = [c for c in all_cols if c not in [outcome_col, treatment_col]]

    covs_treated = config.covariates_treated or potential_covs
    covs_untreated = config.covariates_untreated or potential_covs
    covs_choice = config.covariates_choice or potential_covs

    return {
        "ESTIMATION": {
            "file": config.file,
            "dependent": outcome_col,
            "indicator": treatment_col,
            "start": config.start_values or "auto",
            "maxiter": config.max_iterations,
            "optimizer": config.optimizer,
            "gtol": config.tolerance,
        },
        "TREATED": {"order": covs_treated},
        "UNTREATED": {"order": covs_untreated},
        "CHOICE": {"order": covs_choice},
        "AUX": {"criteria": None},
    }
