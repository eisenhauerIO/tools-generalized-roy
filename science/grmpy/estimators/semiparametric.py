"""
Semiparametric LIV estimator.

Assumption: No distributional assumptions on unobservables.
Uses Local Instrumental Variables (LIV) for flexible MTE estimation.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from grmpy.core.contracts import Config, EstimationConfig, EstimationResult
from grmpy.core.exceptions import DataValidationError, EstimationError


def fit(config: Config, data: pd.DataFrame = None) -> EstimationResult:
    """
    Estimate MTE using semiparametric LIV method.

    Args:
        config: Configuration object with estimation settings.
        data: Optional DataFrame. If not provided, loads from config.file.

    Returns:
        EstimationResult with MTE and coefficients.

    Raises:
        EstimationError: If estimation fails or kernreg not installed.
        DataValidationError: If data validation fails.
    """
    est_config = config.estimation
    if est_config is None:
        raise EstimationError("No estimation configuration provided")

    # Load data if not provided
    if data is None:
        data = _load_data(est_config.file)

    # Validate data
    _validate_data(data, est_config)

    # Import legacy estimation function
    try:
        from grmpy.estimate.estimate_semipar import semipar_fit
    except ImportError as e:
        raise EstimationError(
            f"Failed to import semiparametric module: {e}. "
            "Note: Requires the 'kernreg' package."
        )

    # Build legacy dict format and run estimation
    legacy_dict = _build_legacy_dict(est_config, data)

    try:
        raw_result = semipar_fit(legacy_dict, data)
    except ImportError as e:
        raise EstimationError(
            f"Semiparametric estimation failed: {e}. "
            "The 'kernreg' package may not be installed."
        )

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
            "function": "semiparametric",
            "bandwidth": est_config.bandwidth,
            "gridsize": est_config.gridsize,
        },
    )


def _load_data(file_path: str) -> pd.DataFrame:
    """Load data from file path."""
    if not file_path:
        raise EstimationError(
            "No data file specified. Either provide data directly or set 'file' in PARAMS."
        )

    try:
        if file_path.endswith(".pkl"):
            return pd.read_pickle(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".dta"):
            return pd.read_stata(file_path)
        else:
            return pd.read_pickle(file_path)
    except Exception as e:
        raise EstimationError(f"Failed to load data from '{file_path}': {e}") from e


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

    # Semiparametric requires larger samples
    if len(data) < config.min_sample_size:
        raise DataValidationError(
            f"Semiparametric estimation requires at least {config.min_sample_size} "
            f"observations. Got: {len(data)}. Adjust min_sample_size in PARAMS if needed."
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

    ps_start, ps_end = config.ps_range

    return {
        "ESTIMATION": {
            "file": config.file,
            "dependent": outcome_col,
            "indicator": treatment_col,
            "bandwidth": config.bandwidth,
            "gridsize": config.gridsize,
            "ps_range": [ps_start, ps_end],
        },
        "TREATED": {"order": covs_treated},
        "UNTREATED": {"order": covs_untreated},
        "CHOICE": {"order": covs_choice},
    }
