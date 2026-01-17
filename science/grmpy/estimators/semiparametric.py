"""
Semiparametric estimator using Local Instrumental Variables (LIV).

Assumption: No distributional assumptions on unobservables.
Estimates MTE via local polynomial regression.

Design Decision: All estimation logic is self-contained in this module,
eliminating dependencies on the legacy grmpy package.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from skmisc.loess import loess

from grmpy.core.contracts import Config, EstimationConfig, EstimationResult
from grmpy.core.exceptions import GrmpyError
from grmpy.utils.kernreg import locpoly


def estimate(config: Config, data: pd.DataFrame) -> EstimationResult:
    """
    Estimate MTE using semiparametric Local Instrumental Variables method.

    Args:
        config: Configuration object with estimation settings.
        data: DataFrame with outcome, treatment, and covariates.

    Returns:
        EstimationResult with MTE and coefficients.

    Raises:
        GrmpyError: If estimation fails or data validation fails.
    """
    est_config = config.estimation
    if est_config is None:
        raise GrmpyError("No estimation configuration provided")

    # Validate data
    _validate_data(data, est_config)

    # Check minimum sample size
    if len(data) < est_config.min_sample_size:
        raise GrmpyError(
            f"Sample size ({len(data)}) is below minimum required "
            f"({est_config.min_sample_size}) for semiparametric estimation."
        )

    # Run semiparametric estimation
    raw_result = _semipar_fit(est_config, data)

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


def _validate_data(data: pd.DataFrame, config: EstimationConfig) -> None:
    """Validate input data meets requirements."""
    required = [config.dependent, config.treatment]
    missing = set(required) - set(data.columns)
    if missing:
        raise GrmpyError(f"Missing required columns: {sorted(missing)}. Available columns: {sorted(data.columns)}")

    # Check treatment is binary
    unique_vals = data[config.treatment].unique()
    if not set(unique_vals).issubset({0, 1}):
        raise GrmpyError(
            f"Treatment column '{config.treatment}' must be binary (0/1). Found values: {sorted(unique_vals)}"
        )


# -----------------------------------------------------------------------------
# Core Estimation Functions
# -----------------------------------------------------------------------------


def _semipar_fit(est_config: EstimationConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run semiparametric estimation via Local Instrumental Variables.

    Args:
        est_config: Estimation configuration.
        data: DataFrame to perform estimation on.

    Returns:
        Result dictionary containing MTE, coefficients, and metadata.
    """
    # Build legacy dict format for internal processing
    dict_ = _build_legacy_dict(est_config, data)

    # Process estimation parameters
    bins, logit, bandwidth, gridsize, startgrid, endgrid = _process_primary_inputs(dict_)
    trim, rbandwidth, reestimate_p = _process_secondary_inputs(dict_)

    # Estimate propensity scores
    data = _estimate_treatment_propensity(dict_, data, logit)

    # Trim to common support
    X, Y, prop_score = _trim_support(dict_, data, logit, bins, trim, reestimate_p)

    # Double residual regression
    b0, b1_b0 = _double_residual_reg(X, Y, prop_score, rbandwidth)

    # Construct MTE
    quantiles = np.linspace(startgrid, endgrid, gridsize)
    mte_x = _mte_observed(X, b1_b0)
    mte_u = _mte_unobserved(X, Y, b0, b1_b0, prop_score, bandwidth, gridsize, startgrid, endgrid)

    # Combine MTE components
    mte = mte_x.mean(axis=0) + mte_u
    mte_min = np.min(mte_x) + mte_u
    mte_max = np.max(mte_x) + mte_u

    return {
        "quantiles": quantiles,
        "mte": mte,
        "mte_x": mte_x,
        "mte_u": mte_u,
        "mte_min": mte_min,
        "mte_max": mte_max,
        "X": X,
        "b0": b0,
        "b1": b1_b0 + b0,
    }


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
            "bandwidth": config.bandwidth,
            "gridsize": config.gridsize,
            "ps_range": list(config.ps_range),
            "bins": 25,
            "logit": True,
            "trim_support": True,
            "reestimate_p": False,
            "rbandwidth": 0.05,
        },
        "TREATED": {"order": covs_treated},
        "UNTREATED": {"order": covs_untreated},
        "CHOICE": {"order": covs_choice},
    }


def _process_primary_inputs(dict_: Dict[str, Any]) -> Tuple[int, bool, float, int, float, float]:
    """
    Process primary estimation parameters.

    Returns:
        Tuple of (bins, logit, bandwidth, gridsize, startgrid, endgrid).
    """
    bins = dict_["ESTIMATION"].get("bins", 25)
    logit = dict_["ESTIMATION"].get("logit", True)
    bandwidth = dict_["ESTIMATION"].get("bandwidth") or 0.32
    gridsize = dict_["ESTIMATION"].get("gridsize", 500)
    ps_range = dict_["ESTIMATION"].get("ps_range", [0.005, 0.995])

    return bins, logit, bandwidth, gridsize, ps_range[0], ps_range[1]


def _process_secondary_inputs(dict_: Dict[str, Any]) -> Tuple[bool, float, bool]:
    """
    Process secondary estimation parameters.

    Returns:
        Tuple of (trim, rbandwidth, reestimate_p).
    """
    trim = dict_["ESTIMATION"].get("trim_support", True)
    rbandwidth = dict_["ESTIMATION"].get("rbandwidth", 0.05)
    reestimate_p = dict_["ESTIMATION"].get("reestimate_p", False)

    return trim, rbandwidth, reestimate_p


def _estimate_treatment_propensity(dict_: Dict[str, Any], data: pd.DataFrame, logit: bool) -> pd.DataFrame:
    """
    Estimate propensity scores via Logit or Probit.

    Args:
        dict_: Configuration dictionary.
        data: Input DataFrame.
        logit: If True use logit, else probit.

    Returns:
        DataFrame with added 'prop_score' column.
    """
    D = data[dict_["ESTIMATION"]["indicator"]].values
    Z = data[dict_["CHOICE"]["order"]]

    if logit:
        model = sm.Logit(D, Z).fit(disp=0)
    else:
        model = sm.Probit(D, Z).fit(disp=0)

    data = data.copy()
    data["prop_score"] = model.predict(Z)

    return data


def _trim_support(
    dict_: Dict[str, Any],
    data: pd.DataFrame,
    logit: bool,
    bins: int = 25,
    trim: bool = True,
    reestimate_p: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Define common support and trim data.

    Returns:
        Tuple of (X, Y, prop_score) DataFrames/Series.
    """
    prop_score = data["prop_score"]
    common_support = _define_common_support(dict_, data, bins)

    if trim:
        data = data[(data.prop_score >= common_support[0]) & (data.prop_score <= common_support[1])]
        prop_score = prop_score[(prop_score >= common_support[0]) & (prop_score <= common_support[1])]

        if reestimate_p:
            data = _estimate_treatment_propensity(dict_, data, logit)

    data = data.sort_values(by="prop_score", ascending=True)
    prop_score = prop_score.sort_values(ascending=True)
    X = data[dict_["TREATED"]["order"]]
    Y = data[[dict_["ESTIMATION"]["dependent"]]]

    return X, Y, prop_score


def _define_common_support(dict_: Dict[str, Any], data: pd.DataFrame, bins: int = 25) -> List[float]:
    """
    Define common support as the overlapping region of propensity score histograms.

    Returns:
        List of [lower_limit, upper_limit].
    """
    indicator = dict_["ESTIMATION"]["indicator"]

    treated = data[data[indicator] == 1]["prop_score"].values
    untreated = data[data[indicator] == 0]["prop_score"].values

    # Create histograms
    hist_treated, bin_edges = np.histogram(treated, bins=bins, range=(0, 1))
    hist_untreated, _ = np.histogram(untreated, bins=bins, range=(0, 1))

    # Find lower limit (from treated sample)
    lower_limit = np.min(treated)
    for i in range(bins):
        if bin_edges[i] > 0.5:
            break
        if bin_edges[i] < np.min(treated):
            continue
        if hist_treated[i] == 0:
            lower_limit = bin_edges[i + 1]

    # Find upper limit (from untreated sample)
    upper_limit = np.max(untreated)
    for i in reversed(range(bins)):
        if bin_edges[i] < 0.5:
            break
        if bin_edges[i] > np.max(untreated):
            continue
        if hist_untreated[i] == 0:
            upper_limit = bin_edges[i]

    return [lower_limit, upper_limit]


def _double_residual_reg(
    X: pd.DataFrame, Y: pd.DataFrame, prop_score: pd.Series, rbandwidth: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Double Residual Regression of X, Xp, and Y on propensity score.

    Returns:
        Tuple of (b0, b1_b0) coefficient arrays.
    """
    # Construct Xp := X * P(z)
    Xp = _construct_Xp(X, prop_score)

    # Generate residuals from loess regressions
    res_X = _generate_residuals(prop_score, X, rbandwidth)
    res_Xp = _generate_residuals(prop_score, Xp, rbandwidth)
    res_Y = _generate_residuals(prop_score, Y, rbandwidth)

    # Combine residuals
    col_names = list(X) + list(Xp)
    res_X_Xp = pd.DataFrame(np.append(res_X, res_Xp, axis=1), columns=col_names)

    # OLS regression: e_Y = e_X * b0 + e_Xp * (b1 - b0)
    model = sm.OLS(res_Y, res_X_Xp)
    results = model.fit()
    b0 = results.params[: len(list(X))]
    b1_b0 = results.params[len(list(X)) :]

    return np.array(b0), np.array(b1_b0)


def _construct_Xp(X: pd.DataFrame, prop_score: pd.Series) -> pd.DataFrame:
    """Construct X * propensity score interaction terms."""
    P_z = pd.concat([prop_score] * len(X.columns), axis=1, ignore_index=True)
    Xp = pd.DataFrame(X.values * P_z.values, columns=[key_ + "_ps" for key_ in list(X)], index=X.index)
    return Xp


def _generate_residuals(exog: pd.Series, endog: pd.DataFrame, bandwidth: float = 0.05) -> np.ndarray:
    """
    Generate residuals from loess regressions.

    Args:
        exog: Explanatory variable (propensity score).
        endog: Response variables.
        bandwidth: Span for loess regression.

    Returns:
        Array of residuals.

    Raises:
        GrmpyError: If sample size is insufficient for loess regression.
    """
    exog = np.array(exog)
    endog = np.array(endog)
    n = endog.shape[0]

    # loess requires minimum sample size; empirically ~20 per degree for stability
    min_loess_n = max(30, int(1.0 / bandwidth) + 5)
    if n < min_loess_n:
        raise GrmpyError(
            f"Semiparametric estimation requires at least {min_loess_n} observations "
            f"for loess regression with bandwidth={bandwidth}. Got: {n}. "
            "Consider using 'parametric' estimation for small samples."
        )

    try:
        if endog.ndim == 1:
            y_fit = loess(exog, endog, span=bandwidth, degree=1)
            y_fit.fit()
            return y_fit.outputs.fitted_residuals
        else:
            columns = endog.shape[1]
            res = np.zeros([n, columns])
            for col in range(columns):
                y_fit = loess(exog, endog[:, col], span=bandwidth, degree=1)
                y_fit.fit()
                res[:, col] = y_fit.outputs.fitted_residuals
            return res
    except Exception as e:
        raise GrmpyError(
            f"Loess regression failed: {e}. This may occur with very small samples or extreme bandwidth values."
        )


def _mte_observed(X: pd.DataFrame, b1_b0: np.ndarray) -> np.ndarray:
    """Compute observed component of MTE (depends on X)."""
    return np.dot(X, b1_b0)


def _mte_unobserved(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    b0: np.ndarray,
    b1_b0: np.ndarray,
    prop_score: pd.Series,
    bandwidth: float,
    gridsize: int,
    startgrid: float,
    endgrid: float,
) -> np.ndarray:
    """
    Compute unobserved component of MTE (depends on u_D).

    Uses local polynomial regression to estimate the derivative
    of E[Y|X,P] with respect to P.
    """
    # Construct Xp := X * P(z)
    Xp = _construct_Xp(X, prop_score)

    # Convert to arrays
    X_arr = np.array(X)
    Xp_arr = np.array(Xp)
    Y_arr = np.array(Y).ravel()
    prop_score_arr = np.array(prop_score)

    # Compute unobserved part of Y
    Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

    # Estimate mte_u via local polynomial regression with derivative
    rslt_locpoly = locpoly(
        x=prop_score_arr,
        y=Y_tilde,
        derivative=1,
        degree=2,
        bandwidth=bandwidth,
        gridsize=gridsize,
        a=startgrid,
        b=endgrid,
    )
    mte_u = rslt_locpoly["curvest"]

    return mte_u
