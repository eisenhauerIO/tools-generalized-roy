"""
Parametric normal model estimator.

Assumption: Joint normality of unobservables (U_1, U_0, V).
Estimates MTE via maximum likelihood.

Design Decision: All estimation logic is self-contained in this module,
eliminating dependencies on the legacy grmpy package.
"""

from random import randint
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.stats import norm, t
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from grmpy.core.contracts import Config, EstimationConfig, EstimationResult
from grmpy.core.exceptions import GrmpyError


def estimate(config: Config, data: pd.DataFrame) -> EstimationResult:
    """
    Estimate MTE using parametric normal model via maximum likelihood.

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

    # Run parametric estimation
    raw_result = _par_fit(est_config, data)

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


def _par_fit(est_config: EstimationConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run parametric estimation of the marginal treatment effect.

    Args:
        est_config: Estimation configuration.
        data: DataFrame to perform estimation on.

    Returns:
        Result dictionary containing MTE, coefficients, and optimization info.
    """
    # Build legacy dict format for internal processing
    dict_ = _build_legacy_dict(est_config, data)

    # Process data into arrays
    D, X1, X0, Z1, Z0, Y1, Y0 = _process_data(data, dict_)

    # Process optimization options
    opt_dict, method, grad_opt, start_option, seed_ = _process_inputs(dict_)
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    # Set seed for reproducibility
    np.random.seed(seed_)

    # Create result container
    rslt_cont = _create_rslt_df(dict_)

    # Determine start values
    x0 = _start_values(dict_, D, X1, X0, Z1, Z0, Y1, Y0, start_option)
    dict_["AUX"]["criteria"] = _calculate_criteria(x0, X1, X0, Z1, Z0, Y1, Y0)
    rslt_cont["start_values"] = _backward_transformation(x0)

    # Run optimization
    bfgs_dict: Dict[str, Dict] = {"parameter": {}, "crit": {}, "grad": {}}
    opt_rslt = minimize(
        _minimizing_interface,
        x0,
        args=(X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, bfgs_dict, grad_opt),
        method=method,
        options=opt_dict,
        jac=grad_opt,
    )

    # Adjust output
    rslt = _adjust_output(opt_rslt, dict_, rslt_cont, x0, method, start_option, X1, X0, Z1, Z0, Y1, Y0, bfgs_dict)

    # Calculate MTE
    quantiles, cov, X, b1_b0, b1, b0 = _prepare_mte_calc(rslt["opt_rslt"], data)
    mte_x = np.dot(X, b1_b0)
    mte_u = (cov[2, 0] - cov[2, 1]) * norm.ppf(quantiles)

    # Combine MTE components
    mte = mte_x.mean(axis=0) + mte_u
    mte_min = np.min(mte_x) + mte_u
    mte_max = np.max(mte_x) + mte_u

    rslt.update(
        {
            "quantiles": quantiles,
            "mte": mte,
            "mte_x": mte_x,
            "mte_u": mte_u,
            "mte_min": mte_min,
            "mte_max": mte_max,
            "X": X,
            "b1": b1,
            "b0": b0,
        }
    )

    return rslt


def _build_legacy_dict(config: EstimationConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build legacy dictionary format from config.

    This bridges the new typed configuration with the internal
    estimation algorithms that expect the legacy dict format.
    """
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
        "DIST": {"params": [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]},  # Default distribution
    }


def _process_data(
    data: pd.DataFrame, dict_: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process data for optimization, splitting by treatment status.

    Returns:
        Tuple of (D, X1, X0, Z1, Z0, Y1, Y0) arrays.
    """
    indicator = dict_["ESTIMATION"]["indicator"]
    outcome = dict_["ESTIMATION"]["dependent"]
    D = data[indicator].values

    data1 = data[data[indicator] == 1]
    data0 = data[data[indicator] == 0]

    X1 = data1[dict_["TREATED"]["order"]].values
    X0 = data0[dict_["UNTREATED"]["order"]].values
    Z1 = data1[dict_["CHOICE"]["order"]].values
    Z0 = data0[dict_["CHOICE"]["order"]].values

    Y1 = data1[outcome].values
    Y0 = data0[outcome].values

    return D, X1, X0, Z1, Z0, Y1, Y0


def _process_inputs(dict_: Dict[str, Any]) -> Tuple[Dict, str, bool, str, int]:
    """
    Process optimization specifications from configuration.

    Returns:
        Tuple of (opt_dict, method, grad_opt, start_option, seed).
    """
    method = dict_["ESTIMATION"].get("optimizer", "BFGS")
    grad_opt = method == "BFGS"
    opt_dict = dict_.get("SCIPY-" + method, {})
    start_option = dict_["ESTIMATION"].get("start", "auto")

    maxiter = dict_["ESTIMATION"].get("maxiter")
    if maxiter is not None:
        opt_dict["maxiter"] = maxiter
        if maxiter == 0:
            start_option = "init"

    seed_ = dict_.get("SIMULATION", {}).get("seed", randint(0, 9999))

    return opt_dict, method, grad_opt, start_option, seed_


def _create_rslt_df(dict_: Dict[str, Any]) -> pd.DataFrame:
    """Create DataFrame container for estimation results."""
    index = []
    for section in ["TREATED", "UNTREATED", "CHOICE"]:
        index += [(section, i) for i in dict_[section]["order"]]
    for subsection in ["sigma1", "rho1", "sigma0", "rho0"]:
        index += [("DIST", subsection)]

    column_names = ["params", "start_values", "std", "t_values", "p_values", "conf_int_low", "conf_int_up"]

    return pd.DataFrame(
        index=pd.MultiIndex.from_tuples(index, names=["section", "name"]),
        columns=column_names,
    )


def _start_values(
    dict_: Dict[str, Any],
    D: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    start_option: str,
) -> np.ndarray:
    """
    Select start values for the minimization process.

    If option is 'init', returns values from initialization file.
    Otherwise, uses OLS and Probit to estimate start values.
    """
    if start_option == "init":
        rho1 = dict_["DIST"]["params"][2] / dict_["DIST"]["params"][0]
        rho0 = dict_["DIST"]["params"][4] / dict_["DIST"]["params"][3]
        dist = [dict_["DIST"]["params"][0], rho1, dict_["DIST"]["params"][3], rho0]
        n_params = X1.shape[1] + X0.shape[1] + Z1.shape[1]
        init_values = dict_["AUX"].get("init_values", np.zeros(n_params))[:-6]
        x0 = np.concatenate((init_values, dist))
    elif start_option == "auto":
        try:
            if D.shape[0] == sum(D):
                raise PerfectSeparationError("All observations are treated")

            beta = []
            sd_ = []

            for data_out in [(Y1, X1), (Y0, X0)]:
                ols_results = sm.OLS(data_out[0], data_out[1]).fit()
                beta += [ols_results.params]
                sd = np.sqrt(ols_results.scale)
                rho = np.random.uniform(-sd, sd, 1) / sd
                sd_ += [sd, rho[0]]

            Z = np.vstack((Z0, Z1))
            probitRslt = sm.Probit(np.sort(D), Z).fit(disp=0)
            gamma = probitRslt.params
            x0 = np.concatenate((beta[0], beta[1], gamma, sd_))

            # Validate start values
            if not np.all(np.isfinite(x0)):
                raise GrmpyError("Automatic start values are not finite")

        except (PerfectSeparationError, ValueError, GrmpyError):
            # Fall back to init values
            rho1 = dict_["DIST"]["params"][2] / dict_["DIST"]["params"][0]
            rho0 = dict_["DIST"]["params"][4] / dict_["DIST"]["params"][3]
            dist = [dict_["DIST"]["params"][0], rho1, dict_["DIST"]["params"][3], rho0]
            init_vals = dict_["AUX"].get("init_values", np.zeros(X1.shape[1] + X0.shape[1] + Z1.shape[1] + 6))
            x0 = np.concatenate((init_vals[:-6], dist))
    else:
        raise GrmpyError(f"Unknown start option: {start_option}. Use 'init' or 'auto'.")

    x0 = _start_value_adjustment(x0)
    return np.array(x0)


def _start_value_adjustment(x: np.ndarray) -> np.ndarray:
    """
    Transform start values using Lokshin and Sajaia (2004) approach.

    Takes log of sigma values and inverse hyperbolic tangent of rho values.
    This ensures sigma > 0 and rho is bounded between -1 and 1.
    """
    x = x.copy()
    x[-4:] = [
        np.log(x[-4]),
        np.log((1 + x[-3]) / (1 - x[-3])) / 2,
        np.log(x[-2]),
        np.log((1 + x[-1]) / (1 - x[-1])) / 2,
    ]
    return x


def _backward_transformation(x_trans: np.ndarray, bfgs_dict: Optional[Dict] = None) -> np.ndarray:
    """Reverse the sigma/rho transformation."""
    x_rev = x_trans.copy()
    x_rev[-4:] = [
        np.exp(x_rev[-4]),
        (np.exp(2 * x_rev[-3]) - 1) / (np.exp(2 * x_rev[-3]) + 1),
        np.exp(x_rev[-2]),
        (np.exp(2 * x_rev[-1]) - 1) / (np.exp(2 * x_rev[-1]) + 1),
    ]
    if bfgs_dict is not None:
        bfgs_dict["parameter"][str(len(bfgs_dict["parameter"]))] = x_rev
    return x_rev


def _log_likelihood(
    x0: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    num_treated: int,
    num_untreated: int,
    bfgs_dict: Optional[Dict] = None,
    grad_opt: bool = True,
) -> Any:
    """
    Log-likelihood function for the parametric normal model.

    Returns negative log-likelihood (and gradient if grad_opt=True).
    """
    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt(1 - rho1v**2))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt(1 - rho0v**2))

    treated = (1 / sd1) * norm.pdf(nu1) * norm.cdf(lambda1)
    untreated = (1 / sd0) * norm.pdf(nu0) * (1 - norm.cdf(lambda0))

    likl = -np.mean(np.log(np.append(treated, untreated)))

    if bfgs_dict is not None:
        bfgs_dict["crit"][str(len(bfgs_dict["crit"]))] = likl

    if grad_opt:
        llh_grad = _gradient(X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v)
        return likl, llh_grad
    else:
        return likl


def _calculate_criteria(
    x0: np.ndarray, X1: np.ndarray, X0: np.ndarray, Z1: np.ndarray, Z0: np.ndarray, Y1: np.ndarray, Y0: np.ndarray
) -> float:
    """Compute criterion function value for given parameters."""
    x = _backward_transformation(x0)
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]
    return _log_likelihood(x, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, None, False)


def _minimizing_interface(
    x0: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    num_treated: int,
    num_untreated: int,
    bfgs_dict: Dict,
    grad_opt: bool,
) -> Any:
    """Objective function for scipy.optimize.minimize."""
    x0 = _backward_transformation(x0, bfgs_dict)
    return _log_likelihood(x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, bfgs_dict, grad_opt)


def _adjust_output(
    opt_rslt: Any,
    dict_: Dict[str, Any],
    rslt_cont: pd.DataFrame,
    start_values: np.ndarray,
    method: str,
    start_option: str,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    bfgs_dict: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Add optimization information to the estimation output."""
    rslt: Dict[str, Any] = {
        "opt_info": {
            "optimizer": method,
            "start": start_option,
            "indicator": dict_["ESTIMATION"]["indicator"],
            "dependent": dict_["ESTIMATION"]["dependent"],
            "observations": Y1.shape[0] + Y0.shape[0],
        }
    }

    if opt_rslt["nit"] == 0:
        x = _backward_transformation(opt_rslt["x"])
        rslt["opt_info"]["success"], rslt["opt_info"]["status"] = False, 2
        rslt["opt_info"]["message"] = "---"
        rslt["opt_info"]["nit"] = 0
        rslt["opt_info"]["crit"] = dict_["AUX"]["criteria"]
        rslt["opt_info"]["warning"] = ["---"]
    else:
        check, flag = _check_rslt_parameters(opt_rslt["x"], X1, X0, Z1, Z0, Y1, Y0, bfgs_dict)
        if check:
            x, crit, warning = _process_output(dict_, bfgs_dict, opt_rslt["x"], flag)
            rslt["opt_info"]["crit"] = crit
            rslt["opt_info"]["warning"] = [warning]
        else:
            x = _backward_transformation(opt_rslt["x"])
            rslt["opt_info"]["crit"] = opt_rslt["fun"]
            rslt["opt_info"]["warning"] = ["---"]

        rslt["opt_info"]["success"] = opt_rslt.get("success")
        rslt["opt_info"]["status"] = opt_rslt.get("status")
        rslt["opt_info"]["message"] = opt_rslt.get("message")
        rslt["opt_info"]["nit"] = opt_rslt.get("nit")

    rslt_cont["params"] = x

    # Calculate standard errors
    maxiter = dict_["ESTIMATION"].get("maxiter", 1)
    se, hess_inv, conf_low, conf_up, p_values, t_values, warning_se = _calculate_se(x, maxiter, X1, X0, Z1, Z0, Y1, Y0)
    rslt_cont["std"] = se
    rslt["hessian_inv"] = hess_inv
    rslt_cont["conf_int_low"] = conf_low
    rslt_cont["conf_int_up"] = conf_up
    rslt_cont["p_values"] = p_values
    rslt_cont["t_values"] = t_values
    rslt["opt_rslt"] = rslt_cont

    if warning_se is not None:
        rslt["opt_info"]["warning"] = rslt["opt_info"].get("warning", []) + warning_se

    return rslt


def _check_rslt_parameters(
    x0: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    bfgs_dict: Optional[Dict],
) -> Tuple[bool, Optional[str]]:
    """Check if optimization returned the best parameters."""
    if bfgs_dict is None:
        return False, None

    crit = _calculate_criteria(x0, X1, X0, Z1, Z0, Y1, Y0)
    x = min(bfgs_dict["crit"], key=bfgs_dict["crit"].get)

    if False in np.isfinite(x0).tolist():
        return True, "notfinite"
    elif bfgs_dict["crit"][str(x)] < crit:
        return True, "adjustment"
    else:
        return False, None


def _process_output(
    init_dict: Dict[str, Any], bfgs_dict: Dict, x0: np.ndarray, flag: str
) -> Tuple[np.ndarray, float, str]:
    """Process output when optimization needs adjustment."""
    x = min(bfgs_dict["crit"], key=bfgs_dict["crit"].get)

    if flag == "adjustment":
        if bfgs_dict["crit"][str(x)] < init_dict["AUX"]["criteria"]:
            x0_out = bfgs_dict["parameter"][str(x)].tolist()
            crit = bfgs_dict["crit"][str(x)]
            warning = "Optimization adjusted to minimum criterion value found during search."
        else:
            x0_out = x0
            crit = bfgs_dict["crit"][str(x)]
            warning = "NONE"
    elif flag == "notfinite":
        x0_out = init_dict["AUX"].get("starting_values", x0)
        crit = init_dict["AUX"]["criteria"]
        warning = "Optimization returned non-finite values. Using start values."
    else:
        x0_out = _backward_transformation(x0)
        crit = bfgs_dict["crit"][str(x)]
        warning = "NONE"

    return np.array(x0_out), crit, warning


def _calculate_se(
    x: np.ndarray,
    maxiter: int,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[list]]:
    """Calculate standard errors via Hessian approximation."""
    num_ind = Y1.shape[0] + Y0.shape[0]
    x0 = x.copy()
    warning = None

    if maxiter == 0:
        n_params = len(x0)
        se = np.full(n_params, np.nan)
        hess_inv = np.full((n_params, n_params), np.nan)
        conf_low = np.full(n_params, np.nan)
        conf_up = np.full(n_params, np.nan)
        p_values = np.full(n_params, np.nan)
        t_values = np.full(n_params, np.nan)
    else:
        norm_value = norm.ppf(0.975)
        hess = approx_fprime_cs(x0, _gradient_hessian, args=(X1, X0, Z1, Z0, Y1, Y0))
        try:
            hess_inv = np.linalg.inv(hess)
            se = np.sqrt(np.diag(hess_inv) / num_ind)
            aux = norm_value * se
            conf_low = x0 - aux
            conf_up = x0 + aux
            t_values = np.divide(x0, se)
            p_values = 2 * (1 - t.cdf(np.abs(t_values), df=num_ind - len(x0)))
        except LinAlgError:
            n_params = len(x0)
            se = np.full(n_params, np.nan)
            hess_inv = np.full((n_params, n_params), np.nan)
            conf_low = np.full(n_params, np.nan)
            conf_up = np.full(n_params, np.nan)
            t_values = np.full(n_params, np.nan)
            p_values = np.full(n_params, np.nan)

        if not np.all(np.isfinite(se)):
            warning = ["Standard errors could not be computed (singular Hessian)."]

    return se, hess_inv, conf_low, conf_up, p_values, t_values, warning


def _gradient(
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    nu1: np.ndarray,
    nu0: np.ndarray,
    lambda1: np.ndarray,
    lambda0: np.ndarray,
    gamma: np.ndarray,
    sd1: float,
    sd0: float,
    rho1v: float,
    rho0v: float,
) -> np.ndarray:
    """Compute gradient of the log-likelihood function."""
    n_obs = X1.shape[0] + X0.shape[0]

    grad_beta1 = np.sum(
        np.einsum(
            "ij, i ->ij",
            X1,
            -(norm.pdf(lambda1) / norm.cdf(lambda1)) * (rho1v / (np.sqrt(1 - rho1v**2) * sd1)) - nu1 / sd1,
        ),
        0,
    )

    grad_beta0 = np.sum(
        np.einsum(
            "ij, i ->ij",
            X0,
            norm.pdf(lambda0) / (1 - norm.cdf(lambda0)) * (rho0v / (np.sqrt(1 - rho0v**2) * sd0)) - nu0 / sd0,
        ),
        0,
    )

    grad_sd1 = np.sum(
        sd1
        * (
            1 / sd1
            - (norm.pdf(lambda1) / norm.cdf(lambda1)) * (rho1v * nu1 / (np.sqrt(1 - rho1v**2) * sd1))
            - nu1**2 / sd1
        ),
        keepdims=True,
    )

    grad_sd0 = np.sum(
        sd0
        * (
            1 / sd0
            + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0))) * (rho0v * nu0 / (np.sqrt(1 - rho0v**2) * sd0))
            - nu0**2 / sd0
        ),
        keepdims=True,
    )

    grad_rho1v = np.sum(
        -(norm.pdf(lambda1) / norm.cdf(lambda1)) * ((np.dot(gamma, Z1.T) * rho1v) - nu1) / (1 - rho1v**2) ** 0.5,
        keepdims=True,
    )

    grad_rho0v = np.sum(
        (norm.pdf(lambda0) / (1 - norm.cdf(lambda0))) * ((np.dot(gamma, Z0.T) * rho0v) - nu0) / (1 - rho0v**2) ** 0.5,
        keepdims=True,
    )

    grad_gamma = sum(
        np.einsum(
            "ij, i ->ij",
            Z1,
            (norm.pdf(lambda1) / norm.cdf(lambda1)) * 1 / np.sqrt(1 - rho1v**2),
        )
    ) - sum(
        np.einsum(
            "ij, i ->ij",
            Z0,
            (norm.pdf(lambda0) / (1 - norm.cdf(lambda0))) * (1 / np.sqrt(1 - rho0v**2)),
        )
    )

    grad = np.concatenate((grad_beta1, grad_beta0, -grad_gamma, grad_sd1, grad_rho1v, grad_sd0, grad_rho0v))

    return grad / n_obs


def _gradient_hessian(
    x0: np.ndarray, X1: np.ndarray, X0: np.ndarray, Z1: np.ndarray, Z0: np.ndarray, Y1: np.ndarray, Y0: np.ndarray
) -> np.ndarray:
    """Compute gradient for Hessian approximation."""
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt(1 - rho1v**2))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt(1 - rho0v**2))

    grad = _gradient(X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v)

    multiplier = np.concatenate(
        (
            np.ones(len(grad[:-4])),
            np.array([1 / sd1, 1 / (1 - rho1v**2), 1 / sd0, 1 / (1 - rho0v**2)]),
        )
    )

    return multiplier * grad


def _prepare_mte_calc(
    opt_rslt: pd.DataFrame, data: pd.DataFrame
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare MTE calculation from optimization results.

    Returns:
        Tuple of (quantiles, cov, X, b1_b0, b1, b0).
    """
    quantiles = [0.0001] + np.arange(0.01, 1.0, 0.01).tolist() + [0.9999]

    # Create covariance matrix from estimation results
    dist_params = opt_rslt.loc["DIST", "params"]
    cov = np.zeros((3, 3))
    np.fill_diagonal(cov, np.array([dist_params[0], dist_params[2], 1.0]) ** 2)
    cov[2, 0] = dist_params[0] * dist_params[1]
    cov[2, 1] = dist_params[2] * dist_params[3]

    x_treated = data[opt_rslt.loc["TREATED"].index.values].values
    x_untreated = data[opt_rslt.loc["UNTREATED"].index.values].values
    X = np.append(x_treated, -x_untreated, axis=1)

    beta1 = opt_rslt.loc["TREATED"].params.values
    beta0 = opt_rslt.loc["UNTREATED"].params.values
    b1_b0 = np.append(beta1, beta0)

    return quantiles, cov, X, b1_b0, beta1, beta0
