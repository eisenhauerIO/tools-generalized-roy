"""
Generalized Roy Model simulator.

Provides a straightforward simulate() function for generating synthetic
data according to the generalized Roy model specification.

Design Decision: All simulation logic is self-contained in this module,
eliminating dependencies on the legacy grmpy package.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grmpy.core.contracts import Config, SimulationConfig, SimulationResult
from grmpy.core.exceptions import GrmpyError


def simulate(config: Config) -> SimulationResult:
    """
    Generate synthetic data according to the generalized Roy model.

    Executes simulation by:
    1. Validating configuration
    2. Setting random seed for reproducibility
    3. Simulating unobservables from multivariate normal
    4. Simulating covariates
    5. Computing outcomes and treatment decisions

    Args:
        config: Configuration object containing simulation parameters.

    Returns:
        SimulationResult with simulated data including:
        - Y: Observed outcome
        - Y_1, Y_0: Potential outcomes
        - D: Treatment indicator
        - U_1, U_0, V: Unobservables

    Raises:
        GrmpyError: If simulation configuration is missing or invalid.
    """
    if config.simulation is None:
        raise GrmpyError("No simulation configuration found in config file. " "Please add a SIMULATION section.")

    sim_config = config.simulation

    # Validate configuration
    _validate_config(sim_config)

    # Set seed for reproducibility
    if sim_config.seed is not None:
        np.random.seed(sim_config.seed)

    # Build internal dict format
    internal_dict = _build_internal_dict(sim_config)

    # Execute simulation steps
    U = _simulate_unobservables(internal_dict)
    X = _simulate_covariates(internal_dict)
    df = _simulate_outcomes(internal_dict, X, U)

    # Optionally write output
    if sim_config.output_file:
        _write_output(internal_dict, df)

    return SimulationResult(
        data=df,
        parameters={
            "agents": sim_config.agents,
            "seed": sim_config.seed,
            "coefficients_treated": sim_config.coefficients_treated,
            "coefficients_untreated": sim_config.coefficients_untreated,
            "coefficients_choice": sim_config.coefficients_choice,
            "covariance": sim_config.covariance,
        },
        metadata={"function": "roy_model"},
    )


def _validate_config(sim_config: SimulationConfig) -> None:
    """Validate simulation configuration."""
    if sim_config.agents <= 0:
        raise GrmpyError(f"Number of agents must be positive, got: {sim_config.agents}")

    if not sim_config.coefficients_treated:
        raise GrmpyError("coefficients_treated must be provided")

    if not sim_config.coefficients_untreated:
        raise GrmpyError("coefficients_untreated must be provided")

    if not sim_config.coefficients_choice:
        raise GrmpyError("coefficients_choice must be provided")

    if sim_config.covariance:
        # Validate covariance matrix
        cov = np.array(sim_config.covariance)
        if cov.shape != (3, 3):
            raise GrmpyError(f"Covariance matrix must be 3x3, got shape: {cov.shape}")
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(cov)
        if np.any(eigenvalues < -1e-10):
            raise GrmpyError("Covariance matrix must be positive semi-definite")


def _build_internal_dict(sim_config: SimulationConfig) -> Dict[str, Any]:
    """
    Build internal dictionary format from SimulationConfig.

    This structure is used by the simulation functions internally.
    """
    n_treated = len(sim_config.coefficients_treated)
    n_untreated = len(sim_config.coefficients_untreated)
    n_choice = len(sim_config.coefficients_choice)

    # Collect all unique labels for covariates
    all_labels = []
    treated_labels = [f"X{i}" for i in range(n_treated)]
    untreated_labels = [f"X{i}" for i in range(n_untreated)]
    choice_labels = [f"Z{i}" for i in range(n_choice)]

    for label in treated_labels + untreated_labels:
        if label not in all_labels:
            all_labels.append(label)
    for label in choice_labels:
        if label not in all_labels:
            all_labels.append(label)

    return {
        "SIMULATION": {
            "agents": sim_config.agents,
            "seed": sim_config.seed,
            "source": sim_config.source,
            "output_file": sim_config.output_file,
        },
        "ESTIMATION": {
            "dependent": "Y",
            "indicator": "D",
        },
        "TREATED": {
            "params": sim_config.coefficients_treated,
            "order": treated_labels,
        },
        "UNTREATED": {
            "params": sim_config.coefficients_untreated,
            "order": untreated_labels,
        },
        "CHOICE": {
            "params": sim_config.coefficients_choice,
            "order": choice_labels,
        },
        "DIST": {
            "params": _flatten_covariance(sim_config.covariance),
        },
        "DETERMINISTIC": False,
        "AUX": {
            "num_covars": len(all_labels),
            "labels": all_labels,
        },
    }


def _flatten_covariance(covariance: List[List[float]]) -> List[float]:
    """
    Flatten 3x3 covariance matrix to the legacy 6-element format.

    Legacy format: [sigma1, rho1v*sigma1, rho1v, sigma0, rho0v*sigma0, sigma_v]
    where:
    - sigma1 = sqrt(Var(U1))
    - sigma0 = sqrt(Var(U0))
    - sigma_v = sqrt(Var(V)) = 1 (normalized)
    - rho1v = Cov(U1,V) / (sigma1 * sigma_v)
    - rho0v = Cov(U0,V) / (sigma0 * sigma_v)
    """
    if not covariance:
        # Default: independent standard normal
        return [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    cov = np.array(covariance)
    sigma1 = np.sqrt(cov[0, 0])
    sigma0 = np.sqrt(cov[1, 1])
    sigma_v = np.sqrt(cov[2, 2])

    # Covariances
    cov_1v = cov[0, 2]  # Cov(U1, V)
    cov_0v = cov[1, 2]  # Cov(U0, V)

    return [sigma1, cov_1v, cov_1v / (sigma1 * sigma_v) if sigma1 > 0 else 0.0, sigma0, cov_0v, sigma_v]


def _construct_covariance_matrix(internal_dict: Dict[str, Any]) -> np.ndarray:
    """Construct 3x3 covariance matrix from parameters."""
    params = internal_dict["DIST"]["params"]

    if len(params) == 6:
        # Legacy format: [sigma1, cov_1v, rho1v, sigma0, cov_0v, sigma_v]
        sigma1, cov_1v, _, sigma0, cov_0v, sigma_v = params
        cov = np.zeros((3, 3))
        cov[0, 0] = sigma1**2
        cov[1, 1] = sigma0**2
        cov[2, 2] = sigma_v**2
        cov[0, 2] = cov[2, 0] = cov_1v
        cov[1, 2] = cov[2, 1] = cov_0v
    else:
        # Assume it's already a flattened 3x3 matrix
        cov = np.array(params).reshape(3, 3)

    return cov


def _simulate_unobservables(internal_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulate unobservable error terms (U1, U0, V) from multivariate normal.

    Args:
        internal_dict: Internal configuration dictionary.

    Returns:
        DataFrame with columns U1, U0, V.
    """
    num_agents = internal_dict["SIMULATION"]["agents"]
    cov = _construct_covariance_matrix(internal_dict)

    U = np.random.multivariate_normal(np.zeros(3), cov, num_agents)

    return pd.DataFrame(U, columns=["U1", "U0", "V"])


def _simulate_covariates(internal_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulate covariates from standard normal distribution.

    First covariate is set to 1 (intercept).

    Args:
        internal_dict: Internal configuration dictionary.

    Returns:
        DataFrame with covariate columns.
    """
    num_agents = internal_dict["SIMULATION"]["agents"]
    num_covars = internal_dict["AUX"]["num_covars"]
    labels = internal_dict["AUX"]["labels"]

    # Simulate from standard normal
    means = np.zeros(num_covars)
    covs = np.eye(num_covars)
    X = np.random.multivariate_normal(means, covs, num_agents)

    df = pd.DataFrame(X, columns=labels)

    # Set first covariate (intercept) to 1
    if labels:
        df[labels[0]] = 1.0

    return df


def _simulate_outcomes(internal_dict: Dict[str, Any], X: pd.DataFrame, U: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate potential outcomes, treatment decision, and observed outcome.

    Args:
        internal_dict: Internal configuration dictionary.
        X: DataFrame of covariates.
        U: DataFrame of unobservables (U1, U0, V).

    Returns:
        DataFrame with all simulated variables.
    """
    dep = internal_dict["ESTIMATION"]["dependent"]
    indicator = internal_dict["ESTIMATION"]["indicator"]

    # Join covariates and unobservables
    df = X.join(U)

    # Get covariate matrices for each equation
    Z = df[internal_dict["CHOICE"]["order"]].values
    X_treated = df[internal_dict["TREATED"]["order"]].values
    X_untreated = df[internal_dict["UNTREATED"]["order"]].values

    # Get coefficients
    coeffs_treated = np.array(internal_dict["TREATED"]["params"])
    coeffs_untreated = np.array(internal_dict["UNTREATED"]["params"])
    coeffs_choice = np.array(internal_dict["CHOICE"]["params"])

    # Calculate potential outcomes
    # Y_1 = X * beta_1 + U_1
    df[dep + "1"] = np.dot(X_treated, coeffs_treated) + df["U1"]
    # Y_0 = X * beta_0 + U_0
    df[dep + "0"] = np.dot(X_untreated, coeffs_untreated) + df["U0"]

    # Treatment decision: D = 1 if Z * gamma > V
    C = np.dot(Z, coeffs_choice) - df["V"]
    df[indicator] = (C > 0).astype(float)

    # Observed outcome: Y = D * Y_1 + (1-D) * Y_0
    df[dep] = df[indicator] * df[dep + "1"] + (1 - df[indicator]) * df[dep + "0"]

    return df


def _write_output(internal_dict: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Write simulated data to pickle and text files.

    Args:
        internal_dict: Internal configuration dictionary.
        df: DataFrame with simulated data.
    """
    source = internal_dict["SIMULATION"]["source"]

    # Write pickle file
    df.to_pickle(f"{source}.grmpy.pkl")

    # Write text file
    with open(f"{source}.grmpy.txt", "w") as f:
        df.to_string(f, index=False, na_rep=".", col_space=15, justify="left")
