"""
Estimators module for MTE estimation.

Provides parametric and semiparametric estimators for the
Marginal Treatment Effect in the generalized Roy model.

Design Decision: AVAILABLE_FUNCTIONS exported for dynamic validation,
enabling contracts.py to derive valid options rather than hard-coding.
"""

from typing import List

import pandas as pd

from grmpy.core.contracts import Config, EstimationResult
from grmpy.core.exceptions import GrmpyError

# Available estimation functions - used by contracts.py for validation
AVAILABLE_FUNCTIONS: List[str] = ["parametric", "semiparametric"]


def estimate(config: Config, data: pd.DataFrame) -> EstimationResult:
    """
    Estimate MTE using method specified in config.

    Dispatches to parametric or semiparametric estimator based on
    ESTIMATION.FUNCTION in configuration.

    Args:
        config: Configuration with ESTIMATION.FUNCTION set.
        data: DataFrame with outcome, treatment, and covariates.

    Returns:
        EstimationResult with MTE and coefficients.

    Raises:
        GrmpyError: If FUNCTION is not recognized.
    """
    if config.estimation is None:
        raise GrmpyError("No ESTIMATION configuration provided")

    function = config.estimation.function

    if function == "parametric":
        from grmpy.estimators.parametric import estimate as estimate_parametric

        return estimate_parametric(config, data)

    elif function == "semiparametric":
        from grmpy.estimators.semiparametric import estimate as estimate_semiparametric

        return estimate_semiparametric(config, data)

    else:
        raise GrmpyError(f"Unknown ESTIMATION.FUNCTION: '{function}'. Available: {AVAILABLE_FUNCTIONS}")


__all__ = ["estimate", "AVAILABLE_FUNCTIONS"]
