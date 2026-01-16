"""
Estimators module for MTE estimation.

Provides parametric and semiparametric estimators for the
Marginal Treatment Effect in the generalized Roy model.
"""

from typing import Optional

import pandas as pd

from grmpy.core.contracts import Config, EstimationResult
from grmpy.core.exceptions import ConfigurationError


def fit(config: Config, data: Optional[pd.DataFrame] = None) -> EstimationResult:
    """
    Estimate MTE using method specified in config.

    Dispatches to parametric or semiparametric estimator based on
    ESTIMATION.FUNCTION in configuration.

    Args:
        config: Configuration with ESTIMATION.FUNCTION set.
        data: Optional DataFrame. If not provided, loads from config.

    Returns:
        EstimationResult with MTE and coefficients.

    Raises:
        ConfigurationError: If FUNCTION is not recognized.
    """
    if config.estimation is None:
        raise ConfigurationError("No ESTIMATION configuration provided")

    function = config.estimation.function

    if function == "parametric":
        from grmpy.estimators.parametric import fit as fit_parametric
        return fit_parametric(config, data)

    elif function == "semiparametric":
        from grmpy.estimators.semiparametric import fit as fit_semiparametric
        return fit_semiparametric(config, data)

    else:
        raise ConfigurationError(
            f"Unknown ESTIMATION.FUNCTION: '{function}'. "
            f"Available: 'parametric', 'semiparametric'"
        )


__all__ = ["fit"]
