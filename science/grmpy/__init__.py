"""
grmpy: Generalized Roy Model in Python

A package for simulation and estimation of the generalized Roy model,
an econometric framework for policy evaluation and causal inference.

Public API:
    fit(config_path) -> EstimationResult
        Estimate the Marginal Treatment Effect from data

    simulate(config_path) -> DataFrame
        Generate synthetic data according to the Roy model

    plot_mte(result, ...) -> None
        Visualize the Marginal Treatment Effect curve

Example:
    >>> import grmpy
    >>> data = grmpy.simulate("simulation.grmpy.yml")
    >>> result = grmpy.fit("estimation.grmpy.yml")
    >>> grmpy.plot_mte(result)

For more information, see: https://grmpy.readthedocs.io
"""

__version__ = "2.0.0"

# Public API
from grmpy.engine import fit, plot_mte, simulate

# Estimator registration for extensibility
from grmpy.estimators.adapter_parametric import ParametricEstimator
from grmpy.estimators.adapter_semiparametric import SemiparametricEstimator
from grmpy.estimators.factory import register_estimator

# Auto-register built-in estimators
register_estimator("parametric", ParametricEstimator)
register_estimator("semiparametric", SemiparametricEstimator)

__all__ = [
    "fit",
    "simulate",
    "plot_mte",
    "__version__",
    # For extension
    "register_estimator",
]
