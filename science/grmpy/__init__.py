"""
grmpy: Generalized Roy Model in Python

A package for simulation and estimation of the generalized Roy model,
an econometric framework for policy evaluation and causal inference.

Public API:
    process_config(path) -> Config
        Load and validate configuration from YAML file

    simulate(config) -> DataFrame
        Generate synthetic data according to the Roy model

    estimate(config, data) -> EstimationResult
        Estimate the Marginal Treatment Effect from data

    plot(result, ...) -> None
        Visualize the Marginal Treatment Effect curve

Example:
    >>> import grmpy
    >>> config = grmpy.process_config("config.yml")
    >>> data = grmpy.simulate(config)
    >>> result = grmpy.estimate(config, data)
    >>> grmpy.plot(result)

For more information, see: https://grmpy.readthedocs.io
"""

__version__ = "2.0.0"

from grmpy.config import process_config
from grmpy.engine import estimate, plot, simulate

__all__ = [
    "process_config",
    "simulate",
    "estimate",
    "plot",
    "__version__",
]
