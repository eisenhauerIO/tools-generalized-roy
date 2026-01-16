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

from grmpy.engine import fit, plot_mte, simulate

__all__ = [
    "fit",
    "simulate",
    "plot_mte",
    "__version__",
]
