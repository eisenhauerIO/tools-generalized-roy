"""
Estimators module containing base interface, factory, and adapters.
"""

from grmpy.estimators.base import Estimator
from grmpy.estimators.factory import (
    ESTIMATOR_REGISTRY,
    get_estimator,
    list_estimators,
    register_estimator,
)
from grmpy.estimators.manager import EstimatorManager

__all__ = [
    "Estimator",
    "EstimatorManager",
    "ESTIMATOR_REGISTRY",
    "get_estimator",
    "list_estimators",
    "register_estimator",
]
