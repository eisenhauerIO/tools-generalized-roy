"""
Unit tests for estimator and simulator factories.

Tests registry pattern, factory functions, and extension mechanism.
"""

import pytest

from grmpy.estimators.base import Estimator
from grmpy.estimators.factory import (
    ESTIMATOR_REGISTRY,
    get_estimator,
    list_estimators,
    register_estimator,
    unregister_estimator,
)
from grmpy.simulators.base import Simulator
from grmpy.simulators.factory import (
    SIMULATOR_REGISTRY,
    get_simulator,
    register_simulator,
)


class TestEstimatorRegistry:
    """Tests for estimator registration and retrieval."""

    def test_get_estimator_returns_parametric(self):
        """Factory returns ParametricEstimator for 'parametric' key."""
        # Ensure registration
        from grmpy.estimators.adapter_parametric import ParametricEstimator

        if "parametric" not in ESTIMATOR_REGISTRY:
            register_estimator("parametric", ParametricEstimator)

        estimator = get_estimator("parametric")
        assert isinstance(estimator, Estimator)

    def test_get_estimator_returns_semiparametric(self):
        """Factory returns SemiparametricEstimator for 'semiparametric' key."""
        from grmpy.estimators.adapter_semiparametric import SemiparametricEstimator

        if "semiparametric" not in ESTIMATOR_REGISTRY:
            register_estimator("semiparametric", SemiparametricEstimator)

        estimator = get_estimator("semiparametric")
        assert isinstance(estimator, Estimator)

    def test_get_estimator_raises_for_unknown_method(self):
        """Factory raises ValueError with available options for unknown method."""
        with pytest.raises(ValueError) as exc_info:
            get_estimator("unknown_method")

        assert "unknown_method" in str(exc_info.value)
        assert "Available methods" in str(exc_info.value)

    def test_register_estimator_adds_to_registry(self):
        """Custom estimator can be registered at runtime."""
        from grmpy.core.contracts import EstimationConfig, EstimationResult
        from grmpy.estimators.base import Estimator

        class CustomEstimator(Estimator):
            def connect(self, config):
                pass

            def validate_connection(self):
                return True

            def validate_data(self, data):
                pass

            def get_required_columns(self):
                return []

            def transform_outbound(self, data):
                return data

            def fit(self, data, **kwargs):
                return {}

            def transform_inbound(self, results):
                return EstimationResult(
                    mte=[],
                    mte_x=[],
                    mte_u=[],
                    quantiles=[],
                    coefficients={},
                )

        register_estimator("custom_test", CustomEstimator)

        assert "custom_test" in ESTIMATOR_REGISTRY

        # Cleanup
        unregister_estimator("custom_test")

    def test_register_estimator_rejects_non_estimator_class(self):
        """Registration fails for classes not implementing Estimator."""

        class NotAnEstimator:
            pass

        with pytest.raises(TypeError) as exc_info:
            register_estimator("invalid", NotAnEstimator)

        assert "Estimator base class" in str(exc_info.value)

    def test_unregister_estimator_removes_from_registry(self):
        """Unregister removes estimator from registry."""
        from grmpy.estimators.adapter_parametric import ParametricEstimator

        # Register temporarily
        register_estimator("temp_test", ParametricEstimator)
        assert "temp_test" in ESTIMATOR_REGISTRY

        # Unregister
        unregister_estimator("temp_test")
        assert "temp_test" not in ESTIMATOR_REGISTRY

    def test_unregister_raises_for_unknown(self):
        """Unregister raises ValueError for unknown estimator."""
        with pytest.raises(ValueError) as exc_info:
            unregister_estimator("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_list_estimators_returns_descriptions(self):
        """list_estimators returns dict with descriptions."""
        from grmpy.estimators.adapter_parametric import ParametricEstimator

        if "parametric" not in ESTIMATOR_REGISTRY:
            register_estimator("parametric", ParametricEstimator)

        estimators = list_estimators()

        assert isinstance(estimators, dict)
        assert "parametric" in estimators
        assert isinstance(estimators["parametric"], str)


class TestSimulatorRegistry:
    """Tests for simulator registration and retrieval."""

    def test_get_simulator_returns_roy_model(self):
        """Factory returns RoyModelSimulator for 'roy_model' key."""
        from grmpy.simulators.adapter_roy_model import RoyModelSimulator

        if "roy_model" not in SIMULATOR_REGISTRY:
            register_simulator("roy_model", RoyModelSimulator)

        simulator = get_simulator("roy_model")
        assert isinstance(simulator, Simulator)

    def test_get_simulator_raises_for_unknown(self):
        """Factory raises ValueError for unknown simulator."""
        with pytest.raises(ValueError) as exc_info:
            get_simulator("unknown_simulator")

        assert "unknown_simulator" in str(exc_info.value)

    def test_register_simulator_rejects_non_simulator_class(self):
        """Registration fails for classes not implementing Simulator."""

        class NotASimulator:
            pass

        with pytest.raises(TypeError) as exc_info:
            register_simulator("invalid", NotASimulator)

        assert "Simulator base class" in str(exc_info.value)
