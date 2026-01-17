"""
Integration tests for complete estimation workflow.

Tests end-to-end pipeline from configuration to results.
"""

import pandas as pd
import pytest

from grmpy.core.contracts import EstimationResult
from grmpy.core.exceptions import ConfigurationError


@pytest.mark.integration
class TestEstimationPipeline:
    """End-to-end tests for complete estimation workflow."""

    def test_estimator_manager_workflow(self, mock_estimator, sample_estimation_data):
        """Test manager workflow with mock estimator."""
        from grmpy.core.contracts import Config, EstimationConfig
        from grmpy.estimators.manager import EstimatorManager

        # Create config
        config = Config(
            estimation=EstimationConfig(
                method="parametric",
                file="",  # Not used with direct data
            )
        )

        # Create manager with mock
        manager = EstimatorManager(mock_estimator, config)

        # Connect
        manager.connect()
        assert manager.is_connected

        # Fit with provided data
        result = manager.fit(sample_estimation_data)

        # Verify mock was called correctly
        mock_estimator.connect.assert_called_once()
        mock_estimator.validate_data.assert_called_once()
        mock_estimator.fit.assert_called_once()

    def test_manager_raises_without_connect(self, mock_estimator):
        """Manager raises error if fit called before connect."""
        from grmpy.core.contracts import Config, EstimationConfig
        from grmpy.core.exceptions import EstimationError
        from grmpy.estimators.manager import EstimatorManager

        config = Config(
            estimation=EstimationConfig(method="parametric", file="")
        )
        manager = EstimatorManager(mock_estimator, config)

        with pytest.raises(EstimationError) as exc_info:
            manager.fit(pd.DataFrame())

        assert "connect()" in str(exc_info.value)


@pytest.mark.integration
class TestSimulationPipeline:
    """End-to-end tests for simulation workflow."""

    def test_simulator_manager_workflow(self):
        """Test manager workflow with simulator."""
        from grmpy.core.contracts import Config, SimulationConfig
        from grmpy.simulators.adapter_roy_model import RoyModelSimulator
        from grmpy.simulators.manager import SimulatorManager

        # Create config with minimal parameters
        config = Config(
            simulation=SimulationConfig(
                agents=50,
                seed=42,
                coefficients_treated=[1.0, 0.5],
                coefficients_untreated=[0.5, 0.3],
                coefficients_choice=[0.0, 1.0],
                covariance=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

        # Create manager
        simulator = RoyModelSimulator()
        manager = SimulatorManager(simulator, config)

        # This test verifies the structure without running legacy code
        manager.connect()
        assert manager.is_connected
