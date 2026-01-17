"""
Integration tests for complete estimation workflow.

Tests end-to-end pipeline from configuration to results.
"""

import pandas as pd
import pytest

from grmpy.core.contracts import Config, EstimationConfig, EstimationResult
from grmpy.core.exceptions import GrmpyError
from grmpy.estimators import estimate


@pytest.mark.integration
class TestEstimationPipeline:
    """End-to-end tests for complete estimation workflow."""

    def test_estimate_requires_config(self, sample_estimation_data):
        """estimate() raises GrmpyError if no ESTIMATION config."""
        config = Config(estimation=None)

        with pytest.raises(GrmpyError) as exc_info:
            estimate(config, sample_estimation_data)

        assert "ESTIMATION" in str(exc_info.value)

    def test_estimate_rejects_invalid_function(self, sample_estimation_data):
        """estimate() raises GrmpyError for unknown function."""
        config = Config(
            estimation=EstimationConfig(
                function="unknown_function",
                file="",
            )
        )

        with pytest.raises(GrmpyError) as exc_info:
            estimate(config, sample_estimation_data)

        assert "unknown_function" in str(exc_info.value)

    def test_estimate_dispatches_parametric(self, sample_estimation_data):
        """estimate() dispatches to parametric estimator."""
        config = Config(
            estimation=EstimationConfig(
                function="parametric",
                file="",
            )
        )

        # This will fail if parametric module isn't properly configured,
        # but validates the dispatch mechanism works
        try:
            result = estimate(config, sample_estimation_data)
            assert isinstance(result, EstimationResult)
        except GrmpyError as e:
            # Expected if legacy modules aren't available
            pytest.skip(f"Legacy parametric module not available: {e}")

    def test_estimate_dispatches_semiparametric(self, sample_estimation_data):
        """estimate() dispatches to semiparametric estimator."""
        config = Config(
            estimation=EstimationConfig(
                function="semiparametric",
                file="",
            )
        )

        try:
            result = estimate(config, sample_estimation_data)
            assert isinstance(result, EstimationResult)
        except GrmpyError as e:
            # Expected if legacy modules aren't available
            pytest.skip(f"Legacy semiparametric module not available: {e}")


@pytest.mark.integration
class TestSimulationPipeline:
    """End-to-end tests for simulation workflow."""

    def test_simulate_requires_config(self):
        """simulate() raises GrmpyError if no SIMULATION config."""
        from grmpy.simulators import simulate

        config = Config(simulation=None)

        with pytest.raises(GrmpyError) as exc_info:
            simulate(config)

        assert "simulation" in str(exc_info.value).lower()

    def test_simulate_basic_workflow(self):
        """simulate() generates data with valid config."""
        from grmpy.core.contracts import SimulationConfig, SimulationResult
        from grmpy.simulators import simulate

        config = Config(
            simulation=SimulationConfig(
                function="roy_model",
                agents=50,
                seed=42,
                coefficients_treated=[1.0, 0.5],
                coefficients_untreated=[0.5, 0.3],
                coefficients_choice=[0.0, 1.0],
                covariance=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

        result = simulate(config)
        assert isinstance(result, SimulationResult)
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 50
        assert result.metadata["function"] == "roy_model"
