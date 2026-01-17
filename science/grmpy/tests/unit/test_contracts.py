"""
Unit tests for core contracts and schemas.

Tests data validation, schema transformations, and result contracts.
"""

import numpy as np
import pandas as pd
import pytest

from grmpy.core.contracts import (
    Config,
    DataSchema,
    EstimationConfig,
    EstimationDataSchema,
    EstimationResult,
    SimulationConfig,
)
from grmpy.core.exceptions import ConfigurationError, DataValidationError


class TestDataSchema:
    """Tests for base DataSchema class."""

    def test_validate_accepts_valid_dataframe(self):
        """Schema accepts DataFrame with all required columns."""
        schema = DataSchema(required_fields=["A", "B"])
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        schema.validate(df)  # Should not raise

    def test_validate_rejects_missing_columns(self):
        """Schema raises DataValidationError for missing columns."""
        schema = DataSchema(required_fields=["A", "B", "C"])
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        with pytest.raises(DataValidationError) as exc_info:
            schema.validate(df)

        assert "C" in str(exc_info.value)
        assert "Missing required columns" in str(exc_info.value)

    def test_from_external_applies_mappings(self):
        """Schema transforms external field names to standard names."""
        schema = DataSchema(
            required_fields=["Y", "D"],
            field_mappings={"outcome": "Y", "treatment": "D"},
        )
        external_df = pd.DataFrame({"outcome": [1, 2], "treatment": [0, 1]})

        result = schema.from_external(external_df)

        assert "Y" in result.columns
        assert "D" in result.columns
        assert "outcome" not in result.columns

    def test_to_external_reverses_mappings(self):
        """Schema transforms standard names back to external format."""
        schema = DataSchema(
            required_fields=["Y", "D"],
            field_mappings={"outcome": "Y", "treatment": "D"},
        )
        internal_df = pd.DataFrame({"Y": [1, 2], "D": [0, 1]})

        result = schema.to_external(internal_df)

        assert "outcome" in result.columns
        assert "treatment" in result.columns


class TestEstimationDataSchema:
    """Tests for estimation input data validation."""

    def test_validate_accepts_valid_dataframe(self, sample_estimation_data):
        """Schema accepts DataFrame with all required columns."""
        schema = EstimationDataSchema()
        schema.validate(sample_estimation_data)  # Should not raise

    def test_validate_rejects_missing_columns(self):
        """Schema raises DataValidationError listing missing columns."""
        schema = EstimationDataSchema()
        incomplete_df = pd.DataFrame({"Y": [1, 2], "D": [0, 1]})  # Missing Z

        with pytest.raises(DataValidationError) as exc_info:
            schema.validate(incomplete_df)

        assert "Z" in str(exc_info.value)

    @pytest.mark.parametrize("missing_col", ["Y", "D", "Z"])
    def test_validate_identifies_each_missing_column(self, missing_col):
        """Schema correctly identifies which specific column is missing."""
        schema = EstimationDataSchema()
        cols = {"Y": [1], "D": [1], "Z": [1]}
        del cols[missing_col]

        with pytest.raises(DataValidationError) as exc_info:
            schema.validate(pd.DataFrame(cols))

        assert missing_col in str(exc_info.value)


class TestEstimationConfig:
    """Tests for estimation configuration contract."""

    def test_validate_accepts_valid_config(self):
        """Valid config passes validation."""
        config = EstimationConfig(
            method="parametric",
            file="data.pkl",
        )
        config.validate()  # Should not raise

    def test_validate_rejects_invalid_method(self):
        """Invalid method raises ConfigurationError with options."""
        config = EstimationConfig(
            method="invalid_method",
            file="data.pkl",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()

        assert "invalid_method" in str(exc_info.value)
        assert "parametric" in str(exc_info.value)  # Lists valid options

    def test_validate_rejects_invalid_optimizer(self):
        """Invalid optimizer raises ConfigurationError."""
        config = EstimationConfig(
            method="parametric",
            file="data.pkl",
            optimizer="invalid_optimizer",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()

        assert "invalid_optimizer" in str(exc_info.value)


class TestSimulationConfig:
    """Tests for simulation configuration contract."""

    def test_validate_accepts_valid_config(self):
        """Valid config passes validation."""
        config = SimulationConfig(agents=1000)
        config.validate()  # Should not raise

    def test_validate_rejects_non_positive_agents(self):
        """Non-positive agents raises ConfigurationError."""
        config = SimulationConfig(agents=0)

        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()

        assert "positive" in str(exc_info.value).lower()


class TestEstimationResult:
    """Tests for estimation result dataclass."""

    def test_result_stores_all_required_fields(self):
        """Result object stores MTE and related quantities."""
        result = EstimationResult(
            mte=np.array([0.1, 0.2]),
            mte_x=np.array([[0.1], [0.2]]),
            mte_u=np.array([0.05, 0.15]),
            quantiles=np.array([0.25, 0.75]),
            coefficients={"b0": np.array([1.0]), "b1": np.array([0.5])},
        )

        assert len(result.mte) == 2
        assert result.quantiles[0] == 0.25
        assert "b0" in result.coefficients

    def test_result_metadata_defaults_to_empty_dict(self):
        """Metadata field defaults to empty dict if not provided."""
        result = EstimationResult(
            mte=np.zeros(1),
            mte_x=np.zeros((1, 1)),
            mte_u=np.zeros(1),
            quantiles=np.zeros(1),
            coefficients={},
        )

        assert result.metadata == {}

    def test_to_dict_converts_arrays(self):
        """to_dict converts numpy arrays to lists."""
        result = EstimationResult(
            mte=np.array([0.1, 0.2]),
            mte_x=np.array([[0.1]]),
            mte_u=np.array([0.1]),
            quantiles=np.array([0.5]),
            coefficients={"b0": np.array([1.0])},
        )

        d = result.to_dict()

        assert isinstance(d["mte"], list)
        assert isinstance(d["coefficients"]["b0"], list)


class TestConfig:
    """Tests for root Config class."""

    def test_from_dict_creates_estimation_config(self, sample_config_dict):
        """from_dict creates EstimationConfig from ESTIMATION section."""
        config = Config.from_dict(sample_config_dict)

        assert config.estimation is not None
        assert config.estimation.method == "parametric"

    def test_from_dict_creates_simulation_config(self, sample_config_dict):
        """from_dict creates SimulationConfig from SIMULATION section."""
        config = Config.from_dict(sample_config_dict)

        assert config.simulation is not None
        assert config.simulation.agents == 1000

    def test_from_dict_handles_missing_sections(self):
        """from_dict handles configs with missing sections gracefully."""
        config = Config.from_dict({})

        assert config.estimation is None
        assert config.simulation is None
