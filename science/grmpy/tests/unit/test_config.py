"""
Unit tests for configuration management.

Tests configuration loading, merging, and validation.
"""

import pytest
import yaml

from grmpy.config import DEFAULTS, _deep_merge, _load_yaml, process_config
from grmpy.core.exceptions import ConfigurationError


class TestDeepMerge:
    """Tests for deep merge functionality."""

    def test_merge_flat_dicts(self):
        """Merge flat dictionaries with override precedence."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)

        assert result["a"] == 1  # From base
        assert result["b"] == 3  # Overridden
        assert result["c"] == 4  # From override

    def test_merge_nested_dicts(self):
        """Merge preserves nested structure."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 3, "inner3": 4}}

        result = _deep_merge(base, override)

        assert result["outer"]["inner1"] == 1  # Preserved
        assert result["outer"]["inner2"] == 3  # Overridden
        assert result["outer"]["inner3"] == 4  # Added

    def test_merge_does_not_modify_original(self):
        """Merge returns new dict without modifying originals."""
        base = {"a": 1}
        override = {"b": 2}

        result = _deep_merge(base, override)

        assert "b" not in base
        assert result is not base


class TestLoadYaml:
    """Tests for YAML loading."""

    def test_load_valid_yaml(self, temp_directory):
        """Load valid YAML file."""
        config_path = temp_directory / "test.yml"
        config_path.write_text("key: value\nnested:\n  inner: 42")

        result = _load_yaml(config_path)

        assert result["key"] == "value"
        assert result["nested"]["inner"] == 42

    def test_load_empty_yaml_returns_empty_dict(self, temp_directory):
        """Empty YAML file returns empty dict."""
        config_path = temp_directory / "empty.yml"
        config_path.write_text("")

        result = _load_yaml(config_path)

        assert result == {}

    def test_load_invalid_yaml_raises_error(self, temp_directory):
        """Invalid YAML raises ConfigurationError."""
        config_path = temp_directory / "invalid.yml"
        config_path.write_text("key: [unclosed bracket")

        with pytest.raises(ConfigurationError) as exc_info:
            _load_yaml(config_path)

        assert "Failed to parse" in str(exc_info.value)


class TestProcessConfig:
    """Tests for main configuration processing."""

    def test_process_config_loads_file(self, temp_directory):
        """process_config loads and parses YAML file."""
        config_path = temp_directory / "config.yml"
        config_content = {
            "ESTIMATION": {
                "method": "parametric",
                "file": "data.pkl",
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        config = process_config(config_path)

        assert config.estimation is not None
        assert config.estimation.method == "parametric"

    def test_process_config_merges_defaults(self, temp_directory):
        """process_config merges user config with defaults."""
        config_path = temp_directory / "config.yml"
        config_content = {
            "ESTIMATION": {
                "method": "parametric",
                "file": "data.pkl",
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        config = process_config(config_path)

        # Should have default optimizer
        assert config.estimation.optimizer == "BFGS"

    def test_process_config_raises_for_missing_file(self):
        """process_config raises ConfigurationError for missing file."""
        with pytest.raises(ConfigurationError) as exc_info:
            process_config("nonexistent.yml")

        assert "not found" in str(exc_info.value)

    def test_process_config_validates_estimation_method(self, temp_directory):
        """process_config validates estimation method."""
        config_path = temp_directory / "config.yml"
        config_content = {
            "ESTIMATION": {
                "method": "invalid_method",
                "file": "data.pkl",
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        with pytest.raises(ConfigurationError) as exc_info:
            process_config(config_path)

        assert "invalid_method" in str(exc_info.value)


class TestDefaults:
    """Tests for default configuration values."""

    def test_defaults_has_estimation_section(self):
        """DEFAULTS contains ESTIMATION section."""
        assert "ESTIMATION" in DEFAULTS

    def test_defaults_has_simulation_section(self):
        """DEFAULTS contains SIMULATION section."""
        assert "SIMULATION" in DEFAULTS

    def test_estimation_defaults_has_method(self):
        """Estimation defaults include method."""
        assert "method" in DEFAULTS["ESTIMATION"]
        assert DEFAULTS["ESTIMATION"]["method"] == "parametric"

    def test_estimation_defaults_has_optimizer(self):
        """Estimation defaults include optimizer."""
        assert "optimizer" in DEFAULTS["ESTIMATION"]
        assert DEFAULTS["ESTIMATION"]["optimizer"] == "BFGS"
