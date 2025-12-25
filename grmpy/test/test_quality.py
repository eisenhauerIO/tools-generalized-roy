"""This module contains test that check the code quality of the package."""
import pytest


@pytest.mark.skip(reason="Linting is run separately via 'hatch run lint'")
def test_lint():
    """Code quality is checked via ruff in CI/CD pipeline."""
    pass
