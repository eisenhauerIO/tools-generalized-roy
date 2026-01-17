"""
Regression tests for Carneiro, Heckman, Vytlacil (2011) replication.

Tests semiparametric MTE estimation against known results from the
American Economic Review paper using mock NLSY data.

Reference:
    Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011).
    Estimating Marginal Returns to Education.
    American Economic Review, 101(6), 2754-2781.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import grmpy
from grmpy.core.contracts import Config, EstimationConfig

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Covariates used in the Carneiro et al. specification
COVARIATES_OUTCOME = [
    "exp",
    "expsq",
    "lwage5",
    "lurate",
    "cafqt",
    "cafqtsq",
    "mhgc",
    "mhgcsq",
    "numsibs",
    "numsibssq",
    "urban14",
    "lavlocwage17",
    "lavlocwage17sq",
    "avurate",
    "avuratesq",
    "d57",
    "d58",
    "d59",
    "d60",
    "d61",
    "d62",
    "d63",
]

COVARIATES_CHOICE = [
    "const",
    "cafqt",
    "cafqtsq",
    "mhgc",
    "mhgcsq",
    "numsibs",
    "numsibssq",
    "urban14",
    "lavlocwage17",
    "lavlocwage17sq",
    "avurate",
    "avuratesq",
    "d57",
    "d58",
    "d59",
    "d60",
    "d61",
    "d62",
    "d63",
    "lwage5_17numsibs",
    "lwage5_17mhgc",
    "lwage5_17cafqt",
    "lwage5_17",
    "lurate_17",
    "lurate_17numsibs",
    "lurate_17mhgc",
    "lurate_17cafqt",
    "tuit4c",
    "tuit4cnumsibs",
    "tuit4cmhgc",
    "tuit4ccafqt",
    "pub4",
    "pub4numsibs",
    "pub4mhgc",
    "pub4cafqt",
]


@pytest.mark.regression
class TestCarneiroReplication:
    """Regression tests against Carneiro et al. (2011) results."""

    @pytest.fixture
    def carneiro_data(self) -> pd.DataFrame:
        """Load Carneiro mock data."""
        return pd.read_pickle(RESOURCES_DIR / "aer-replication-mock.pkl")

    @pytest.fixture
    def expected_mte(self) -> np.ndarray:
        """Load expected MTE values."""
        return pd.read_pickle(RESOURCES_DIR / "replication-results-mte.pkl")

    @pytest.fixture
    def expected_mte_u(self) -> np.ndarray:
        """Load expected MTE_U values."""
        return pd.read_pickle(RESOURCES_DIR / "replication-results-mte_u.pkl")

    def test_semiparametric_mte_replication(self, carneiro_data, expected_mte, expected_mte_u):
        """
        Replicate semiparametric MTE estimation from Carneiro et al. (2011).

        This test verifies that grmpy's semiparametric estimator produces
        results matching R's locpoly implementation used in the original paper.
        """
        config = Config(
            estimation=EstimationConfig(
                function="semiparametric",
                file="",
                dependent="wage",
                treatment="state",
                covariates_treated=COVARIATES_OUTCOME,
                covariates_untreated=COVARIATES_OUTCOME,
                covariates_choice=COVARIATES_CHOICE,
                bandwidth=0.322,
                gridsize=500,
                ps_range=(0.005, 0.995),
            )
        )

        result = grmpy.estimate(config, carneiro_data)

        np.testing.assert_array_almost_equal(result.mte_u, expected_mte_u, decimal=6)
        np.testing.assert_array_almost_equal(result.mte, expected_mte, decimal=6)
