"""Test carryover."""

import pytest
from sklearn.utils.estimator_checks import check_estimator

from mamimo.carryover import ExponentialCarryover, GeneralGaussianCarryover


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialCarryover(),
        GeneralGaussianCarryover(),
    ],
)
def test_check_estimator(estimator):
    """Test if check_estimator passes."""
    check_estimator(estimator)
