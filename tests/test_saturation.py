import pytest
from sklearn.utils.estimator_checks import check_estimator

from mamimo.saturation import (
    AdbudgSaturation,
    BoxCoxSaturation,
    ExponentialSaturation,
    HillSaturation,
)


@pytest.mark.parametrize(
    "estimator",
    [
        BoxCoxSaturation(),
        AdbudgSaturation(),
        HillSaturation(),
        ExponentialSaturation(),
    ],
)
def test_check_estimator(estimator):
    """Test if check_estimator passes."""
    check_estimator(estimator)
