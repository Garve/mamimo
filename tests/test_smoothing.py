import pytest
from sklearn.utils.estimator_checks import check_estimator

from mamimo.smoothing import ExponentialDecaySmoother, GeneralGaussianSmoother


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialDecaySmoother(),
        GeneralGaussianSmoother(),
    ],
)
def test_check_estimator(estimator):
    """Test if check_estimator passes."""
    check_estimator(estimator)
