"""Test the linear models."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from mamimo.linear_model import (
    ImbalancedLinearRegression,
    LADRegression,
    LinearRegression,
    QuantileRegression,
)

test_batch = [
    (np.array([0, 0, 3, 0, 6]), 3),
    (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
    (np.array([4, -4]), 0),
    (np.array([0.1]), 1000),
]


def _create_dataset(coefs, intercept, noise=0.0):
    np.random.seed(0)
    X = np.random.randn(1000, coefs.shape[0])
    y = X @ coefs + intercept + noise * np.random.randn(1000)

    return X, y


@pytest.mark.parametrize("coefs, intercept", test_batch)
@pytest.mark.parametrize(
    "model",
    [LADRegression, QuantileRegression, ImbalancedLinearRegression, LinearRegression],
)
def test_coefs_and_intercept__no_noise(coefs, intercept, model):
    """Regression problems without noise."""
    X, y = _create_dataset(coefs, intercept)
    regressor = model()
    regressor.fit(X, y)

    assert regressor.score(X, y) > 0.99


@pytest.mark.parametrize("coefs, intercept", test_batch)
@pytest.mark.parametrize(
    "model",
    [LADRegression, QuantileRegression, ImbalancedLinearRegression, LinearRegression],
)
def test_score(coefs, intercept, model):
    """Tests with noise on an easy problem. Parameter reconstruction should be easy."""
    X, y = _create_dataset(coefs, intercept, noise=1.0)
    regressor = model()
    regressor.fit(X, y)

    np.testing.assert_almost_equal(regressor.coef_, coefs, decimal=1)
    np.testing.assert_almost_equal(regressor.intercept_, intercept, decimal=1)


@pytest.mark.parametrize("coefs, intercept", test_batch)
@pytest.mark.parametrize(
    "model",
    [LADRegression, QuantileRegression, ImbalancedLinearRegression, LinearRegression],
)
def test_coefs_and_intercept__no_noise_positive(coefs, intercept, model):
    """Test with only positive coefficients."""
    X, y = _create_dataset(coefs, intercept, noise=0.0)
    regressor = model(positive=True)
    regressor.fit(X, y)

    assert all(regressor.coef_ >= 0)
    assert regressor.score(X, y) > 0.3


@pytest.mark.parametrize("coefs, intercept", test_batch)
@pytest.mark.parametrize(
    "model",
    [LADRegression, QuantileRegression, ImbalancedLinearRegression, LinearRegression],
)
def test_fit_intercept_and_copy(coefs, intercept, model):
    """Test if fit_intercept and copy_X work."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    regressor = model(fit_intercept=False, copy_X=False)
    regressor.fit(X, y)

    assert regressor.intercept_ == 0.0


@pytest.mark.parametrize(
    "model",
    [LADRegression, QuantileRegression, ImbalancedLinearRegression, LinearRegression],
)
def test_check_estimator(model):
    """Conduct all scikit-learn estimator tests."""
    regressor = model()

    check_estimator(regressor)
