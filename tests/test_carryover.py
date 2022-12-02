"""Test carryover."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, lists
from sklearn.utils.estimator_checks import check_estimator

from mamimo.carryover import ExponentialCarryover, GeneralGaussianCarryover

numpy_arrays = lists(floats(min_value=0, max_value=1e30), min_size=1).map(
    lambda x: np.array(x).reshape(-1, 1)
)


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


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialCarryover(),
        GeneralGaussianCarryover(),
    ],
)
@given(inputs=numpy_arrays)
def test_output_is_greater_than_input(inputs, estimator):
    """Test if carryover increases the values of the original input array."""
    outputs = estimator.fit_transform(inputs)
    assert (outputs >= inputs).all()


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialCarryover(),
        GeneralGaussianCarryover(),
    ],
)
@given(inputs=numpy_arrays)
def test_output_is_equal_to_input_in_the_first_component(inputs, estimator):
    """Test if carryover is equal to the input array in the first component."""
    outputs = estimator.fit_transform(inputs)
    assert outputs[0] == inputs[0]
