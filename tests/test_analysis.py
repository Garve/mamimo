"""Test analysis."""

import numpy as np
import pytest
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mamimo.analysis import breakdown
from mamimo.carryover import ExponentialCarryover
from mamimo.datasets import load_fake_mmm
from mamimo.linear_model import LinearRegression
from mamimo.saturation import ExponentialSaturation
from mamimo.time_utils import PowerTrend, add_date_indicators, add_time_features


@pytest.fixture()
def create_model():
    """Create a model to test the breakdown function."""
    data = load_fake_mmm()

    X = data.drop(columns=["Sales"])
    y = data.Sales

    X = X.pipe(add_time_features, month=True).pipe(
        add_date_indicators, special_date=["2020-01-05"]
    )
    X["Trend"] = range(200)

    preprocessing = make_column_transformer(
        (
            make_pipeline(
                ExponentialCarryover(window=4, strength=0.5),
                ExponentialSaturation(exponent=0.0001),
            ),
            ["TV"],
        ),
        (
            make_pipeline(
                ExponentialCarryover(window=2, strength=0.2),
                ExponentialSaturation(exponent=0.0001),
            ),
            ["Radio"],
        ),
        (
            make_pipeline(
                ExponentialCarryover(), ExponentialSaturation(exponent=0.0001)
            ),
            ["Banners"],
        ),
        (OneHotEncoder(sparse_output=False), ["month"]),
        (PowerTrend(power=1.2), ["Trend"]),
        (ExponentialCarryover(window=10, strength=0.6), ["special_date"]),
    )

    model = make_pipeline(
        preprocessing, LinearRegression(positive=True, fit_intercept=False)
    )

    return model.fit(X, y), X, y


def test_breakdown(create_model):
    """Tests if the sum of channel contribution equals the observed targets."""
    model, X, y = create_model
    breakdowns = breakdown(model, X, y)

    np.testing.assert_array_almost_equal(breakdowns.sum(axis=1), y)


def test_group(create_model):
    """Checks if grouping together channels works."""
    model, X, y = create_model

    breakdowns = breakdown(
        model,
        X,
        y,
        group_channels={
            "Base": [f"onehotencoder__month_{i}" for i in range(1, 13)]
            + ["powertrend__Trend"],
            "Media": ["pipeline-1__TV", "pipeline-2__Radio", "pipeline-3__Banners"],
        },
    )

    assert breakdowns.columns.tolist() == [
        "exponentialcarryover__special_date",
        "Base",
        "Media",
    ]
