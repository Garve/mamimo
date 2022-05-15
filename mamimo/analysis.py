"""Analyze trained marketing mix models."""

from typing import Any

import numpy as np
import pandas as pd


def breakdown(model: Any, X: pd.DataFrame, y: np.ndarray):
    """
    Compute the contributions for each channel.

    Parameters
    ----------
    model : Any
        The trained marketing mix model. Should be a pipeline consisting of two steps:
        1. preprocessing (e.g. adstock transformations)
        2. regression via a linear model.

    X : pd.Dataframe of shape (n_samples, n_features)
        The training data.

    y : np.ndarray, 1-dimensional
        The training labels.

    Returns
    -------
    pd.DataFrame
        A table consisting of the contributions of each channel in each timestep.
        The row-wise sum of this dataframe equals `y`.

    """
    preprocessing = model.steps[-2][1]
    regression = model.steps[-1][1]
    channel_names = preprocessing.get_feature_names_out()

    after_preprocessing = pd.DataFrame(
        preprocessing.transform(X), columns=channel_names, index=X.index
    )

    regression_weights = pd.Series(regression.coef_, index=channel_names)

    base = regression.intercept_

    unadjusted_breakdown = after_preprocessing.mul(regression_weights).assign(Base=base)
    adjusted_breakdown = unadjusted_breakdown.div(
        unadjusted_breakdown.sum(axis=1), axis=0
    ).mul(y, axis=0)

    return adjusted_breakdown
