"""Analyze trained marketing mix models."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def breakdown(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    group_channels: Optional[Dict[str, List[str]]] = None,
):
    """
    Compute the contributions for each channel.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The trained marketing mix model. Should be a pipeline consisting of two steps:
        1. preprocessing (e.g. adstock transformations)
        2. regression via a linear model.

    X : pd.Dataframe of shape (n_samples, n_features)
        The training data.

    y : np.ndarray, 1-dimensional
        The training labels.

    group_channels : Dict[str, List[str]], default=None
        Create new channels by grouping (i.e. summing) the channels in the input.

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

    if group_channels is not None:
        for new_channel, old_channels in group_channels.items():
            adjusted_breakdown[new_channel] = sum(
                [adjusted_breakdown.pop(old_channel) for old_channel in old_channels]
            )

    return adjusted_breakdown
