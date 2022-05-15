"""Saturation classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class Saturation(BaseEstimator, TransformerMixin, ABC):
    """Base class for all saturations, such as Box-Cox, Adbudg, ..."""

    def fit(self, X: np.ndarray, y: None = None) -> Saturation:
        """
        Fit the transformer.

        In this special case, nothing is done.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        Saturation
            Fitted transformer.

        """
        X = check_array(X)
        self._check_n_features(X, reset=True)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the saturation effect.

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed.

        Returns
        -------
        np.ndarray
            Data with saturation effect applied.

        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        return self._transformation(X)

    @abstractmethod
    def _transformation(self, X: np.ndarray) -> np.ndarray:
        """Generate the transformation formula."""


class BoxCoxSaturation(Saturation):
    """
    Apply the Box-Cox saturation.

    The formula is ((x + shift) ** exponent-1) / exponent if exponent!=0,
    else ln(x+shift).

    Parameters
    ----------
    exponent: float, default=1.0
        The exponent.

    shift : float, default=1.0
        The shift.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> BoxCoxSaturation(exponent=0.5).fit_transform(X)
    array([[ 0.82842712, 61.27716808],
           [ 1.46410162, 61.27716808],
           [ 2.        , 61.27716808]])

    """

    def __init__(self, exponent: float = 1.0, shift: float = 1.0) -> None:
        """Initialize."""
        self.exponent = exponent
        self.shift = shift

    def _transformation(self, X: np.ndarray) -> np.ndarray:
        """Generate the transformation formula."""
        if self.exponent != 0:
            return ((X + self.shift) ** self.exponent - 1) / self.exponent
        else:
            return np.log(X + self.shift)


class AdbudgSaturation(Saturation):
    """
    Apply the Adbudg saturation.

    The formula is x ** exponent / (denominator_shift + x ** exponent).

    Parameters
    ----------
    exponent : float, default=1.0
        The exponent.

    denominator_shift : float, default=1.0
        The shift in the denominator.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> AdbudgSaturation().fit_transform(X)
    array([[0.5       , 0.999001  ],
           [0.66666667, 0.999001  ],
           [0.75      , 0.999001  ]])

    """

    def __init__(self, exponent: float = 1.0, denominator_shift: float = 1.0) -> None:
        """Initialize."""
        self.exponent = exponent
        self.denominator_shift = denominator_shift

    def _transformation(self, X: np.ndarray) -> np.ndarray:
        """Generate the transformation formula."""
        return X**self.exponent / (self.denominator_shift + X**self.exponent)


class HillSaturation(Saturation):
    """
    Apply the Hill saturation.

    The formula is 1 / (1 + (half_saturation / x) ** exponent).

    Parameters
    ----------
    exponent : float, default=1.0
        The exponent.

    half_saturation : float, default=1.0
        The point of half saturation, i.e. Hill(half_saturation) = 0.5.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> HillSaturation().fit_transform(X)
    array([[0.5       , 0.999001  ],
           [0.66666667, 0.999001  ],
           [0.75      , 0.999001  ]])

    """

    def __init__(self, exponent: float = 1.0, half_saturation: float = 1.0) -> None:
        """Initialize."""
        self.half_saturation = half_saturation
        self.exponent = exponent

    def _transformation(self, X: np.ndarray) -> np.ndarray:
        """Generate the transformation formula."""
        eps = np.finfo(np.float64).eps
        return 1 / (1 + (self.half_saturation / (X + eps)) ** self.exponent)


class ExponentialSaturation(Saturation):
    """
    Apply exponential saturation.

    The formula is 1 - exp(-exponent * x).

    Parameters
    ----------
    exponent : float, default=1.0
        The exponent.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> ExponentialSaturation().fit_transform(X)
    array([[0.63212056, 1.        ],
           [0.86466472, 1.        ],
           [0.95021293, 1.        ]])

    """

    def __init__(self, exponent: float = 1.0) -> None:
        """Initialize."""
        self.exponent = exponent

    def _transformation(self, X: np.ndarray) -> np.ndarray:
        """Generate the transformation formula."""
        return 1 - np.exp(-self.exponent * X)
