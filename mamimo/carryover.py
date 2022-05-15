"""Smooth time series data."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import convolve2d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class Carryover(BaseEstimator, TransformerMixin, ABC):
    """
    Smooth the columns of an array by applying a convolution.

    Parameters
    ----------
    window : int
        Size of the sliding window. The effect of a holiday will reach from
        approximately date - `window/2 * frequency` to date + `window/2 * frequency`,
        i.e. it is centered around the dates in `dates`.

    mode : str
        Which convolution mode to use. Can be one of
            - "full": The output is the full discrete linear convolution of the inputs.
            - "valid": The output consists only of those elements that do not rely on
            the zero-padding.
            - "same": The output is the same size as the first input, centered with
            respect to the 'full' output.

    """

    def __init__(
        self,
        window: int,
        mode: str,
    ) -> None:
        """Initialize."""
        self.window = window
        self.mode = mode

    @abstractmethod
    def _get_sliding_window(self) -> np.ndarray:
        """
        Calculate the sliding window.

        Returns
        -------
        sliding_window : np.array
            The sliding window.

        """

    def fit(self, X: np.ndarray, y: None = None) -> Carryover:
        """
        Fit the estimator.

        The frequency is computed and the sliding window is created.

        Parameters
        ----------
        X : np.ndarray
            Used for inferring the frequency, if not provided during initialization.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        Carryover
            Fitted transformer.

        """
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = self._get_sliding_window()
        self.sliding_window_ = (
            self.sliding_window_.reshape(-1, 1) / self.sliding_window_.sum()
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Add the new date feature to the dataframe.

        Parameters
        ----------
        X : np.ndarray
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        np.ndarray
            The input dataframe with an additional column for special dates.

        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        convolution = convolve2d(X, self.sliding_window_, mode=self.mode)

        if self.mode == "full" and self.window > 1:
            convolution = convolution[: -self.window + 1]

        return convolution


class GeneralGaussianCarryover(Carryover):
    """
    Smoothes time series data with a Gaussian window.

    Smooth the columns of an array by applying a convolution with a generalized
    Gaussian curve.

    Parameters
    ----------
    window : int, default=1
        Size of the sliding window. The effect of a holiday will reach from
        approximately date - `window/2 * frequency` to date + `window/2 * frequency`,
        i.e. it is centered around the dates in `dates`.

    p : float, default=1
        Parameter for the shape of the curve. p=1 yields a typical Gaussian curve
        while p=0.5 yields a Laplace curve, for example.

    sig : float, default=1
        Parameter for the standard deviation of the bell-shaped curve.

    tails : str, default="both"
        Which tails to use. Can be one of
            - "left"
            - "right"
            - "both"

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> GeneralGaussianCarryover().fit_transform(X)
    array([[0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.]])

    >>> GeneralGaussianCarryover(window=5, p=1, sig=1).fit_transform(X)
    array([[0.        ],
           [0.05448868],
           [0.24420134],
           [0.40261995],
           [0.24420134],
           [0.05448868],
           [0.        ]])

    >>> GeneralGaussianCarryover(window=7, tails="right").fit_transform(X)
    array([[0.        ],
           [0.        ],
           [0.        ],
           [0.57045881],
           [0.34600076],
           [0.0772032 ],
           [0.00633722]])

    """

    def __init__(
        self,
        window: int = 1,
        p: float = 1,
        sig: float = 1,
        tails: str = "both",
    ) -> None:
        """Initialize."""
        super().__init__(window, mode="same")
        self.p = p
        self.sig = sig
        self.tails = tails

    def _get_sliding_window(self) -> np.ndarray:
        """
        Calculate the sliding window.

        Returns
        -------
        sliding_window : np.array
            The sliding window.

        Raises
        ------
        ValueError
            If the provided value for `tails` is not "left", "right" or "both".

        """
        sliding_window = np.exp(
            -0.5
            * np.abs(np.arange(-self.window // 2 + 1, self.window // 2 + 1) / self.sig)
            ** (2 * self.p)
        )
        if self.tails == "left":
            sliding_window[self.window // 2 + 1 :] = 0
        elif self.tails == "right":
            sliding_window[: self.window // 2] = 0
        elif self.tails != "both":
            raise ValueError(
                "tails keyword has to be one of 'both', 'left' or 'right'."
            )
        return sliding_window


class ExponentialCarryover(Carryover):
    """
    Smoothes time series data with an exponential window.

    Smooth the columns of an array by applying a convolution with a exponentially
    decaying curve. This class can be used for modelling carry over effects in
    marketing mix models.

    Parameters
    ----------
    window : int, default=1
        Size of the sliding window. The effect of a holiday will reach from
        approximately date - `window/2 * frequency` to date + `window/2 * frequency`,
        i.e. it is centered around the dates in `dates`.

    strength : float, default=0.0
        Fraction of the spending effect that is carried over.

    peak : float, default=0.0
        Where the carryover effect peaks.

    exponent : float, default=1.0
        To further widen or narrow the carryover curve. A value of 1.0 yields a normal
        exponential decay. With values larger than 1.0, a super exponential decay can
        be achieved.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> ExponentialCarryover().fit_transform(X)
    array([[0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.]])

    >>> ExponentialCarryover(window=3, strength=0.5).fit_transform(X)
    array([[0.        ],
           [0.        ],
           [0.        ],
           [0.57142857],
           [0.28571429],
           [0.14285714],
           [0.        ]])

    >>> ExponentialCarryover(window=3, strength=0.5, peak=1).fit_transform(X)
    array([[0.  ],
           [0.  ],
           [0.  ],
           [0.25],
           [0.5 ],
           [0.25],
           [0.  ]])

    """

    def __init__(
        self,
        window: int = 1,
        strength: float = 0.0,
        peak: float = 0.0,
        exponent: float = 1.0,
    ) -> None:
        """Initialize."""
        super().__init__(window, mode="full")
        self.strength = strength
        self.peak = peak
        self.exponent = exponent

    def _get_sliding_window(self) -> np.ndarray:
        """
        Calculate the sliding window.

        Returns
        -------
        sliding_window : np.array
            The sliding window.

        """
        sliding_window = self.strength ** (
            np.abs(np.arange(self.window) - self.peak) ** self.exponent
        )

        return sliding_window
