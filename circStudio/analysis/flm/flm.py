import numpy as np
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from ..metrics import daily_profile


class FLM:
    """ Class for Functional Linear Modelling"""

    def __init__(self, basis, sampling_freq, max_order=None):

        bases = ('fourier', 'spline')
        if basis not in bases:
            raise ValueError(
                '`basis` must be "%s". You passed: "%s"' %
                ('" or "'.join(bases), basis)
            )

        self.basis = basis
        self.sampling_freq = sampling_freq
        self.nsamples = None
        self.max_order = max_order
        self.basis_functions = None
        self.beta = {}

    @staticmethod
    def _spread(data: np.ndarray) -> float:
        """
        Estimate the variability in the data in a way that is robust to outliers.

        Actigraphy signals often contain spikes or artifacts (e.g., non-wear). If
        variability is estimated using standard deviation alone, a few extreme values
        can make the signal appear more variable than it truly is.

        standard deviation alone, a few extreme values can make the data look more variable
        than it really is.

        This function computes two dispersion metrics:
        * Standard deviation (sensitive to outliers)
        * Interquartile range (IQR), rescaled so it behaves like a standard deviation when
        the data are approximately normal.

        The smaller of the two values is returned. This provides a conservative, robust estimate
        of the signal's variability.

        Parameters
        ----------
        data: np.ndarray
            Data to estimate variability for.

        Returns
        -------
        float
            Robust estimate of data's variability (either standard deviation or interquartile range).
        """
        # Convert IQR to a standard-deviation-like scale for normally distributed data
        normalization_factor = 1.349

        # Compute standard deviation using the middle 50% of data (less sensitive to outliers)
        iqr = (np.percentile(data, 75) - np.percentile(data, 25)) / normalization_factor

        # Return the smaller of standard deviation and scaled iqr
        return np.minimum(data.std(ddof=1), iqr)


    def _bandwidth_factor(self, data):
        """
        Compute a bandwidth factor given a data array.

        The amount of smoothing increases when the signal is more variable
        and decreases when more data points are available.

        Parameters
        ----------
        data : np.ndarray
            Data to compute bandwidth factor.

        Returns
        -------
        float
            Bandwidth factor used to determine Gaussian kernel width.

        """
        return self._spread(data) * np.power(data.size, -0.2)


    def _get_kernel_size(self, data, method):
        """
        Compute Gaussian kernel width (sigma)

        The kernel width controls how strongly the signal is smoothed:
            - Larger values -> stronger smoothing
            - Smaller values -> less smoothing

        The width can be selected automatically using common rules
        ('scott' or 'silverman'), or set manually by providing a number.

        Parameters
        ----------
        data : np.ndarray
            Data to be smoothed.

        method : {'scott', 'silverman'} or float
            Strategy for selecting the kernel width. If a float is provided,
            it is used directly as the smoothing parameter.

        Returns
        -------
        float
            Gaussian kernel width (sigma)
        """
        # Calculate optimal kernel bandwidth (i.e., sigma)
        bw = self._bandwidth_factor(data)

        match method:
            case 'scott':
                # Standard rule-of-thumb smoothing (more smoothing)
                return 1.059 * bw
            case 'silverman':
                # Slightly more conservative smoothing rule (less smoothing)
                return 0.9 * bw
            case _:
                # If the user provides a numeric value, use it directly
                if np.isscalar(method):
                    return method
                else:
                    # Raise ValueError in all other scenarios
                    raise ValueError('Method must be "scott", "silverman" or a scalar')

    def fit(self, data, verbose=False):
        """Fit the actigraphy data using a basis function expansion.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be fitted.
        binarize: bool.
            If True, the data are binarized (i.e 0 or 1).
            Default is False.
        verbose : bool.
            If True, print the fit summary.
            Default is False.

        Returns
        -------
        y_est : ndarray
            Returns the functional form of the actigraphy data.
        """

        daily_avg = daily_profile(data)
        self.nsamples = daily_avg.index.size

        # Fourier
        if self.basis == 'fourier':

            X = np.stack(self.basis_functions, axis=1)
            y = daily_avg.values
            model = sm.OLS(y, X)
            results = model.fit()

            if verbose:
                print(results.summary())

            self.beta['beta'] = results.params

        # Spline
        elif self.basis == 'spline':

            from scipy.interpolate import splrep

            T = self.nsamples
            t = np.linspace(0, T, T, endpoint=True)
            k = 3 if self.max_order is None else self.max_order

            if verbose:
                print('Finding the {}-degree B-spline representation of'
                      'the input data'.format(k))

            self.beta['beta'] = list(
                splrep(t, daily_avg.values, k=k)
            )

    def evaluate(self, r=10):
        """Evaluate the basis function expansion.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements used to create the basis functions.
        r : int
            Ratio between the number of points at which the basis functions are
            evaluated and the number of points at which the basis functions
            were fitted.
            Default is 10.
            N.B.: only valid for splines.

        Returns
        -------
        y_est : ndarray
            Returns the functional form of the actigraphy data.
        """

        if not self.beta:
            raise ValueError(
                'The basis function expansion parameters are empty.\n'
                'Please run the `self.fit` method first.'
            )

        # Fourier
        if self.basis == 'fourier':
            X = np.stack(self.basis_functions, axis=1)
            y_est = np.dot(X, self.beta['beta'])
            return y_est

        # Spline
        elif self.basis == 'spline':
            from scipy.interpolate import BSpline
            T = self.nsamples
            t = np.linspace(0, T, r*T, endpoint=True)
            y_est = BSpline(*self.beta['beta'], extrapolate=False)(t)
            return y_est

    def smooth(self, data, method='scott', verbose=False):
        """Smooth the actigraphy data using a gaussian kernel.

        Wrapper for the scipy.ndimage.gaussian_filter1d function.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be smoothed.
        binarize: bool.
            If True, the data are binarized (i.e 0 or 1).
            Default is False.
        method: str, float.
            Method to calculate the width of the gaussian kernel.
            Available methods are `scott`, `silverman`. Method can be
            a scalar value too.
            Default is `scott`.
        verbose: bool.
            If True, print the kernel size used to smooth the data.
            Default is False.

        Returns
        -------
        y_est : ndarray
            Returns the smoothed form of the actigraphy data.
        """

        daily_avg = daily_profile(data)

        # Calculate optimal kernel size
        bw = self._get_kernel_size(daily_avg.values, method=method)

        if verbose:
            print('Kernel size used to smooth the data: {}'.format(bw))

        return gaussian_filter1d(
            daily_avg,
            sigma=bw,
            mode='wrap'
        )