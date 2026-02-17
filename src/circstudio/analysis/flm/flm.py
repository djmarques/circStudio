import numpy as np
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from scipy.interpolate import splrep, BSpline
from ..metrics import daily_profile


class FLM:
    """
    Class for Functional Linear Modelling.

    This class smoothes noisy activity signals in order to capture the main
    24-hour structure. It supports two main workflows:

    Daily-template modeling
        - `fit()` builds an average 24-hour profile and fits a smooth function using either
            - Fourier basis (period waves; ideal for circadian rhythms).
            - B-splines (flexible smooth curves).
        - `evaluate()` reconstructs the fitted 24-hour cycle.

        Use this to characterize the typical daily rhythm (shape or phase comparisons).

    Full time-series smoothing
        - `smooth_timeseries()` denoises the entire recording while preserving the original
        timeline.

        Smoothing widths are chosen automatically using robust rules that reduce
        sensitivity to spikes or artifacts (e.g., non-wear).

    Parameters
    ----------
    basis : {'fourier', 'spline'}
        Type of smooth representation used by `fit()` / `evaluate()`.

    sampling_freq : str or pandas-like frequency
        Sampling frequency of the signal (e.g., '1min', '30s').

    max_order : int or None, optional
        Controls model complexity (Fourier harmonics or spline degree).

    Attributes
    ----------
    beta : dict
        Stores fitted parameters after `fit()`.

    nsamples : int
        Number of time points in one 24-hour profile.

    Examples
    --------
    >>> flm = FLM(basis='fourier', sampling_freq='1min', max_order=3)
    >>> flm.fit(activity)
    >>> daily_curve = flm.evaluate()

    >>> daily_smoothed = flm.smooth_daily_profile(activity)
    >>> ts_smoothed = flm.smooth_timeseries(activity)
    """

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

    def _create_basis_functions(self, T):
        phi = []
        # Construct the fourier functions (cosine and sine)
        if self.basis == 'fourier':
            # T = int(pd.Timedelta('24H')/pd.Timedelta(self.sampling_freq))
            omega = 2*np.pi / T
            t = np.linspace(0, T, T, endpoint=False)
            phi.append(np.cos(0 * t))
            for n in np.arange(1, self.max_order+1):
                phi.append(np.cos(n * omega * t))
                phi.append(np.sin(n * omega * t))

        self.basis_functions = phi

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
        """
        Fit a smooth function representation of the actigraphy signal.

        The input data are first converted into an average 24-hour profile.
        This profile is then modeled using either:

            - Fourier basis functions
            - B-splines

        The fitted model parameters are stored internally and can later be
        evaluated using `evaluate()`.

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Actigraphy data to fit.

        verbose : bool.
            If True, print information about fitting process.
            Default is False.

        Returns
        -------
        None
            The fitted parameters are stored in `self.beta`.
        """
        # Convert the full signal into an average 24-hour pattern
        daily_avg = daily_profile(data)

        # Store the number of time points in the daily profile
        self.nsamples = daily_avg.index.size

        match self.basis:
            # Fourier
            case 'fourier':
                if self.basis_functions is None:
                    if self.max_order is None:
                        raise ValueError('Basis function and/or max_order must be set.')
                    self._create_basis_functions(self.nsamples)

                # Build matrix containing the periodic basis functions
                x = np.stack(self.basis_functions, axis=1)

                # Extract the observed 24-hour activity values
                y = daily_avg.values

                # Estimate how much each periodic component contributes to the signal
                model = sm.OLS(y, x)
                results = model.fit()

                if verbose:
                    print(results.summary())

                # Store the fitted Fourier coefficients
                self.beta['beta'] = results.params

            # B-splines
            case 'spline':
                # Create s time vector spanning one 24-h cycle
                t = np.linspace(
                    0,
                    self.nsamples,
                    self.nsamples,
                    endpoint=True)

                # Degrees of the spline (default: cubic spline)
                k = 3 if self.max_order is None else self.max_order

                if verbose:
                    print(f'Finding the {k}-degree B-spline representation of the input data')

                # Fit spline representation and store parameters
                self.beta['beta'] = list(
                    splrep(t, daily_avg.values, k=k)
                )


    def evaluate(self, r=10):
        """
        Reconstruct the smooth 24-hour activity curve from the fitted model.

        This method uses the parameters estimated in `fit()` to generate
        the modeled activity pattern across one 24-hour cycle.

        For spline models, the curve can be evaluated at a higher resolution
        than the original data by increasing `r

        Parameters
        ----------
        r : int, default=10, optional
            Ratio between the number of points at which the basis functions are
            evaluated and the number of points at which the basis functions
            were fitted. Only relevant for spline models.
            Default is 10.

        Returns
        -------
        ndarray
            Smooth reconstructed 24-hour activity profile.
        """
        # Make sure the model has been fitted before trying to reconstruct the curve
        if not self.beta:
            raise ValueError(
                'The basis function expansion parameters are empty. ',
                'Please run the `self.fit` method first.'
            )

        match self.basis:
            # Fourier
            case 'fourier':
                # Stack the components that describe the daily rhythm (one per column)
                x = np.stack(self.basis_functions, axis=1)

                # Combine the components using their fitted weights to rebuild the signal
                return np.dot(x, self.beta['beta'])
            case 'spline':
                # Create a time axis covering one full 24-hour cycle
                t = np.linspace(0, self.nsamples, r * self.nsamples, endpoint=True)

                # Evaluate the smooth curve at those time points
                return BSpline(*self.beta['beta'], extrapolate=False)(t)
            case _:
                raise ValueError('self.basis must be "fourier" or "spline"')


    def smooth_daily_profile(self, data, method='scott', verbose=False):
        """
        Smooth the average 24-hour activity profile using a Gaussian filter.

        The actigraphy signal is first converted into an average daily pattern.
        A Gaussian kernel is then applied to reduce short-term fluctuations
        while preserving the overall circadian structure.

        Because circadian data are cyclic (24h repeats), smoothing is performed
        in a circular manner so that midnight connects smoothly to the next day.

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Actigraphy data to fit.

        method : str, float.
            Method to calculate the width of the gaussian kernel.
            Available methods are `scott`, `silverman`. A numeric
            value can also be provided to set the width.
            Default is `scott`.

        verbose : bool.
            If True, print the kernel size used to smooth the data.
            Default is False.

        Returns
        -------
        ndarray
            Smoothed 24-hour activity profile.
        """
        # Convert the full signal into an average 24-hour pattern
        daily_avg = daily_profile(data)

        # Calculate the optimal kernel size
        bw = self._get_kernel_size(daily_avg.values, method=method)

        if verbose:
            print(f'Kernel size used to smooth the data: {bw}')

        # Apply circular Gaussian smoothing so the 24-h cycle wraps around
        return gaussian_filter1d(daily_avg, sigma=bw, mode='wrap')


    def smooth_timeseries(self, data, method='scott', verbose=False, mode='reflect'):
        """
        Smooth the full actigraphy time series using a Gaussian filter.

        This reduces short-term fluctuations while keeping the original timeline.
        Use this when you want a denoised version of the entire recording (e.g.,
        using mathematical models of circadian rhythms).

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Actigraphy data to fit.

        method : str, float.
            Method to calculate the width of the gaussian kernel.
            Available methods are `scott`, `silverman`. A numeric
            value can also be provided to set the width.
            Default is `scott`.

        verbose : bool.
            If True, print the kernel size used to smooth the data.
            Default is False.

        mode : str, default='reflect'
            Boundary handling for the filter.

        Returns
        -------
        np.ndarray
            Smoothed full time series (same length as the input data).
        """
        # Calculate the optimal kernel size
        bw = self._get_kernel_size(data.values, method=method)

        if verbose:
            print(f'Kernel size used to smooth the data: {bw}')

        # Smooth continuously without wrapping the end of the recording to the start
        return gaussian_filter1d(np.asarray(data), sigma=bw, mode=mode)