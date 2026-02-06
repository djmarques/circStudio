import pandas as pd
import numpy as np
from numpy.polynomial import polynomial
from scipy.stats import linregress


class Fractal:
    """
    (Multifractal) detrended fluctuation analysis - (MF)DFA.

    This analysis quantifies the scale-invariant (fractal) organization of
    temporal fluctuations in time series. In simple terms, it addresses the
    question:

        * How similar are activity fluctuations across different time scales?*

    Informally, this (MF)DFA removes the average (trend) of the signal to assess how
    it deviates from it.

    In the context of actigraphy, (MF)DFA analysis determines whether the temporal
    structure of activity variations observed over short time scales (e.g., minutes)
    resembles that observed over longer time scales (e.g. hours), and whether this
    temporal organization changes with aging, disease, or circadian disruption.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L. (1994). Mosaic
    organization of DNA nucleotides. Physical Review E, 49(2), 1685–1689. https://doi.org/10.1103/PhysRevE.49.1685

    [2] Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S., Bunde, A., & Stanley, H. E. (2002).
    Multifractal detrended fluctuation analysis of nonstationary time series. Physica A: Statistical Mechanics and
    Its Applications, 316(1–4), 87–114. https://doi.org/10.1016/S0378-4371(02)01383-3

    [3] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [4] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """

    def __init__(self, n_array=None, q_array=None):

        self.__n_array = n_array
        self.__q_array = q_array

    # ----------------------------
    # Internal helper functions
    # ----------------------------
    @staticmethod
    def profile(x: np.ndarray) -> np.ndarray:
        """
        Create a cumulative profile of a time series after removing its average.

        This is the first step of (MF)DFA. The idea is to measure how the signal
        deviates from its overall average. We first calculate the global mean of
        the signal (its trend) and then subtract it from each value in the time
        series. Finally, we cumulatively sum the deviations to create a (MF)DFA
        profile that emphasizes fluctuations over time.

        Parameters
        ----------
        x: numpy.array
            Input time series.

        Returns
        -------
        profile : numpy.array
            Mean-centered cumulative sum of the input signal.
        """
        # Compute the global mean (or trend) of the signal (e.g., activity), ignoring NaN values
        trend = np.nanmean(x)

        # Subtract the mean from the signal and return cumulative sum, ignoring NaN values
        return np.nancumsum(x - trend)

    @staticmethod
    def _rolling_window(x: np.ndarray, n: int) -> np.ndarray:
        """
        Generate overlapping windows from a one-dimensional array.

        This function creates all consecutive subarrays (windows) of length
        ``n`` from the input signal. Each window is shifted at a time. These
        windows are useful for identifying local patterns within the signal.

        Parameters
        ----------
        x : numpy.array
            Input time series.
        n : int
            Length of each window.

        Returns
        -------
        roll : numpy.array
            2D array where each row is a consecutive window of the signal.
        """
        shape = x.shape[:-1] + (x.shape[-1] - n + 1, n)
        strides = x.strides + (x.strides[-1],)
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    @classmethod
    def segmentation(
        cls, x: np.ndarray, n: int, backward: bool = False, overlap: bool = False
    ) -> np.ndarray:
        """
        Split a signal into smaller segments for local analysis.

        This function divides a time series into consecutive segments of length ``n``.
        Segmentation can start from the beginning or end of the signal, and windows
        can overlap to capture more fine-grained fluctuations. Each segment can be
        analyzed independently to measure local trends and variability.

        Parameters
        ----------
        x  : numpy.array
            Input time series (typically, the DFA profile).
        n : int
            Window size.
        backward: bool, optional.
            If True, start segmentation from the end of the series.
            Default is False.
        overlap: bool, optional
            If True, consecutive segments overlap by 50% to improve resolution.
            Default is False.

        Returns
        -------
        segments : numpy.array
            2D array where each row is one segment of the signal.
        """
        # Compute consecutive windows, shifted by one element
        windows = cls.__class__._rolling_window(x, n)

        # Compute distance between the center of two consecutive windows
        stride = n // 2 if overlap else n

        # Non-overlapping segments
        if backward:
            segments = windows[::stride]
        else:
            segments = windows[::-stride]

        return segments

    @classmethod
    def local_msq_residuals(cls, segment, deg):
        """
        Measure how much a segment of the signal fluctuates around its local mean
        (local trend).

        This function fits a smooth curve (a polynomial of degree ``deg``) to a
        segment of the time series and calculates how much the actual data points
        deviate from this curve. In other words, it tells you how bumpy or irregular
        the signal is in that segment after removing the overall trend.

        Parameters
        ----------
        segment : numpy.array
            A short portion of the time series to analyze.
        deg : int
            Degree of the polynomial used to fit the trend:
            - 1 = straight line (linear trend)
            - 2 = parabola (quadratic trend)
            - etc.

        Returns
        -------
        float
            Mean squared deviation of the data from the fitted trend.
            Higher values indicate more fluctuations.
        """
        # Segment length
        n = len(segment)

        # X-axis (time axis)
        x = np.linspace(1, n, n)

        # Fit polynomial of order deg to the segment
        # deg = 1 <-> linear DFA
        # deg = 2 <-> quadratic detrending
        _, fit_result = polynomial.polyfit(
            y=segment,
            x=x,
            deg=deg,
            full=True
        )

        # Return mean squared residuals
        return fit_result[0][0] / n if fit_result[0].size != 0 else np.nan

    # ----------------------------
    # Fluctuation calculations
    # ----------------------------

    @classmethod
    def fluctuations(cls, X, n, deg, overlap=False):
        r"""Fluctuation function

        The fluctuations are defined as the mean squared residuals
        of the least squares fit in each non-overlapping window.

        Parameters
        ----------
        X: numpy.array
            Array of activity counts.
        n: int
            Window size.
        deg: int
            Degree(s) of the fitting polynomials.
        overlap: bool, optional
            If set to True, consecutive windows during segmentation
            overlap by 50%. Default is False.

        Returns
        -------
        F: numpy.array
            Array of mean squared residuals in each segment.
        """

        # Detrend and integrate time series
        Y = cls.profile(X)

        # Define non-overlapping segments
        segments_fwd = cls.segmentation(Y, n, backward=False, overlap=overlap)
        segments_bwd = cls.segmentation(Y, n, backward=True, overlap=overlap)

        # Assert equal numbers of segments
        assert segments_fwd.shape == segments_bwd.shape

        F = np.empty(len(segments_fwd) + len(segments_bwd))
        # Compute the sum of the squared residuals for each segment
        for i in range(len(segments_fwd)):
            F[i * 2] = cls.local_msq_residuals(segments_fwd[i], deg)
            F[i * 2 + 1] = cls.local_msq_residuals(segments_bwd[i], deg)

        return F

    @classmethod
    def q_th_order_mean_square(cls, F, q):
        r"""Qth-order mean square function

        Compute the q-th order mean squares.

        Parameters
        ----------
        F: numpy.array
            Array of fluctuations.
        q: scalar
            Order.

        Returns
        -------
        qth_msq: numpy.float
            Q-th order mean square.
        """

        if q == 0:
            qth_msq = np.exp(0.5 * np.nanmean(np.log(F)))
        else:
            qth_msq = np.power(np.nanmean(np.power(F, q / 2)), 1 / q)

        return qth_msq

    # ----------------------------
    # DFA and MF-DFA
    # ----------------------------
    @classmethod
    def dfa(cls, ts, n_array, deg=1, overlap=False, log=False):
        r"""
        Perform Detrended Fluctuation Analysis (DFA) on a time series (ts).

        DFA measures how much a signal fluctuates around its local trends
        across different timescales. This can help identify patterns in
        activity or other time-dependent signals (e.g., light exposure,
        body temperature).

        The method works by dividing the signal into windows of increasing
        size, removing slow trends within each window, and measuring the
        remaining fluctuations. The resulting fluctuation amplitudes
        describe how variability changes as a function of timescale.

        Note
        ____
        DFA corresponds to a special case of Multifractal DFA (MF-DFA) in
        which a single q-value is used (q = 2). The q-value controls how
        fluctuations of different magnitudes contribute to the fluctuation
        measure. When q = 2, small and large fluctuations are combined into
        a single, average measure of variability at each timescale.

        Parameters
        ----------
        ts: pandas.Series
            Input time series (e.g., activity counts).
        n_array: array of int
            Window sizes (time scales) over which to measure
            fluctuations (in minutes).
        deg: int, optional
            Degree of the polynomial used for detrending each window.
            Default is 1 (linear detrending).
        overlap: bool, optional
            If True, consecutive windows during segmentation overlap by 50%.
            Default is False.
        log: bool, optional
            If True, return log-transformed fluctuations.
            Default is False.

        Returns
        -------
        numpy.array
            Array of fluctuation amplitudes for each window size.
        """
        # Ensure the time series is evenly sampled.
        # DFA requires a fixed sampling interval because window sizes
        # are defined in number of samples

        freq = ts.index.freq

        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency.\n"
                "Ensure the time series has a fixed interval or use pandas.Series.resample."
            )
        else:
            # Convert window sizes from minutes to number of samples
            factor = pd.Timedelta("1min") / freq

        # Compute mean squared fluctuations (q = 2) for each timescale/window size
        # DFA is equivalent to MF-DFA evaluated at q = 2.
        iterable = (
            cls.q_th_order_mean_square(
                cls.fluctuations(
                    ts.values,
                    n=int(n),
                    deg=deg,
                    overlap=overlap),
                q=2
            )
            for n in factor * n_array
        )

        q_th_order_msq_fluctuations = np.fromiter(
            iterable,
            dtype="float",
            count=len(n_array)
        )

        # Optionally log transform fluctuation amplitudes
        if log:
            q_th_order_msq_fluctuations = np.log(q_th_order_msq_fluctuations)

        return q_th_order_msq_fluctuations

    @classmethod
    def mfdfa(cls, ts, n_array, q_array, deg=1, overlap=False, log=False):
        r"""
        Perform Multifractal Detrended Fluctuation Analysis (MF-DFA).

        MF-DFA is an extension of standard Detrended Fluctuation Analysis (DFA).
        In standard DFA, a single fluctuation measure is computed using a fixed
        q-value (q = 2). This corresponds to describing the average behavior of
        fluctuations, without distinguishing between small and large deviations
        from the local trend.

        MF-DFA generalizes this approach by repeating the analysis for multiple
        q-values. The q-order controls how strongly different fluctuation magnitudes
        influence the result:

            - Positive q-values give more weight to segments with large fluctuations,
            - Negative q-values give more weight to segments with small fluctuations.

        Parameters
        ----------
        ts: pandas.Series
            Input time series (e.g., activity counts).
        n_array: array of int
            Window sizes (timescales) over which to measure fluctuations (in minutes).
        q_array : array of float
            Orders of the mean squares; different q emphasize small or large fluctuations.
        deg: int, optional
            Degree of the polynomial used for detrending each window.
            Default is 1 (linear detrending).
        overlap: bool, optional
            If True, consecutive windows during segmentation overlap by 50%.
            Default is False.
        log: bool, optional
            If True, return log-transformed fluctuations.
            Default is False.

        Returns
        -------
        numpy.array
            2D array (timescales × q-orders) containing fluctuation amplitudes.
        """
        # Ensure the time series is evenly sampled.
        # MF-DFA requires a fixed sampling interval because window sizes
        # are defined in number of samples
        freq = ts.index.freq

        if freq is None:
            raise ValueError(
                "Cannot check the sampling frequency.\n"
                "Ensure the time series has a fixed interval or use pandas.Series.resample."
            )
        else:
            # Convert window sizes from minutes to number of samples
            factor = pd.Timedelta("1min") / freq

        # Output array, where rows store different timescales
        # and different columns different q-values
        q_th_order_msq_fluctuations = np.empty(
            (len(n_array), len(q_array)), dtype="float"
        )

        # Loop over timescales (n_array)
        for idx, n in enumerate(factor * n_array):
            # Compute detrended fluctuation amplitudes at this timescale
            fluctuation = cls.fluctuations(
                ts.values,
                n=int(n),
                deg=deg,
                overlap=overlap
            )

            # Compute fluctuation measures for all q-values at this timescale
            q_th_order_msq_fluctuations[idx] = [
                cls.q_th_order_mean_square(fluctuation, q=q)
                for q in q_array
            ]

        # Optionally log-transform fluctuation amplitudes
        if log:
            q_th_order_msq_fluctuations = np.log(q_th_order_msq_fluctuations)

        # Return q-th order fluctuations
        return q_th_order_msq_fluctuations

    # ----------------------------
    # Hurst and scaling analysis
    # ----------------------------
    @classmethod
    def generalized_hurst_exponent(cls, F_n, n_array, n_center=False, log=False):
        r"""
        Estimate the (generalized) Hurst exponent from DFA/MF-DFA fluctuations.

        In DFA/MF-DFA, we compute a fluctuation amplitude `F(n)` at multiple
        timescales (window sizes) `n`. If the signal exhibits scale-invariant
        structure over the chosen range of scales (window sizes), these quantities
        follow an approximate power-law relationship:

            F(n) ≈ n ** h

        Taking logarithms turns this into a straight-line relationship:

            log(F(n)) ≈ log (n ** h)
            log(F(n)) ≈ h * log(n)

        Intuitively, the Hurst exponent can be thought of as a measure of long-term
        memory in a time series. It describes how strongly fluctuations at one timescale
        resemble fluctuations at other scales, and it is related to the presence of
        long-range correlations in the signal.

        In standard DFA, this slope is commonly referred to as the Hurst exponent `H`
        In MF-DFA, repeating this estimation for different q-orders yields the generalized
        Hurst exponent `h(q)`.

        Parameters
        ----------
        F_n : array
            Array of fluctuation amplitudes (e.g., ``F(n)``)
        n_array: array of int
            Timescales (i.e., window sizes) at which fluctuations were computed,
            expressed in minutes.
        n_center: bool, optional
            If True, the logarithm of the timescales is mean-centered prior
            to regression.
            Default is false.
        log: bool, optional
            If True, assume that F_n and n_array have already been log-transformed.
            Default is False.

        Returns
        -------
        h : float
            Estimated (generalized) Hurst exponent.

        h_err : float
            Standard error of the slope estimate.
        """
        # Convert to logarithmic scale if not already
        # y
        log_fn = np.log(F_n) if not log else F_n
        # x
        log_n = np.log(n_array) if not log else n_array

        # Optionally, apply mean-centering for log(n):
        # slope stays the same, but the intercept shifts.
        if n_center:
            log_n = log_n - np.mean(log_n)

        # Fit log_fn = h*log_n; slope h corresponds to generalized Hurst exponent
        r = linregress(y=log_fn, x=log_n)

        # Return the Hurst exponent and the standard error
        return r.slope, r.stderr

    @classmethod
    def crossover_search(cls, F_n, n_array, n_min=3, log=False):
        """
        Search for potential crossovers in DFA/MF-DFA fluctuations.

        A crossover is defined as a change in scaling properties of the
        fluctuations with respect to timescales. In DFA/MF-DFA, fluctuation
        amplitudes :math:`F(n)` are expected to follow an approximate
        power-law relationship over the chosen range of scales (window sizes),
        :math:`n`. Sometimes, however, the scaling behavior changes at a given
        scale, that is, a crossover happens.

        This function performs a simple crossover search by trying many candidate
        points :math:`n_x`. For each candidate :math:`n_x`, it:

           1) Fits a Hurst exponent using scales below :math:`n_x`.
           2) Fits a Hurst exponent using scales above :math:`n_x`.
           3) Computes the ratio of the two exponents.

        If the ratio deviates strongly from 1, it suggests that scaling differs
        between small and large time scales, consistent with a crossover.

        Parameters
        ----------
        F_n : np.array
            Fluctuation amplitudes evaluated at each timescale in `n_array`
            (e.g., :math:`F(n)`).
        n_array: array of int
            Timescales :math:`n` (window sizes), in minutes.
        n_min: int, optional
            Minimal number of time scales required to estimate the generalized
            Hurst exponent.
            Default is 3.
        log: bool, optional
            If set to True, assume that the input values have already been
            log-transformed.
            Default is False.

        Returns
        -------
        h_ratios, h_ratios_err, n_x: arrays of floats
            Ratio of h(q), and associated uncertainties, obtained for various
            time scales n_x.

        Notes
        -----

        .. warning::

            The calculation of the uncertainty on the ratio of scaling
            exponents assumes uncorrelated variables:
            :math:`\sigma_{A/B}^2=(A/B)^2(\sigma_{A}^2/A^2+\sigma_{B}^2/B^2)`.
            Most likely, the scaling exponents calculated for time scales
            :math:`n<n_x` is not uncorrelated to the scaling exponents
            calculated for time scales :math:`n>n_x`. Therefore, the resulting
            uncertainty is either overestimated in case of positively
            correlated variables or underestimated otherwise. However, the
            magnitude of the calculated uncertainty provides a rough estimate.
        """

        n_x = np.empty(len(n_array) - 2 * n_min + 1)
        h_ratios = np.empty(len(n_array) - 2 * n_min + 1)
        h_ratios_err = np.empty(len(n_array) - 2 * n_min + 1)
        # If the number of points for a single linear fit is less than 3
        if n_min < 3:
            print(
                (
                    "Cannot perform a linear fit on series of less than"
                    " 3 points. Exiting now."
                )
            )
        # If the number of points to fit is less than 2*3
        elif (len(n_array) - 2 * n_min + 1) < 1:
            print(("Total number of points to fit is less than 2*3." "Exiting now."))
        else:
            for t in np.arange(n_min, len(n_array) - n_min + 1):
                # Fit the series of points (F(n) vs n) up to point n_x
                alpha_1, alpha_1_err = cls.generalized_hurst_exponent(
                    F_n[:t], n_array[:t], log
                )
                # Fit the series of points (F(n) vs n) from point n_x to n_max
                alpha_2, alpha_2_err = cls.generalized_hurst_exponent(
                    F_n[t:], n_array[t:], log
                )
                # Alpha ratio and relative uncertainties
                ratio = alpha_1 / alpha_2
                alpha_1_rel_err = alpha_1_err / alpha_1
                alpha_2_rel_err = alpha_2_err / alpha_2

                n_x[t - n_min] = n_array[t]
                h_ratios[t - n_min] = ratio
                h_ratios_err[t - n_min] = ratio * np.sqrt(
                    alpha_1_rel_err * alpha_1_rel_err
                    + alpha_2_rel_err * alpha_2_rel_err
                )

            if log:
                n_x = np.exp(n_x)

        return h_ratios, h_ratios_err, n_x

    @classmethod
    def local_slopes(cls, F_n, n_array, s=2, log=False, verbose=False):
        r"""
        Compute local Hurst exponents (local slopes) of log(F(n)) versus log(n).

        When `F(n)` follows a single power law across all scales, the
        log(F(n)) versus log(n) curve is approximately a straight line
        and the slope is constant.

        In real data, scaling can vary with scale. This function estimates a local
        slope around each scale by fitting a straight line (a polynomial of degree 1)
        to a small sliding window (min. 3)


        to a small sliding window of points in log(F(n)) versus log(n).




        The local slope of the curve log(F(n)) is calculated at each point by
        fitting polynomial of degree 1, using the 2*s surrounding points.
        At the boundaries of {F(n)}, the local slope is calculated with a
        reduced number of points (min. 3).

        Parameters
        ----------
        F_n : array
            Array of fluctuations.
        n_array: array of int
            Time scales (i.e window sizes). In minutes.
        s: int, optional
            Half size of the window used to estimate the local slope.
            The total window size is (2*s+1). Default is 2.
        log: bool, optional
            If set to True, assume that the input values have already been
            log-transformed.
            Default is False.
        verbose: bool, optional
            If set to True, display informations about the calculations.
            Default is False.

        Returns
        -------
        alpha_loc, alpha_loc_err, n_x: arrays of floats
            Local slopes (alpha_loc), and associated uncertainties, obtained
            for various time scales n_x.

        """
        # Check if inputs have the same dimension
        assert len(F_n) == len(n_array)

        # Window size: 2*s + 1
        win_size = 2 * s + 1
        # Check if the number of points for a single linear fit is at least 3
        if win_size < 3:
            raise ValueError(
                (
                    "Cannot perform a linear fit on series of less than"
                    " 3 points; `s` must be greater or equal to 1."
                )
            )
        # Check if the number of points to fit is less than 2*n_min
        if len(n_array) < win_size:
            raise ValueError(("Total number of points to fit is less than (2*s+1)."))

        n_x = np.empty(len(n_array) - 2 * s)
        alpha_loc = np.empty(len(n_array) - 2 * s)
        alpha_loc_err = np.empty(len(n_array) - 2 * s)

        for t in np.arange(0, len(n_array) - win_size + 1):
            # Fit the series of points (F(n) vs n) up to point n_x
            if verbose:
                print("-" * 50)
                print("Fit info:")
                print("- t: {}".format(t))
                print("- current point: {}".format(t + s))
                print("- F_n[t-s:t+s]: {}".format(F_n[t : t + win_size]))
                print("- n_array[t-s:t+s]: {}".format(n_array[t : t + win_size]))
                print("- array center: {}".format(n_array[t + s]))

            alpha_loc[t], alpha_loc_err[t] = cls.generalized_hurst_exponent(
                F_n[t : t + win_size], n_array[t : t + win_size], log
            )
            n_x[t] = n_array[t + s]

            if log:
                n_x = np.exp(n_x)

        return alpha_loc, alpha_loc_err, n_x

    @staticmethod
    def equally_spaced_logscale_range(n, start=1, stop=1440):
        """Equally spaced numbers in log-scale

        Construct a series of equally spaced numbers in log scale

        Parameters
        ----------
        n: int
            Time scales (i.e window sizes). In minutes.
        start: int, optional
            Starting number.
            Default is 1.
        stop: int, optional
            End number.
            Default is 1440.

        Returns
        -------
        n_array: numpy.array of int
            Array of equally spaced numbers.
        """

        # Create series of exponentiated numbers evenly spaced in log-space.
        units = np.geomspace(start, stop, num=n, endpoint=True, dtype=int)

        # Cast as int and remove duplicates
        n_array = np.unique(units)

        # Filter out numbers < start
        n_array = n_array[np.where(n_array >= start)]

        return n_array
