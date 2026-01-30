import numpy as np
import pandas as pd


def prob_stability(ts, threshold):
    """
    Compute the probability that any two consecutive time points are in the same state (wake or sleep).

    The time series (ts) is optionally binarized using a threshold, after which stability
    is defined as the proportion of adjacent samples sharing the same state

    Parameters
    ----------
    ts : array-like
        Time-ordered values (e.g., activity counts or sleep probability).
    threshold : float
        Threshold used to binarize the data. Values greater than the threshold
        are assigned to state 1, others to state 0.

    Returns
    -------
    float
        Probability that two consecutive samples are in the same state,
        ranging from 0 to 1.

    Notes
    -----
    Stability is computed as:

    ``δ(s_i, s_{i+1}) = 1`` if ``s_i = s_{i+1}``, else ``0``

    The probability is then the mean of δ across the time series.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """

    # Construct binarized data if requested
    data = np.where(ts > threshold, 1, 0) if threshold is not None else ts

    # Compute stability as $\delta(s_i,s_{i+1}) = 1$ if $s_i = s_{i+}$
    # Two consecutive values are equal if the 1st order diff is equal to zero.
    # The 1st order diff is either +1 or -1 otherwise.
    prob = np.mean(1-np.abs(np.diff(data)))

    return prob


def sri_profile(data, threshold):
    """
    Compute daily profile of sleep regularity indices.

    For each clock time (hour, minute, second), this function evaluates the
    probability that the behavioral state at that time is the same on
    consecutive days.

    Parameters
    ----------
    data : pandas.Series
        Time-indexed data with a ``DatetimeIndex`` and constant sampling
        frequency.
    threshold : float
        Threshold used to binarize the data into sleep and wake states.

    Returns
    -------
    pandas.Series
        Sleep regularity profile indexed by time of day (as ``Timedelta``),
        with values ranging from 0 to 1.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """
    # Group data by hour/minute/second across all the days contained in the
    # recording
    data_grp = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ])
    # Apply prob_stability to each data group (i.e series of consecutive points
    # that are 24h apart for a given time of day)
    # Ignore any na values in the series of consecutive points
    # sri_prof = data_grp.apply(prob_stability, threshold=threshold)
    sri_prof = data_grp.apply(lambda x: prob_stability(x.dropna(), threshold=threshold))
    sri_prof.index = pd.timedelta_range(
        start='0 day',
        end='1 day',
        freq=data.index.freq,
        closed='left'
    )
    return sri_prof


def sri(data, threshold=None):
    """
    Compute sleep regularity index (SRI).

    The SRI quantifies the day-to-day consistency of sleep-wake patterns by measuring how often an individual
    is in the same state at the same clock time on consecutive days.

    Parameters
    ----------
    data : pandas.Series
        Time-indexed data with a ``DatetimeIndex`` and constant sampling
        frequency.
    threshold : float, optional
        Threshold used to binarize the data into sleep and wake states. If
        ``None``, the data are assumed to be already binarized.

    Returns
    -------
    float
        Sleep Regularity Index, ranging from −100 (maximally irregular) to
        +100 (perfectly regular).

    Notes
    -----
    The SRI is computed as:

    ``SRI = 200 × mean(P_same_state) − 100``

    where ``P_same_state`` is the probability of being in the same state at the
    same clock time on consecutive days.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """
    # Compute daily profile of sleep regularity indices
    p_same_state = sri_profile(data, threshold)

    # Calculate SRI coefficient and return it
    sri_coef = 200*np.mean(p_same_state.values)-100
    return sri_coef
