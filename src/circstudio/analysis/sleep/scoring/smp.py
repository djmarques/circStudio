import numpy as np
import pandas as pd


def minutes_since_midnight(index, scale=2*np.pi, norm=1440.):
    """
    Convert a time-of-day index into minutes since midnight. This function maps a (hour, minute, second)
    tuple to minutes since midnight and rescales it to a circular range (default: radians over 24 hours).

    Parameters
    ----------
    index : tuple of int
        Time-of-day tuple in the form (hour, minute, second).
    scale : float, optional
        Scaling factor applied to the output. Default is ``2π``, mapping the
        24-hour day onto the unit circle.
    norm : float, optional
        Normalization constant corresponding to the full cycle duration.
        Default is ``1440`` minutes (24 hours).

    Returns
    -------
    float
        Time of day expressed as a scaled circular quantity (e.g., radians).

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

    # Input tuple represents (hour,minute,second)

    # Minutes since midnight
    msm = 60.0*index[0]+index[1]+index[2]/60.0

    return msm*scale/norm


def sum_of_sine(msm, scores):
    """
    Compute the weighted sum of sine values for a given circular time.

    Parameters
    ----------
    msm : float
        Minutes since midnight (msm), expressed in radians.
    scores : array-like
        Values observed at this time-of-day across multiple days.

    Returns
    -------
    float
        Sum of ``scores * sin(msm)``.

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

    return np.sum(scores*np.sin(msm))


def sum_of_cosine(msm, scores):
    """
    Compute the weighted sum of cosine values for a given circular time.

    Parameters
    ----------
    msm : float
        Minutes since midnight (msm), expressed in radians.
    scores : array-like
        Values observed at this time-of-day across multiple days.

    Returns
    -------
    float
        Sum of ``scores * cos(msm)``.

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

    return np.sum(scores*np.cos(msm))


def sum_over_time_of_day(data):
    """
    Compute the sum of the sum of sines (sos) and cosines (soc) of the data across days.

    The data are grouped by clock time (hour, minute, seconds) across all days.
    For each time-of-day bin, sine and cosine sums are computed and then summed over the day.

    Parameters
    ----------
    data : pandas.Series
        Time-indexed data (e.g., activity) with a ``DatetimeIndex``.

    Returns
    -------
    ssos : float
        Sum of sines across all times of day.
    ssoc : float
        Sum of cosines across all times of day.

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
    grouped_data = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ])

    # Sum of sine
    sos = np.zeros(len(grouped_data.indices))
    # Sum of cosine
    soc = np.zeros_like(sos)

    # For each time of day
    for idx, (k, v) in enumerate(grouped_data.indices.items()):

        # Minutes since midnight
        msm = minutes_since_midnight(k)

        sos[idx] = sum_of_sine(msm, data.iloc[v])
        soc[idx] = sum_of_cosine(msm, data.iloc[v])

    return np.sum(sos), np.sum(soc)


def sleep_midpoint(data, threshold=None):
    """
    Calculate the sleep midpoint in minutes since midnight.

    The sleep midpoint is defined as the time halfway between sleep onset and wake-up time.

    Notes
    -----
    Sleep midpoint (SMP) is an index of sleep timing and is calculated as the following [1]:

    .. math::

        SMP = \frac{1440}{2\pi} arctan2\left(
              \sum_{j=1}^M\sum_{i=1}^N
              s_{i,j} \times sin\left(\frac{2\pi t_i}{1440}\right),
              \sum_{j=1}^M\sum_{i=1}^N
              s_{i,j} \times cos\left(\frac{2\pi t_i}{1440}\right)
              \right)

    with:
        * :math:`t_j`, time of day in minutes at epoch j,
        * :math:`\delta(s_{i,j}, s_{i+1,j}) = 1` if
          :math:`s_{i,j} = s_{i+1,j}` and 0 otherwise.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Lunsford-Avery, J. R., Engelhard, M. M., Navar, A. M., & Kollins, S. H. (2018). Validation of the Sleep Regularity
    Index in Older Adults and Associations with Cardiometabolic Risk. Scientific Reports, 8(1), 14158.
    https://doi.org/10.1038/s41598-018-32402-5

    [2] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [3] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """

    # Construct binarized data if requested
    data = pd.Series(
        np.where(data > threshold, 1, 0),
        index=data.index
    ) if threshold is not None else data

    # Calculate the sum of sums of sine/cosine
    ssos, ssoc = sum_over_time_of_day(data)

    # Compute sleep midpoint (smp) and return it
    smp = 1440/(2*np.pi)*np.arctan2(ssos, ssoc)
    return smp
