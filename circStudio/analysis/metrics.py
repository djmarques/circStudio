import pandas as pd
import numpy as np
import re
from .tools import *
from statistics import mean
import statsmodels.api as sm


__all__ = [
    "daily_profile",
    "adat",
    "adatp",
    "l5",
    "l5p",
    "m10",
    "m10p",
    "relative_amplitude",
    "relative_amplitude_by_period",
    "interdaily_stability",
    "interdaily_stability_per_period",
    "intradaily_variability",
    "intradaily_variability_per_period"
]


def daily_profile(data, freq="5min", cyclic=False, time_origin=None, whs="1h"):
    r"""Average daily activity/light/temperature distribution

    Calculate the daily profile of activity. Data are averaged over all the
    days.

    Parameters
    ----------
    freq: str, optional
        Data resampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
    cyclic: bool, optional
        If set to True, two daily profiles are concatenated to ensure
        continuity between the last point of the day and the first one.
        Default is False.
    time_origin: str or pd.Timedelta, optional
        If not None, origin of the time axis for the daily profile.
        Original time bins are translated as time delta with respect to
        this new origin.
        Default is None
        Supported time string: 'AonT', 'AoffT', any 'HH:MM:SS'
    whs: str, optional
        Window half size parameter for the detection of the activity
        onset/offset time. Relevant only if time_origin is set to
        'AonT' or AoffT'.
        Default is '1h'.

    Returns
    -------
    raw : pandas.Series
        A Series containing the daily activity profile with a 24h index.
    """
    if time_origin is None:
        return _average_daily_activity(data, cyclic=cyclic)

    else:
        if cyclic is True:
            raise NotImplementedError(
                "Setting a time origin while cyclic option is True is not "
                "implemented."
            )

        avgdaily = _average_daily_activity(data, cyclic=False)

        if isinstance(time_origin, str):
            # Regex pattern for HH:MM:SS time string
            pattern = re.compile(r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$")

            if time_origin == "AonT":
                # Convert width half size from Timedelta to a nr of points
                whs = int(pd.Timedelta(whs) / data.index.freq)
                time_origin = _activity_onset_time(avgdaily, whs=whs)

            elif time_origin == "AoffT":
                # Convert width half size from Timedelta to a nr of points
                whs = int(pd.Timedelta(whs) / data.index.freq)
                time_origin = _activity_offset_time(avgdaily, whs=whs)

            elif pattern.match(time_origin):
                time_origin = pd.Timedelta(time_origin)

            else:
                raise ValueError(
                    "Time origin format ({}) not supported.\n".format(time_origin)
                    + "Supported format: {}.".format("HH:MM:SS")
                )

        elif not isinstance(time_origin, pd.Timedelta):
            raise ValueError(
                "Time origin is neither a time string with a supported"
                "format, nor a pd.Timedelta."
            )

        # Round time origin to the required frequency
        time_origin = time_origin.round(data.index.freq)

        shift = int((pd.Timedelta("12h") - time_origin) / data.index.freq)

        return _shift_time_axis(avgdaily, shift)


def adat(data, freq="10min", rescale=True, exclude_ends=False):
    """Total average daily activity

    Calculate the total activity counts, averaged over all the days.

    Parameters
    ----------
    rescale: bool, optional
        If set to True, the activity counts are rescaled to account for
        masked periods (if any).
        Default is True.
    exclude_ends: bool, optional
        If set to True, the first and last daily periods are excluded from
        the calculation. Useful when the recording does start or end at
        midnigth.
        Default is False.

    Returns
    -------
    adat : int
    """
    return _average_daily_total_activity(data, rescale=rescale, exclude_ends=exclude_ends)


def adatp(data, period="7D", rescale=True, exclude_ends=False, verbose=False):
    """Total average daily activity per period

    Calculate the total activity counts, averaged over each consecutive
    period contained in the data. The number of periods

    Parameters
    ----------
    period: str, optional
        Time length of the period to be considered. Must be understandable
        by pandas.Timedelta
    rescale: bool, optional
        If set to True, the activity counts are rescaled to account for
        masked periods (if any).
        Default is True.
    exclude_ends: bool, optional
        If set to True, the first and last daily periods are excluded from
        the calculation. Useful when the recording does start or end at
        midnigth.
        Default is False.
    verbose: bool, optional
        If set to True, display the number of periods found in the data.
        Also display the time not accounted for.
        Default is False.

    Returns
    -------
    adatp : list of int
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = [
        _average_daily_total_activity(
            data[time[0] : time[1]], rescale=rescale, exclude_ends=exclude_ends
        )
        for time in intervals
    ]

    return results


def l5(data):
    r"""L5

    Mean activity/temperature/light, etc., during the 5 least active hours of the day.

    Returns
    -------
    l5: float

    Notes
    -----

    The L5 [1]_ variable is calculated as the mean, per acquisition period,
    of the average daily activities during the 5 least active hours.

    .. warning:: The value of this variable depends on the length of the
    acquisition period.

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
            (1997). Long-Term Fitness Training Improves the Circadian
            Rest-Activity Rhythm in Healthy Elderly Males.
            Journal of Biological Rhythms, 12(2), 146–156.
            http://doi.org/10.1177/074873049701200206
    """

    l5_onset, l5 = _lmx(data, "5h", lowest=True)
    return l5_onset, l5


def m10(data):
    r"""M10

    Mean activity/light/temperature, etc. during the 10 most active hours of the day.

    Returns
    -------
    m10: float

    Notes
    -----

    The M10 [1]_ variable is calculated as the mean, per acquisition period
    , of the average daily activities during the 10 most active hours.

    .. warning:: The value of this variable depends on the length of the
                 acquisition period.

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
           (1997). Long-Term Fitness Training Improves the Circadian
           Rest-Activity Rhythm in Healthy Elderly Males.
           Journal of Biological Rhythms, 12(2), 146–156.
           http://doi.org/10.1177/074873049701200206
    """
    m10_onset, m10 = _lmx(data, "10h", lowest=False)
    return m10_onset, m10


def relative_amplitude(data):
    r"""Relative rest/activity amplitude

    Relative amplitude between the mean activity during the 10 most active
    hours of the day and the mean activity during the 5 least active hours
    of the day.

    Returns
    -------
    ra: float

    Notes
    -----

    The RA [1]_ variable is calculated as:

    .. math::

        RA = \frac{M10 - L5}{M10 + L5}

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
            (1997). Long-Term Fitness Training Improves the Circadian
            Rest-Activity Rhythm in Healthy Elderly Males.
            Journal of Biological Rhythms, 12(2), 146–156.
            http://doi.org/10.1177/074873049701200206
    """
    _, l5 = _lmx(data, "5h", lowest=True)
    _, m10 = _lmx(data, "10h", lowest=False)
    return (m10 - l5) / (m10 + l5)


def l5p(data, period="7D", verbose=False):
    r"""L5 per period

    The L5 variable is calculated for each consecutive period found in the
    actigraphy recording.

    Parameters
    ----------
    period: str, optional
        Time period for the calculation of IS
        Default is '7D'.
    verbose: bool, optional
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.

    Returns
    -------
    l5p: list of float


    Notes
    -----

    The L5 [1]_ variable is calculated as the mean, per acquisition period,
    of the average daily activities during the 5 least active hours.

    .. warning:: The value of this variable depends on the length of the
                  acquisition period.

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
            (1997). Long-Term Fitness Training Improves the Circadian
            Rest-Activity Rhythm in Healthy Elderly Males.
            Journal of Biological Rhythms, 12(2), 146–156.
            http://doi.org/10.1177/074873049701200206
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = [_lmx(data[time[0] : time[1]], "5h", lowest=True) for time in intervals]
    return [res[1] for res in results]


def m10p(data, period="7D", verbose=False):
    r"""M10 per period

    The M10 variable is calculated for each consecutive period found in the
    actigraphy recording.

    Parameters
    ----------
    period: str, optional
        Time period for the calculation of IS
        Default is '7D'.
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.
    verbose: bool, optional
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.

    Returns
    -------
    m10p: list of float


    Notes
    -----

    The M10 [1]_ variable is calculated as the mean, per acquisition period
    , of the average daily activities during the 10 most active hours.

    .. warning:: The value of this variable depends on the length of the
                     acquisition period.

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
            (1997). Long-Term Fitness Training Improves the Circadian
            Rest-Activity Rhythm in Healthy Elderly Males.
            Journal of Biological Rhythms, 12(2), 146–156.
            http://doi.org/10.1177/074873049701200206
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = [_lmx(data[time[0] : time[1]], "10h", lowest=False) for time in intervals]
    return [res[1] for res in results]


def relative_amplitude_by_period(data, period="7D", verbose=False):
    r"""RA per period

    The RA variable is calculated for each consecutive period found in the
    actigraphy recording.

    Parameters
    ----------
    period: str, optional
        Time period for the calculation of IS
        Default is '7D'.
    verbose: bool, optional
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.

    Returns
    -------
    rap: list of float

    Notes
    -----

    The RA [1]_ variable is calculated as:

    .. math::

        RA = \frac{M10 - L5}{M10 + L5}

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
           (1997). Long-Term Fitness Training Improves the Circadian
           Rest-Activity Rhythm in Healthy Elderly Males.
           Journal of Biological Rhythms, 12(2), 146–156.
           http://doi.org/10.1177/074873049701200206
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = []

    for time in intervals:
        data_subset = data[time[0] : time[1]]
        _, l5 = _lmx(data_subset, "5h", lowest=True)
        _, m10 = _lmx(data_subset, "10h", lowest=False)
        results.append((m10 - l5) / (m10 + l5))

    return results


def interdaily_stability(data):
    r"""Interdaily stability

    The Interdaily stability (IS) quantifies the repeatibilty of the
    daily rest-activity pattern over each day contained in the activity
    recording.

    Parameters
    ----------
    freq: str, optional
        Data resampling `frequency string
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
        Default is '1h'.

    Returns
    -------
    is: float

    Notes
    -----

    This variable is defined in ref [1]_:

    .. math::

        IS = \frac{d^{24h}}{d^{1h}}

    with:

    .. math::

        d^{1h} = \sum_{i}^{n}\frac{\left(x_{i}-\bar{x}\right)^{2}}{n}

    where :math:`x_{i}` is the number of active (counts higher than a
    predefined threshold) minutes during the :math:`i^{th}` period,
    :math:`\bar{x}` is the mean of all data and :math:`n` is the number of
    periods covered by the actigraphy data and with:

    .. math::

        d^{24h} = \sum_{i}^{p} \frac{
                    \left( \bar{x}_{h,i} - \bar{x} \right)^{2}
                    }{p}

    where :math:`\bar{x}^{h,i}` is the average number of active minutes
    over the :math:`i^{th}` period and :math:`p` is the number of periods
    per day. The average runs over all the days.

    For the record, tt is the 24h value from the chi-square periodogram
    (Sokolove and Bushel1 1978).

    References
    ----------

    .. [1] Witting W., Kwa I.H., Eikelenboom P., Mirmiran M., Swaab D.F.
            Alterations in the circadian rest–activity rhythm in aging and
            Alzheimer׳s disease. Biol Psychiatry. 1990;27:563–572.
    """
    d_24h = (
        data.groupby([data.index.hour, data.index.minute, data.index.second])
        .mean()
        .var()
    )

    d_1h = data.var()

    return d_24h / d_1h


def interdaily_stability_per_period(data, period="7D", verbose=False):
    r"""Interdaily stability per period

    The IS is calculated for each consecutive period found in the
    actigraphy recording.

    Parameters
    ----------
    period: str, optional
        Time period for the calculation of IS
        Default is '7D'.
    freq: str, optional
        Data resampling `frequency strings
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
    verbose: bool, optional
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.

    Returns
    -------
    isp: list of float

    Notes
    -----

    Periods are consecutive and all of the required duration. If the last
    consecutive period is shorter than required, the IS is not calculated
    for that period.
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = [interdaily_stability(data[time[0] : time[1]]) for time in intervals]
    return results


def intradaily_variability(data):
    r"""Intradaily variability

    The Intradaily Variability (IV) quantifies the variability of the
    activity recording. This variable thus measures the rest or activity
    fragmentation.

    Parameters
    ----------
    freq: str, optional
        Data resampling `frequency string
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
        Default is '1h'.

    Returns
    -------
    iv: float

    Notes
    -----

    It is defined in ref [1]_:

    .. math::

        IV = \frac{c^{1h}}{d^{1h}}

    with:

    .. math::

        d^{1h} = \sum_{i}^{n}\frac{\left(x_{i}-\bar{x}\right)^{2}}{n}

    where :math:`x_{i}` is the number of active (counts higher than a
    predefined threshold) minutes during the :math:`i^{th}` period,
    :math:`\bar{x}` is the mean of all data and :math:`n` is the number of
    periods covered by the actigraphy data,

    and with:

    .. math::

        c^{1h} = \sum_{i}^{n-1} \frac{
                    \left( x_{i+1} - x_{i} \right)^{2}
                    }{n-1}

    References
    ----------

    .. [1] Witting W., Kwa I.H., Eikelenboom P., Mirmiran M., Swaab D.F.
            Alterations in the circadian rest–activity rhythm in aging and
            Alzheimer׳s disease. Biol Psychiatry. 1990;27:563–572.
    """
    c_1h = data.diff(1).pow(2).mean()

    d_1h = data.var()

    return c_1h / d_1h


def intradaily_variability_per_period(data, period="7D", verbose=False):
    r"""Intradaily variability per period

    The IV is calculated for each consecutive period found in the
    actigraphy recording.

    Parameters
    ----------
    period: str, optional
        Time period for the calculation of IS
        Default is '7D'.
    freq: str, optional
        Data resampling `frequency string
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
        Default is '1h'.
    verbose: bool, optional
        If set to True, display the number of periods found in the activity
        recording, as well as the time not accounted for.
        Default is False.

    Returns
    -------
    ivp: list of float

    Notes
    -----

    Periods are consecutive and all of the required duration. If the last
    consecutive period is shorter than required, the IV is not calculated
    for that period.
    """
    intervals = _interval_maker(data.index, period, verbose)

    results = [
        intradaily_variability(data[time[0] : time[1]]) for time in intervals
    ]
    return results