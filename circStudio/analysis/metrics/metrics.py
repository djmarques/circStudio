import pandas as pd
import numpy as np
import re
from circStudio.analysis.tools import *
from statistics import mean
import statsmodels.api as sm


def daily_profile(data, cyclic=False, time_origin=None, whs="1h"):
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


def daily_profile_auc(data, start_time=None, stop_time=None, time_origin=None):
    r"""AUC of the average daily light profile

    Calculate the area under the curve of the daily profile of light
    exposure. Data are averaged over all the days.

    Parameters
    ----------
    start_time: str, optional
        If not set to None, compute AUC from start time.
        Supported time string: 'HH:MM:SS'
        Default is None.
    stop_time: str, optional
        If not set to None, compute AUC until stop time.
        Supported time string: 'HH:MM:SS'
        Default is None.
    time_origin: str or pd.Timedelta, optional
        If not None, origin of the time axis for the daily profile.
        Original time bins are translated as time delta with respect to
        this new origin.
        Default is None
        Supported time string: 'HH:MM:SS'

    Returns
    -------
    auc : float
        Area under the curve.
    """
    # Compute average daily profile
    avgdaily = _average_daily_activity(data, cyclic=False)

    if time_origin is not None:

        if isinstance(time_origin, str):
            # Regex pattern for HH:MM:SS time string
            pattern = re.compile(r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$")

            if pattern.match(time_origin):
                time_origin = pd.Timedelta(time_origin)
            else:
                raise ValueError(
                    "Time origin format ({}) not supported.\n".format(time_origin)
                    + "Supported format: HH:MM:SS."
                )

        elif not isinstance(time_origin, pd.Timedelta):
            raise ValueError(
                "Time origin is neither a time string with a supported "
                "format, nor a pd.Timedelta."
            )

        # Round time origin to the required frequency
        time_origin = time_origin.round(data.index.freq)

        shift = int((pd.Timedelta("12h") - time_origin) / data.index.freq)

        avgdaily = _shift_time_axis(avgdaily, shift)

    # Restrict profile to start/stop times
    if start_time is not None:
        start_time = pd.Timedelta(start_time)
    if stop_time is not None:
        stop_time = pd.Timedelta(stop_time)

    # In order to avoid indexing with None, check for that too
    if start_time is not None or stop_time is not None:
        return avgdaily.loc[start_time:stop_time].sum()
    else:
        return avgdaily.sum()


def adat(data, rescale=True, exclude_ends=False):
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
    return _average_daily_total_activity(
        data, rescale=rescale, exclude_ends=exclude_ends
    )


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

    results = [intradaily_variability(data[time[0] : time[1]]) for time in intervals]
    return results


def summary_statistics_per_time_bin(light, bins="24h", agg_func=None):
    r"""Summary statistics.

    Calculate summary statistics (ex: mean, median, etc) according to a
    user-defined (regular or arbitrary) binning.

    Parameters
    ----------
    bins: str or list of tuples, optional
        If set to a string, bins is used to define a regular binning where
        every bin is of length "bins". Ex: "2h".
        Otherwise, the list of 2-tuples is used to define an arbitrary
        binning. Ex: \[('2000-01-01 00:00:00','2000-01-01 11:59:00')\].
        Default is '24h'.
    agg_func: list, optional
        List of aggregation functions to be used on every bin.
        Default is \['mean', 'median', 'sum', 'std', 'min', 'max'\].

    Returns
    -------
    ss : pd.DataFrame
        A pandas DataFrame with summary statistics per channel.
    """
    if agg_func is None:
        agg_func = ["mean", "median", "sum", "std", "min", "max"]
    if isinstance(bins, str):
        summary_stats = self.light.resample(bins).agg(agg_func)
    elif isinstance(bins, list):
        df_col = []
        for idx, (start, end) in enumerate(bins):
            df_bins = (
                self.light.loc[start:end, :]
                .apply(agg_func)
                .pivot_table(columns=agg_func)
            )

            channels = df_bins
            channels = channels.loc[:, agg_func]
            df_col.append(pd.concat(channels, axis=1))
            summary_stats = pd.concat(df_col)

    return summary_stats


def light_exposure_level(
    light, threshold=None, start_time=None, stop_time=None, agg="mean"
):
    r"""Light exposure level

    Calculate the aggregated (mean, median, etc) light exposure level
    per epoch.

    Parameters
    ----------
    threshold: float, optional
        If not set to None, discard data below threshold before computing
        exposure levels.
        Default is None.
    start_time: str, optional
        If not set to None, discard data before start time,
        on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    stop_time: str, optional
        If not set to None, discard data after stop time, on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    agg: str, optional
        Aggregating function used to summarize exposure levels.
        Available functions: 'mean', 'median', 'std', etc.
        Default is 'mean'.

    Returns
    -------
    levels : pd.Series
        A pandas Series with aggreagted light exposure levels per channel
    """
    light_exposure = _light_exposure(
        light=light, threshold=threshold, start_time=start_time, stop_time=stop_time
    )

    levels = getattr(light_exposure, agg)

    return levels()


def time_above_threshold(
    data, threshold=None, start_time=None, stop_time=None, oformat=None
):
    r"""Time above light threshold.

    Calculate the total light exposure time above the threshold.

    Parameters
    ----------
    threshold: float, optional
        If not set to None, discard data below threshold before computing
        exposure levels.
        Default is None.
    start_time: str, optional
        If not set to None, discard data before start time,
        on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    stop_time: str, optional
        If not set to None, discard data after stop time, on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    oformat: str, optional
        Output format. Available formats: 'minute' or 'timedelta'.
        If set to 'minute', the result is in number of minutes.
        If set to 'timedelta', the result is a pd.Timedelta.
        If set to None, the result is in number of epochs.
        Default is None.

    Returns
    -------
    tat : pd.Series
        A pandas Series with aggreagted light exposure levels per channel
    """
    available_formats = [None, "minute", "timedelta"]

    if oformat not in available_formats:
        raise ValueError(
            "Specified output format ({}) not supported. ".format(oformat)
            + "Available formats are: {}".format(str(available_formats))
        )

    light_exposure_counts = _light_exposure(
        light=data, threshold=threshold, start_time=start_time, stop_time=stop_time
    ).count()

    if oformat == "minute":
        tat = (
            light_exposure_counts * pd.Timedelta(data.index.freq) / pd.Timedelta("1min")
        )
    elif oformat == "timedelta":
        tat = light_exposure_counts * pd.Timedelta(data.index.freq)
    else:
        tat = light_exposure_counts

    return tat


def time_above_threshold_by_period(
    data, threshold=None, start_time=None, stop_time=None, oformat=None
):
    r"""Time above light threshold (per day).

    Calculate the total light exposure time above the threshold,
    per calendar day.

    Parameters
    ----------
    threshold: float, optional
        If not set to None, discard data below threshold before computing
        exposure levels.
        Default is None.
    start_time: str, optional
        If not set to None, discard data before start time,
        on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    stop_time: str, optional
        If not set to None, discard data after stop time, on a daily basis.
        Supported time string: 'HH:MM:SS'
        Default is None.
    oformat: str, optional
        Output format. Available formats: 'minute' or 'timedelta'.
        If set to 'minute', the result is in number of minutes.
        If set to 'timedelta', the result is a pd.Timedelta.
        If set to None, the result is in number of epochs.
        Default is None.

    Returns
    -------
    tatp : pd.DataFrame
        A pandas DataFrame with aggreagted light exposure levels
        per channel and per day.
    """
    available_formats = [None, "minute", "timedelta"]

    if oformat not in available_formats:
        raise ValueError(
            "Specified output format ({}) not supported. ".format(oformat)
            + "Available formats are: {}".format(str(available_formats))
        )

    light_exposure_counts_per_day = (
        _light_exposure(
            light=data, threshold=threshold, start_time=start_time, stop_time=stop_time
        )
        .groupby(data.index.date)
        .count()
    )

    if oformat == "minute":
        tatp = (
            light_exposure_counts_per_day
            * pd.Timedelta(data.index.freq)
            / pd.Timedelta("1min")
        )
    elif oformat == "timedelta":
        tatp = light_exposure_counts_per_day * pd.Timedelta(data.index.freq)
    else:
        tatp = light_exposure_counts_per_day

    return tatp


def values_above_threshold(data, threshold=None):
    r"""Values above light threshold.

    Returns the light exposure values above the threshold.

    Parameters
    ----------
    threshold: float, optional
        If not set to None, discard data below threshold before computing
        exposure levels.
        Default is None.

    Returns
    -------
    vat : pd.Series
        A pandas Series with light exposure levels per channel
    """
    return _light_exposure(
        light=data, threshold=threshold, start_time=None, stop_time=None
    )


def get_time_barycentre(data):
    # Normalize each epoch to midnight.
    Y_j = data.index - data.index.normalize()
    # Convert to indices.
    Y_j /= pd.Timedelta(data.index.freq)
    # Compute barycentre
    bc = data.multiply(Y_j, axis=0).sum() / data.sum()

    return bc


def mean_light_timing(light, threshold, freq=None):
    r"""Mean light timing.

    Mean light timing above threshold, MLiT^C.


    Parameters
    ----------
    threshold: float
        Threshold value.

    Returns
    -------
    MLiT : pd.DataFrame
        A pandas DataFrame with MLiT^C per channel.

    Notes
    -----

    The MLiT variable is defined in ref [1]_:

    .. math::

        MLiT^C = \frac{\sum_{j}^{m}\sum_{k}^{n} j\times I^{C}_{jk}}{
        \sum_{j}^{m}\sum_{k}^{n} I^{C}_{jk}}

    where :math:`I^{C}_{jk}` is equal to 1 if the light level is higher
    than the threshold C, m is the total number of epochs per day and n is
    the number of days covered by the data.

    References
    ----------

    .. [1] Reid K.J., Santostasi G., Baron K.G., Wilson J., Kang J.,
           Zee P.C., Timing and Intensity of Light Correlate with Body
           Weight in Adults. PLoS ONE 9(4): e92251.
           https://doi.org/10.1371/journal.pone.0092251

    """
    data = _data_processor(light=light, freq=freq)

    # Binarized data and convert to float in order to handle 'DivideByZero'
    I_jk = _binarize(data=data, threshold=threshold).astype("float64")

    MLiT = get_time_barycentre(I_jk)

    # Scaling factor: MLiT is now expressed in minutes since midnight.
    MLiT /= pd.Timedelta("1min") / I_jk.index.freq

    return MLiT


def mean_light_timing_by_period(light, threshold, freq=None):
    r"""Mean light timing per day.

    Mean light timing above threshold, MLiT^C, per calendar day.


    Parameters
    ----------
    threshold: float
        Threshold value.

    Returns
    -------
    MLiTp : pd.DataFrame
        A pandas DataFrame with MLiT^C per channel and per day.

    Notes
    -----

    The MLiT variable is defined in ref [1]_:

    .. math::

        MLiT^C = \frac{\sum_{j}^{m}\sum_{k}^{n} j\times I^{C}_{jk}}{
        \sum_{j}^{m}\sum_{k}^{n} I^{C}_{jk}}

    where :math:`I^{C}_{jk}` is equal to 1 if the light level is higher
    than the threshold C, m is the total number of epochs per day and n is
    the number of days covered by the data.

    References
    ----------

    .. [1] Reid K.J., Santostasi G., Baron K.G., Wilson J., Kang J.,
           Zee P.C., Timing and Intensity of Light Correlate with Body
           Weight in Adults. PLoS ONE 9(4): e92251.
           https://doi.org/10.1371/journal.pone.0092251

    """
    data = _data_processor(light=light, freq=freq)

    # Binarized data and convert to float in order to handle 'DivideByZero'
    I_jk = _binarize(data=data, threshold=threshold).astype("float64")

    # Group data per day:
    MLiTp = I_jk.groupby(I_jk.index.date).apply(get_time_barycentre)

    # Scaling factor: MLiT is now expressed in minutes since midnight.
    MLiTp /= pd.Timedelta("1min") / I_jk.index.freq

    return MLiTp


def get_extremum(data, extremum, freq=None):
    r"""Light extremum.

    Return the index and the value of the requested extremum (min or max).

    Parameters
    ----------
    extremum: str
        Name of the extremum.
        Available: 'min' or 'max'.

    Returns
    -------
    ext : pd.DataFrame
        A pandas DataFrame with extremum info per channel.
    """
    data = _data_processor(data=data, freq=freq)

    # Return either the maximum or minimum, as well as the respective timestamp
    if extremum == "max":
        return data.idxmax(), data.max()
    elif extremum == "min":
        return data.idxmin(), data.min()
    else:
        raise ValueError('Extremum must be "min" or "max"')


def lmx(data, length="5h", lowest=True, freq=None):
    r"""Least or Most light period of length X

    Onset and mean hourly light exposure levels during the X least or most
    bright hours of the day.

    Parameters
    ----------
    length: str, optional
        Period length.
        Default is '5h'.
    lowest: bool, optional
        If lowest is set to True, the period of least light exposure is
        considered. Otherwise, consider the period of most light exposure.
        Default is True.

    Returns
    -------
    lmx_t, lmx: (pd.Timedelta, float)
        Onset and mean hourly light exposure level.

    Notes
    -----

    The LMX variable is derived from the L5 and M10 defined in [1]_ as the
    mean hourly activity levels during the 5/10 least/most active hours.

    References
    ----------

    .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
           (1997). Long-Term Fitness Training Improves the Circadian
           Rest-Activity Rhythm in Healthy Elderly Males.
           Journal of Biological Rhythms, 12(2), 146–156.
           http://doi.org/10.1177/074873049701200206

    """
    data = _data_processor(light=data, freq=freq)

    # Calculate time of LMX and the value of LMX
    lmx_ts, lmx = _lmx(data, length, lowest=lowest)

    # Return these values back to the user
    return lmx_ts, lmx
