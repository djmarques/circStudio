import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.tseries.frequencies import to_offset


__all__ = [
    "_average_daily_activity",
    "_shift_time_axis",
    "_onset_detection",
    "_offset_detection",
    "_activity_inflexion_time",
    "_activity_onset_time",
    "_activity_offset_time",
    "_binarize",
    "_resample",
    "_average_daily_total_activity",
    "_lmx",
    "_interval_maker",
    "_count_consecutive_values",
    "_count_consecutive_zeros",
    "_transition_prob",
    "_transition_prob_sustain_region",
    "_td_format",
    "_data_processor",
    "_light_exposure",
    "_create_inactivity_mask"
]

def _average_daily_activity(data, cyclic=False):
    """Calculate the average daily activity distribution"""

    avgdaily = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ]).mean()

    if cyclic:
        avgdaily = pd.concat([avgdaily, avgdaily])
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='2 days',
            freq=data.index.freq,
            closed='left'
        )
    else:
        avgdaily.index = pd.timedelta_range(
            start='0 day',
            end='1 day',
            freq=data.index.freq,
            closed='left'
        )

    return avgdaily


def _shift_time_axis(avgdaily, shift):

    avgdaily_shifted = avgdaily.reindex(
        index=np.roll(avgdaily.index, shift)
    )
    avgdaily_shifted.index = pd.timedelta_range(
        start='-12h',
        end='12h',
        freq=avgdaily.index.freq,
        closed='left'
    )

    return avgdaily_shifted


def _onset_detection(x, whs):
    return np.mean(x[whs:])/np.mean(x[0:whs])-1


def _offset_detection(x, whs):
    return np.mean(x[0:whs])/np.mean(x[whs:])-1


def _activity_inflexion_time(data, fct, whs):

    r = data.rolling(whs*2, center=True)

    aot = r.apply(fct, kwargs={'whs': whs}, raw=True).idxmax()

    return aot


def _activity_onset_time(data, whs=4):

    return _activity_inflexion_time(data, _onset_detection, whs)


def _activity_offset_time(data, whs=4):

    return _activity_inflexion_time(data, _offset_detection, whs)


def _binarize(data, threshold):
    binarized = pd.Series(
        np.where(data > threshold, 1, 0), index=data.index
    ).where(data.notna(), np.nan)
    return binarized


def _resample(data,
              new_freq=None,
              current_freq=None,
              mask_inactivity=False,
              exclude_if_mask = False,
              mask = None):
    r"""Resample data at the specified frequency, with or without mask."""

    # Return original time series if freq is not specified or lower than the sampling frequency
    if current_freq is None or pd.Timedelta(to_offset(new_freq)) <= current_freq:
        # If asked to mask inactivity, return only the values where the mask is 1
        if mask_inactivity:
            # Check if the mask exists and apply it
            if mask is not None:
                return data.where(mask > 0)
            else:
                # Return original data in case no mask is found.
                return data
        else:
            # Return original data is mask_inactivity is set to False
            return data

    else:
        # After the initial checks, resample activity trace (sum all the counts within the resampling window)
        resampled_data = data.resample(new_freq, origin="start").sum()

        # If mask inactivity is set to False, return the resampled trace
        if not mask_inactivity:
            return resampled_data

        # Catch the scenario where mask inactivity is true but no mask is found
        elif mask_inactivity and mask is None:
            return resampled_data

        # When resampling, exclude all the resampled timepoints within the new resampling window
        elif mask_inactivity and exclude_if_mask:
            # Capture the minimum (0) for each resampling bin
            resampled_mask = mask.resample(new_freq, origin="start").min()

            # Return the masked resampled activity trace
            return resampled_data.where(resampled_mask > 0)

        # When resampling, do not exclude all the resampled timepoints within the new resampling window
        else:
            resampled_mask = mask.resample(new_freq, origin="start").max()

            # Return the masked resampled activity trace
            return resampled_data.where(resampled_mask > 0)


def _average_daily_total_activity(data, rescale, exclude_ends):
    r"""Calculate the average daily activity"""

    # Shortcut: if rescale is false, compute the daily average
    if rescale is False:
        daily_sum = data.resample("1D").sum()
    else:
        # Aggregate data daily:
        # - compute the daily sum
        # - count the number of epochs included in each day
        daily_agg = data.resample("1D").agg(["count", "sum"])

        # Compute weights as a function of the number of epochs per day:
        # weight =  (#epochs/day) / (#count/day)
        # NB: needed to account for potentially masked periods.
        daily_agg["weigth"] = pd.Timedelta("24h") / data.index.freq
        daily_agg["weigth"] /= daily_agg["count"]

        # Rescale activity
        daily_sum = daily_agg["sum"] * daily_agg["weigth"]

    # Exclude first and last days
    if exclude_ends:
        daily_sum = daily_sum.iloc[1:-1]

    return daily_sum.mean()


def _lmx(data, period, lowest=True):
    """Calculate the start time and mean activity of the period of
    lowest/highest activity"""

    avgdaily = _average_daily_activity(data=data, cyclic=True)

    n_epochs = int(pd.Timedelta(period) / avgdaily.index.freq)

    mean_activity = avgdaily.rolling(period).sum().shift(-n_epochs + 1)

    if lowest:
        t_start = mean_activity.idxmin()
    else:
        t_start = mean_activity.idxmax()

    lmx = mean_activity[t_start] / n_epochs
    return t_start, lmx


def _interval_maker(index, period, verbose):
    (num_periods, td) = divmod((index[-1] - index[0]), pd.Timedelta(period))
    if verbose:
        print(
            "Number of periods: {0}\n Time unaccounted for: {1}".format(
                num_periods,
                "{} days, {}h, {}m, {}s".format(
                    td.days,
                    td.seconds // 3600,
                    (td.seconds // 60) % 60,
                    td.seconds % 60,
                ),
            )
        )

    intervals = [
        (
            index[0] + (i) * pd.Timedelta(period),
            index[0] + (i + 1) * pd.Timedelta(period),
        )
        for i in range(0, num_periods)
    ]

    return intervals


def _count_consecutive_values(data):
    """Create a count list for identical consecutive numbers
    together with a state for each series:
     - 1 if the sum of the consecutive series numbers is positive
     - 0 otherwise
    """

    consecutive_values = data.groupby(
        # create identical 'labels' for identical consecutive numbers
        [data.diff().ne(0).cumsum()]
    ).agg(["count", lambda x: (np.sum(x) > 0).astype(int)])
    # rename columns
    consecutive_values.columns = ["counts", "state"]

    return consecutive_values


def _count_consecutive_zeros(data):
    ccz = _count_consecutive_values(data)
    ccz["end"] = ccz["counts"].cumsum()
    ccz["start"] = ccz["end"].shift(1).fillna(0).astype(int)
    return ccz[ccz["state"] < 1]


def _transition_prob(data, from_zero_to_one):
    # Create a list of consecutive sequence of active/rest epochs
    ccv = _count_consecutive_values(data)
    # filter out sequences of active epochs
    if from_zero_to_one is True:
        bouts = ccv[ccv["state"] < 1]["counts"]
    else:
        bouts = ccv[ccv["state"] > 0]["counts"]
    # Count the number of sequences of length N for N=1...Nmax
    Nt = bouts.groupby(bouts).count()
    # Create its reverse cumulative sum so that Nt at index t is equal to
    # the number of sequences of lengths t or longer.
    Nt = np.cumsum(Nt[::-1])[::-1]
    # Rest->Activity (or Activity->Rest) transition probability at time t,
    # defined as the number of sequences for which R->A at time t+1 / Nt
    prob = Nt.diff(-1) / Nt
    # Correct pRA for discontinuities due to sparse data
    prob = prob.dropna() / np.diff(prob.index.tolist())
    # Define the weights as the square root of the number of runs
    # contributing to each probability estimate
    prob_weights = np.sqrt(Nt + Nt.shift(-1)).dropna()

    return prob, prob_weights


def _transition_prob_sustain_region(prob, prob_weights, frac=0.3, it=0):

    # Fit the 'prob' distribution with a LOWESS
    lowess = sm.nonparametric.lowess(
        prob.values, prob.index, return_sorted=False, frac=frac, it=it
    )

    # Calculate the pRA std
    std = prob.std()

    # Check which residuals are below 1 sigma
    prob_residuals_below_one_std = _count_consecutive_values(
        ((prob - lowess).abs() < std).astype(int)
    )

    # Find the index of the longest series of consecutive values below 1 SD
    index = (
        prob_residuals_below_one_std[prob_residuals_below_one_std["state"] > 0][
            "counts"
        ].idxmax()
        - 1
    )

    # Calculate the cumulative sum of the indices of series of consecutive
    # values of residuals below 1 SD in order to find the number of points
    # before the "index".
    prob_cumsum = prob_residuals_below_one_std["counts"].cumsum()

    # Calculate the start and end indices
    if index < prob_cumsum.index.min():
        start_index = 0
    else:
        start_index = prob_cumsum[index]
    # start_index = prob_cumsum[index]+1
    end_index = prob_cumsum[index + 1]

    kProb = np.average(
        prob[start_index:end_index], weights=prob_weights[start_index:end_index]
    )
    return kProb


def _td_format(td):
    return "{:02}:{:02}:{:02}".format(
        td.components.hours, td.components.minutes, td.components.seconds
    )


def _data_processor(data,
                    binarize=False,
                    threshold=0,
                    mask=None,
                    current_freq = None,
                    new_freq = None,
                    mask_inactivity = False,
                    exclude_if_mask = False
                    ):
    """
    This function binarizes or resamples data.
    Parameters
    ----------
    data
    binarize
    threshold

    Returns
    -------
    Resampled light time series
    """
    # Binarize data using a given threshold
    if binarize:
        data = _binarize(data=data, threshold=threshold)
    else:
        data = data

    # Resample and apply existing mask if enabled

    return _resample(data=data,
                     mask=mask,
                     new_freq=new_freq,
                     mask_inactivity=mask_inactivity,
                     current_freq=current_freq,
                     exclude_if_mask = exclude_if_mask)



def _light_exposure(light, threshold=None, start_time=None, stop_time=None):
    r"""Light exposure

    Calculate the light exposure level and time

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

    Returns
    -------
    masked_data : pandas.DataFrame
        A DataFrame where the original data are set to Nan if below
        threshold and/or outside time window.
    """
    if threshold is not None:
        data_mask = light.mask(light < threshold)
    else:
        data_mask = light

    if start_time is stop_time is None:
        return data_mask
    elif (start_time is None) or (stop_time is None):
        raise ValueError('Both start and stop times have to be specified, if any.')
    else:
        return data_mask.between_time(start_time=start_time, end_time=stop_time, include_end=False)


def _create_inactivity_mask(data, duration, threshold):

    if duration is None:
        return None

    # Create the mask filled with ones by default.
    mask = np.ones_like(data)

    # If duration is -1, return a mask with 1s for later manual edition.
    if duration == -1:
        return pd.Series(mask, index=data.index)

    # Binary data
    binary_data = np.where(data >= threshold, 1, 0)

    # The first order diff Series indicates the indices of the transitions
    # between series of zeroes and series of ones.
    # Add zero at the beginning of this series to mark the beginning of the
    # first sequence found in the data.
    edges = np.concatenate([[0], np.diff(binary_data)])

    # Test if there is no edge (i.e. no consecutive zeroes).
    if all(e == 0 for e in edges):
        return pd.Series(mask, index=data.index)

    # Indices of upper transitions (zero to one).
    idx_plus_one = (edges > 0).nonzero()[0]

    # Indices of lower transitions (one to zero).
    idx_minus_one = (edges < 0).nonzero()[0]

    # Even number of transitions.
    if idx_plus_one.size == idx_minus_one.size:

        # Start with zeros
        if idx_plus_one[0] < idx_minus_one[0]:
            starts = np.concatenate([[0], idx_minus_one])
            ends = np.concatenate([idx_plus_one, [edges.size]])
        else:
            starts = idx_minus_one
            ends = idx_plus_one

    # Odd number of transitions
    # starting with an upper transition
    elif idx_plus_one.size > idx_minus_one.size:
        starts = np.concatenate([[0], idx_minus_one])
        ends = idx_plus_one
    # starting with an lower transition
    else:
        starts = idx_minus_one
        ends = np.concatenate([idx_plus_one, [edges.size]])

    # Index pairs (start,end) of the sequences of zeroes
    seq_idx = np.c_[starts, ends]
    # Length of the aforementioned sequences
    seq_len = ends - starts

    for i in seq_idx[np.where(seq_len >= duration)]:
        mask[i[0] : i[1]] = 0

    return pd.Series(mask, index=data.index)

