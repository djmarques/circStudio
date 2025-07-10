import numpy as np
import pandas as pd

__all__ = [
    "_actiware_automatic_threshold",
    "_padded_data",
    "_ratio_sequences_of_zeroes",
    "_estimate_zeta",
    "_window_convolution",
    "_window_convolution",
    "filter_ts_duration"
]

def _actiware_automatic_threshold(data, scale_factor=0.88888):
    r'''Automatic Wake Threshold Value calculation


    1. Sum the activity counts for all epochs of the data set.
    2. Count the number of epochs scored as MOBILE for the data set
    (the definition of MOBILE follows).
    3. Compute the MOBILE TIME (number of epochs scored as MOBILE from step 2
    multiplied by the Epoch Length) in minutes.
    4. Compute the Auto Wake Threshold = ((sum of activity counts from step 1)
    divided by (MOBILE TIME from step 3) ) multiplied by 0.88888.

    Definition of Mobile:
    An epoch is scored as MOBILE if the number of activity counts recorded in
    that epoch is greater than or equal to the epoch length in 15-second
    intervals. For example,there are four 15-second intervals for a 1-minute
    epoch length; hence, the activity value in an epoch must be greater than,
    or equal to, four, to be scored as MOBILE.
    '''

    # Sum of activity counts
    counts_sum = data.sum()

    # Definition of the "mobile" threshold
    mobile_thr = int(data.index.freq/pd.Timedelta('15sec'))

    # Counts the number of epochs scored as "mobile"
    counts_mobile = (data.values >= mobile_thr).astype(int).sum()

    # Mobile time
    mobile_time = counts_mobile*(data.index.freq/pd.Timedelta('1min'))

    # Automatic wake threshold
    automatic_thr = (counts_sum/mobile_time)*scale_factor

    return automatic_thr


def _padded_data(data, value, periods, frequency):

    date_offset = pd.DateOffset(seconds=frequency.total_seconds())
    pad_beginning = pd.Series(
        data=value,
        index=pd.date_range(
            end=data.index[0],
            periods=periods,
            freq=date_offset,
            inclusive='left'
        ),
        dtype=data.dtype
    )
    pad_end = pd.Series(
        data=value,
        index=pd.date_range(
            start=data.index[-1],
            periods=periods,
            freq=date_offset,
            inclusive='right'
        ),
        dtype=data.dtype
    )
    return pd.concat([pad_beginning, data, pad_end])


def _ratio_sequences_of_zeroes(
    data, seq_length, n_boostrap, seed=0, with_replacement=True
):
    # Calculate a moving sum with a window of size 'seq_length'
    rolling_sum = data.rolling(seq_length).sum()
    # Set seed for reproducibility
    np.random.seed(seed)
    random_sample = np.random.choice(
        rolling_sum.values,
        size=n_boostrap*len(rolling_sum),
        replace=with_replacement
    )
    # Calculate the ratio of zero elements
    ratio = 1 - np.count_nonzero(random_sample)/len(random_sample)
    return ratio


def _estimate_zeta(data, seq_length_max, n_boostrap=100, level=0.05):
    ratios = np.fromiter((
        _ratio_sequences_of_zeroes(data, n, n_boostrap) for n in np.arange(
            1, seq_length_max+1
        )),
        float,
        seq_length_max
    )
    zeta_est = np.argmax(ratios < level)
    return zeta_est


def _window_convolution(x, scale, window, offset=0.0):

    return scale * np.dot(x, window) + offset


def filter_ts_duration(ts, duration_min='3H', duration_max='12H'):
    """Filter time series according to their duration"""

    def duration(s):
        return s.index[-1]-s.index[0]

    td_min = pd.Timedelta(duration_min)
    td_max = pd.Timedelta(duration_max)

    from itertools import filterfalse
    filtered = []
    filtered[:] = filterfalse(
        lambda x: duration(x) < td_min or duration(x) > td_max,
        ts
    )
    return filtered