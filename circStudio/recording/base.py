import numpy as np
import pandas as pd
import warnings
from .utils import _binarized_data
from .utils import _resampled_data

# Deprecated: redundant code, to be fused with BaseRaw
class BaseRecording:
    """ Base class for any type of recording.

    Parameters
    ----------
    name: str
        Name of the recording.
    data: pandas.DataFrame
        Dataframe containing the data found in the recording.

    """

    def __init__(
        self,
        data,
        frequency,
        start_time=None,
        stop_time=None,
        period=None,
        mask=None
    ):

        # Mandatory fields
        self._data = data
        self._frequency = frequency

        # Optional fields
        # User-specified start/stop/period
        self._start_time = start_time
        self._stop_time = stop_time
        self._period = period

        # Mask-related fields
        self._mask = mask
        self._apply_mask = False
        self._exclude_if_mask = True

    @property
    def apply_mask(self):
        r"""Mask indicator."""
        return self._apply_mask

    @apply_mask.setter
    def apply_mask(self, value):
        self._apply_mask = value

    @property
    def start_time(self):
        r"""Start time of the recording."""
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        if (self._stop_time is not None) and (self._period is not None):
            raise ValueError(
                'Stop time and period fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self._start_time = pd.Timestamp(value)

    @property
    def stop_time(self):
        r"""Stop time of the recording."""
        return self._stop_time

    @stop_time.setter
    def stop_time(self, value):
        if (self._start_time is not None) and (self._period is not None):
            raise ValueError(
                'Start time and period fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self._stop_time = pd.Timestamp(value)

    @property
    def period(self):
        r"""Time period of the recording."""
        return self._period

    @period.setter
    def period(self, value):
        if (self._start_time is not None) and (self._stop_time is not None):
            raise ValueError(
                'Start and stop time fields have already been set.\n'
                + 'Use the reset_times function first.'
            )
        self._period = pd.Timedelta(value)
        # Optionally compute start or stop time
        # NB one epoch has to be suppressed as accessing data with .loc[]
        # includes the last epoch.
        if self._start_time is not None:
            self._stop_time = self._start_time + \
                (self._period - self.frequency)
        elif self._stop_time is not None:
            self._start_time = self._stop_time - \
                (self._period - self.frequency)

    def reset_times(self):
        r"""Reset start and stop times, as well as the period of the recording.
        """
        self._start_time = None
        self._stop_time = None
        self._period = None

    @property
    def frequency(self):
        r"""Acquisition frequency of the recording."""
        return self._frequency

    @property
    def raw_data(self):
        r"""Indexed data extracted from the raw file."""
        return self._data

    @property
    def data(self):
        r"""Data of the recording.

        If apply_mask is set to true, the `mask` is used
        to filter out data.
        """
        if self._data is None:
            return self._data

        if self.apply_mask is True:
            if self.mask is not None:
                data = self.raw_data.where(self.mask > 0)
            else:
                warnings.warn(
                    (
                        'Apply_mask set to True but no mask could be'
                        ' found.\n Please create a mask by using the'
                        ' appropriate "create_mask" function.'
                    ),
                    UserWarning
                )
                data = self.raw_data
        else:
            data = self.raw_data
        return data.loc[self.start_time:self.stop_time]

    def resampled_data(self, rsfreq, agg='sum'):

        return _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

    def binarized_data(self, threshold, rsfreq=None, agg='sum'):

        if rsfreq is None:
            rsdata = self.data
        else:
            rsdata = _resampled_data(self.data, rsfreq=rsfreq, agg=agg)

        return _binarized_data(rsdata, threshold)

    @property
    def mask(self):
        r"""Mask used to filter out data."""
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value
