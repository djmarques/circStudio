import pandas as pd
import numpy as np
import warnings

from pandas.tseries.frequencies import to_offset
from ..filters import FiltersMixin
from ..metrics import MetricsMixin, _interval_maker
from ..sleep import SleepDiary, ScoringMixin, SleepBoutMixin


class BaseRaw(SleepBoutMixin, ScoringMixin, MetricsMixin, FiltersMixin):
    """Base class for raw data."""

    def __init__(self,
                 start_time,
                 period,
                 frequency,
                 activity,
                 light,
                 fpath=None):
        self.start_time = start_time
        self.period = period
        self.frequency = frequency
        self.activity = activity
        self.light = light
        self._inactivity_length = None
        self.exclude_if_mask = True
        self.mask_inactivity = False
        self._mask = None
        self.sleep_diary = None

    def length(self):
        r"""Number of activity data acquisition points"""
        return len(self.activity)

    def time_range(self):
        r"""Range (in days, hours, etc) of the activity data acquistion period"""
        return (self.activity.index[-1]-self.activity.index[0])

    def duration(self):
        r"""Duration (in days, hours, etc) of the activity data acquistion period"""
        return self.frequency * self.length()


    def resample_activity(self, freq):
        r"""Resample activity data at the specified frequency, with or without mask."""
        # Return original time series if freq isn't specified or lower than the original sampling frequency
        if freq is None or pd.Timedelta(to_offset(freq)) <= self.frequency:
            return self.activity

        resampled_data = self.activity.resample(freq, origin='start').sum()
        if self.mask_inactivity is True:
            if self.mask is None:
                print('No mask was found. Create a new mask')
                return self.activity

            elif self.exclude_if_mask:
                resampled_mask = self.mask.resample(freq, origin='start').min()

            else:
                resampled_mask = self.mask.resample(freq, origin='start').max()

            return resampled_data.where(resampled_mask > 0)

        else:

            return resampled_data

        # Return resampled activity time series
        return self.activity.resample(freq, origin='start').sum()

    def resample_light(self, freq):
        """Light time series, resampled at the specified frequency."""

        # Return original light time series if
        if freq is None or pd.Timedelta(to_offset(freq)) <= self.frequency:
            return self.light

        # Return resampled light time series
        return light.resample(freq, origin='start').sum()

    @property
    def mask(self):
        r"""Mask used to filter out inactive data."""
        if self._mask is None:
            # Create a mask if it does not exist
            if self._inactivity_length is not None:
                # Create an inactivity mask with the specified length (and above)
                self.create_inactivity_mask(self._inactivity_length)
                return self._mask.loc[self.start_time:self.start_time+self.period]
            else:
                print('Inactivity length set to None. Could not create a mask.')
        else:
            return self._mask.loc[self.start_time:self.start_time+self.period]

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def inactivity_length(self):
        r"""Length of the inactivity mask."""
        return self._inactivity_length

    @inactivity_length.setter
    def inactivity_length(self, value):
        self._inactivity_length = value
        # Discard current mask (will be recreated upon access if needed)
        self._mask = None
        # Set switch to False if None
        if value is None:
            self.mask_inactivity = False


    def read_sleep_diary(
            self,
            input_fname,
            header_size=2,
            state_index=dict(ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1),
            state_colour=dict(NAP='#7bc043', NIGHT='#d3d3d3', NOWEAR='#ee4035')
    ):
        r"""Reader function for sleep diaries.

        Parameters
        ----------
        input_fname: str
            Path to the sleep diary file.
        header_size: int
            Header size (i.e. number of lines) of the sleep diary.
            Default is 2.
        state_index: dict
            The dictionnary of state's indices.
            Default is ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1.
        state_color: dict
            The dictionnary of state's colours.
            Default is NAP='#7bc043', NIGHT='#d3d3d3', NOWEAR='#ee4035'.
        """

        self.sleep_diary = SleepDiary(
            input_fname=input_fname,
            start_time=self.start_time,
            periods=self.length(),
            frequency=self.frequency,
            header_size=header_size,
            state_index=state_index,
            state_colour=state_colour
        )