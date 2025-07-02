import pandas as pd
import numpy as np

from pandas.tseries.frequencies import to_offset
from ..filters import FiltersMixin
from ..metrics import MetricsMixin
from ..sleep import SleepDiary, ScoringMixin, SleepBoutMixin
from .mask import Mask


class BaseRaw(SleepBoutMixin, ScoringMixin, MetricsMixin, FiltersMixin, Mask):
    """Base class for raw actigraphy data."""

    def __init__(self, start_time, period, frequency, activity, light, fpath=None):
        self.start_time = start_time
        self.period = period
        self.frequency = frequency
        self.activity = activity
        self.light = light
        self.sleep_diary = None
        super().__init__(exclude_if_mask=True, mask_inactivity=False, inactivity_length=None, mask=None)

    def length(self):
        r"""Number of activity data acquisition points"""
        return len(self.activity)

    def time_range(self):
        r"""Range (in days, hours, etc) of the activity data acquistion period"""
        return self.activity.index[-1] - self.activity.index[0]

    def duration(self):
        r"""Duration (in days, hours, etc) of the activity data acquistion period"""
        return self.frequency * self.length()

    def resample_light(self, freq):
        """Light time series, resampled at the specified frequency."""

        # Return original time series if freq is not specified or lower than the sampling frequency
        if freq is None or pd.Timedelta(to_offset(freq)) <= self.frequency:
            return self.light

        # Return resampled light time series
        return light.resample(freq, origin="start").sum()

    def read_sleep_diary(
        self,
        input_fname,
        header_size=2,
        state_index=dict(ACTIVE=2, NAP=1, NIGHT=0, NOWEAR=-1),
        state_colour=dict(NAP="#7bc043", NIGHT="#d3d3d3", NOWEAR="#ee4035"),
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
            state_colour=state_colour,
        )
