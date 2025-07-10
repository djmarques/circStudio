import pandas as pd
import numpy as np

from pandas.tseries.frequencies import to_offset
from .mask import Filters
from ..sleep import SleepDiary, ScoringMixin, SleepBoutMixin
from .mask import Mask


class BaseRaw(Filters, Mask):
    """Base class for raw actigraphy data."""

    def __init__(self, period, frequency, activity, light, fpath=None, start_time=None, stop_time=None):
        self.start_time = start_time
        self.stop_time = stop_time
        self.period = period
        self.frequency = frequency
        self.light = light
        self.activity = activity
        self.sleep_diary = None
        super().__init__(
            exclude_if_mask=True,
            mask_inactivity=False,
            inactivity_length=None,
            mask=None,
        )

    def length(self):
        r"""Number of activity data acquisition points"""
        return len(self.activity)

    def time_range(self):
        r"""Range (in days, hours, etc) of the activity data acquistion period"""
        return self.activity.index[-1] - self.activity.index[0]

    def duration(self):
        r"""Duration (in days, hours, etc) of the activity data acquistion period"""
        return self.frequency * self.length()

    def read_sleep_diary(
        self, input_fname, header_size=2, state_index=None, state_colour=None
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
            Dictionnary of state's indices.
        state_color: dict
            Dictionnary of state's colours.
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
