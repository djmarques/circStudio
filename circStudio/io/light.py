import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


class Light:
    def __init__(self):
        print("initialing light")

    def resample_light(self, freq):
        """Light time series, resampled at the specified frequency."""

        # Return original time series if freq is not specified or lower than the sampling frequency
        if freq is None or pd.Timedelta(to_offset(freq)) <= self.frequency:
            return self.light

        # Return resampled light time series
        return light.resample(freq, origin="start").sum()
