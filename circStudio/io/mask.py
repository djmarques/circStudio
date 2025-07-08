import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from ..analysis.auxiliary_functions import *


class Mask:
    def __init__(self, exclude_if_mask, mask_inactivity, inactivity_length, mask):
        self.exclude_if_mask = exclude_if_mask
        self._inactivity_length = inactivity_length
        self.mask_inactivity = mask_inactivity
        self._mask = mask

    @staticmethod
    def binarize(data, threshold):
        return _binarize(data, threshold)

    def resample(self, data, binarize=False, freq=None):
        r"""Resample data at the specified frequency, with or without mask."""
        return _resample(data,
                         binarize=binarize,
                         new_freq=freq,
                         current_freq=self.frequency,
                         mask_inactivity=self.mask_inactivity,
                         exclude_if_mask=self.exclude_if_mask,
                         mask=self._mask)

    @property
    def mask(self):
        r"""Mask used to filter out inactive data."""
        if self._mask is None:
            # Create a mask if it does not exist
            if self._inactivity_length is not None:
                # Create an inactivity mask with the specified length (and above)
                self.create_inactivity_mask(self._inactivity_length)
                return self._mask.loc[self.start_time : self.start_time + self.period]
            else:
                print("Inactivity length set to None. Could not create a mask.")
        else:
            return self._mask.loc[self.start_time : self.start_time + self.period]

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
