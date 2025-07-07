import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


class Mask:
    def __init__(self, exclude_if_mask, mask_inactivity, inactivity_length, mask):
        self.exclude_if_mask = exclude_if_mask
        self._inactivity_length = inactivity_length
        self.mask_inactivity = mask_inactivity
        self._mask = mask

    @staticmethod
    def binarize(data, threshold):
        binarized = pd.Series(
            np.where(data > threshold, 1, 0), index=data.index
        ).where(data.notna(), np.nan)
        return binarized

    def resample(self, data, binarize=False, freq=None):
        r"""Resample data at the specified frequency, with or without mask."""

        # Return original time series if freq is not specified or lower than the sampling frequency
        if freq is None or pd.Timedelta(to_offset(freq)) <= self.frequency:
            # If asked to mask inactivity, return only the values where the mask is 1
            if self.mask_inactivity:
                # Check if the mask exists and apply it
                if self._mask is not None:
                    return data.where(self._mask > 0)
                else:
                    # Return original data in case no mask is found.
                    print("No mask was found. Create a new mask.")
                    return data
            else:
                # Return original data is mask_inactivity is set to False
                return data

        else:
            # After the initial checks, resample activity trace (sum all the counts within the resampling window)
            resampled_data = data.resample(freq, origin="start").sum()

            # If mask inactivity is set to False, return the resampled trace
            if not self.mask_inactivity:
                return resampled_data

            # Catch the scenario where mask inactivity is true but no mask is found
            elif self.mask_inactivity and self._mask is None:
                print("No mask was found. Create a new mask.")
                return resampled_data

            # When resampling, exclude all the resampled timepoints within the new resampling window
            elif self.mask_inactivity and self.exclude_if_mask:
                # Capture the minimum (0) for each resampling bin
                resampled_mask = self._mask.resample(freq, origin="start").min()

                # Return the masked resampled activity trace
                return resampled_data.where(resampled_mask > 0)

            # When resampling, do not exclude all the resampled timepoints within the new resampling window
            else:
                resampled_mask = self._mask.resample(freq, origin="start").max()

                # Return the masked resampled activity trace
                return resampled_data.where(resampled_mask > 0)

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
