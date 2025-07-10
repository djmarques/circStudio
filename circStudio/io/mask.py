import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

from ..analysis.tools import *


class BaseLog:
    """Base class for log files containing time stamps.

    Parameters
    ----------
    fname: str
        Absolute filepath of the input log file.
    log: pandas.DataFrame
        Dataframe containing the data found in the log file.
    """

    def __init__(self, input_fname, log):

        # get absolute file path
        self.__fname = os.path.abspath(input_fname)

        # Add `duration` column
        log["Duration"] = log["Stop_time"] - log["Start_time"]

        # Inplace drop of NA
        log.dropna(inplace=True)

        # add dataframe
        self.__log = log

    @classmethod
    def from_file(cls, input_fname, index_name, *args, **kwargs):
        """Read start/stop-times from log files.

        Generic function to read start and stop times from log files. Supports
        different file format (.ods, .xls(x), .csv).

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.
        *args
            Variable length argument list passed to the subsequent reader
            function.
        **kwargs
            Arbitrary keyword arguments passed to the subsequent reader
            function.

        Returns
        -------
        absname: str
            Absolute filepath of the input log file.
        log: pandas.DataFrame
            Dataframe containing the data found in the log file.

        """

        # get absolute file path
        absname = os.path.abspath(input_fname)

        # get basename and split it into base and extension
        basename = os.path.basename(absname)
        _, ext = os.path.splitext(basename)

        if ext == ".csv":
            log = cls.__from_csv(absname, index_name, *args, **kwargs)
        elif (ext == ".xlsx") or (ext == ".xls") or (ext == ".ods"):
            log = cls.__from_excel(absname, index_name, *args, **kwargs)
        else:
            raise ValueError(
                (
                    "File format for the input file {}".format(basename)
                    + "is not currently supported."
                    + "Supported file format:\n"
                    + ".csv (text),\n"
                    + ".ods (OpenOffice spreadsheet),\n"
                    + ".xls (Excel spreadsheet)."
                )
            )
        return absname, log

    @classmethod
    def __from_csv(cls, input_fname, index_name, sep=",", dayfirst=False):
        """Read start/stop-times from .csv files.

        Specific function to read start and stop times from csv files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.
        sep: str, optional
            Delimiter to use.
            Default is ','.
        dayfirst: bool, optional
            If set to True, use DD/MM/YYYY format dates.
            Default is False.

        Returns
        -------
        log : a pandas.DataFrame
            A dataframe with the start and stop times (columns)

        """

        # Read data from the csv file into a dataframe
        log = pd.read_csv(
            input_fname,
            sep=sep,
            dayfirst=dayfirst,
            header=0,
            index_col=[0],
            usecols=[0, 1, 2],
            names=[index_name, "Start_time", "Stop_time"],
            parse_dates=[1, 2],
        )
        return log

    @classmethod
    def __from_excel(cls, input_fname, index_name):
        """Read start/stop-times from excel-like files.

        Specific function to read start and stop times from .ods/.xls(x) files.

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        index_name: str
            Name of the index.

        Returns
        -------
        log : a pandas.DataFrame
            A dataframe with the start and stop times (columns)

        """

        # Read data from the log file into a np array
        sst_narray = np.array(pxl.get_array(file_name=input_fname))

        # Create a DF with columns: index_name, start_time, stop time
        log = pd.DataFrame(
            sst_narray[1:, 1:3],
            index=sst_narray[1:, 0],
            columns=["Start_time", "Stop_time"],
            # dtype='datetime64[ns]'
        )
        log.index.name = index_name

        return log

    @property
    def fname(self):
        """The absolute filepath of the input log file."""
        return self.__fname

    @property
    def log(self):
        """The dataframe containing the data found in the log file."""
        return self.__log

    def summary(self, colname):
        """Returns a dataframe of summary statistics."""
        return self.__log[colname].describe()


class Mask:
    def __init__(self, exclude_if_mask, mask_inactivity, binarize, threshold, inactivity_length, mask):
        self.exclude_if_mask = exclude_if_mask
        self._inactivity_length = inactivity_length
        self._mask_inactivity = mask_inactivity
        self._original_activity = self.activity
        self._original_light = self.light if self.light is not None else None
        self.binarize = binarize
        self.threshold = threshold if binarize else None
        self.impute_nan = False
        self.imputation_method = 'mean'
        self._mask = mask

    def _filter_data(self,
                     data,
                     new_freq,
                     binarize,
                     impute_nan,
                     threshold,
                     exclude_if_mask,
                     imputation_method):
        self.threshold = threshold
        self.binarize = binarize
        self.impute_nan = impute_nan
        self.imputation_method = imputation_method
        self.exclude_if_mask = exclude_if_mask

        return _data_processor(
            data=data,
            binarize=self.binarize,
            threshold=self.threshold,
            current_freq=self.frequency,
            new_freq=self.frequency if new_freq is None else new_freq,
            mask=self._mask,
            mask_inactivity=self._mask_inactivity,
            impute_nan=self.impute_nan,
            imputation_method=self.imputation_method,
            exclude_if_mask=self.exclude_if_mask,
        )

    def apply_filters(self,
                      new_freq=None,
                      binarize=False,
                      threshold=0,
                      impute_nan=False,
                      exclude_if_mask=False,
                      imputation_method='mean'):
        self._mask_inactivity = True
        if self.activity is not None:
            self.activity = self._filter_data(self.activity,
                                              new_freq=new_freq,
                                              binarize=binarize,
                                              threshold=threshold,
                                              impute_nan=impute_nan,
                                              exclude_if_mask=False,
                                              imputation_method=imputation_method)
        if self.light is not None:
            self.light = self._filter_data(self.light,
                                           new_freq=new_freq,
                                           binarize=binarize,
                                           threshold=threshold,
                                           impute_nan=impute_nan,
                                           exclude_if_mask=False,
                                           imputation_method=imputation_method)


    def reset_filters(self, new_freq=None):
        self._mask_inactivity = False
        if self.activity is not None:
            self.activity = self._original_activity
        if self.light is not None:
            self.light = self._original_light

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
            self._mask_inactivity = False


    def create_inactivity_mask(self, duration):
        """Create a mask for inactivity (count equal to zero) periods.

        This mask has the same length as its underlying data and can be used
        to offuscate inactive periods where the actimeter has most likely been
        removed.
        Warning: use a sufficiently long duration in order not to mask sleep
        periods.
        A minimal duration corresponding to two hours seems reasonable.

        Parameters
        ----------
        duration: int or str
            Minimal number of consecutive zeroes for an inactive period.
            Time offset strings (ex: '90min') can also be used.
        """

        if isinstance(duration, int):
            nepochs = duration
        elif isinstance(duration, str):
            nepochs = int(pd.Timedelta(duration) / self.frequency)
        else:
            nepochs = None
            warnings.warn(
                "Inactivity length must be a int and time offset string (ex: "
                "'90min'). Could not create a mask.",
                UserWarning,
            )

        # Store requested mask duration (and discard current mask)
        self.inactivity_length = nepochs

        # Create actual mask
        self.mask = _create_inactivity_mask(self.activity, nepochs, 1)


    def add_mask_period(self, start, stop):
        """Add a period to the inactivity mask

        Parameters
        ----------
        start: str
            Start time (YYYY-MM-DD HH:MM:SS) of the inactivity period.
        stop: str
            Stop time (YYYY-MM-DD HH:MM:SS) of the inactivity period.
        """

        # Check if a mask has already been created
        # NB : if the inactivity_length is not None, accessing the mask will
        # trigger its creation.
        if self.inactivity_length is None:
            self.inactivity_length = -1
            # self.mask = pd.Series(
            #     np.ones(self.length()),
            #     index=self.data.index
            # )

        # Check if start and stop are within the index range
        if pd.Timestamp(start) < self.mask.index[0]:
            raise ValueError(
                (
                    "Attempting to set the start time of a mask period before "
                    + "the actual start time of the data.\n"
                    + "Mask start time: {}".format(start)
                    + "Data start time: {}".format(self.mask.index[0])
                )
            )
        if pd.Timestamp(stop) > self.mask.index[-1]:
            raise ValueError(
                (
                    "Attempting to set the stop time of a mask period after "
                    + "the actual stop time of the data.\n"
                    + "Mask stop time: {}".format(stop)
                    + "Data stop time: {}".format(self.mask.index[-1])
                )
            )

        # Set mask values between start and stop to zeros
        # self.mask.loc[start:stop] = 0
        self.mask = self.mask.mask(
            (self.mask.index >= start) & (self.mask.index <= stop), 0
        )

    def add_mask_periods(self, input_fname, *args, **kwargs):
        """Add periods to the inactivity mask

        Function to read start and stop times from a Mask log file. Supports
        different file format (.ods, .xls(x), .csv).

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        *args
            Variable length argument list passed to the subsequent reader
            function.
        **kwargs
            Arbitrary keyword arguments passed to the subsequent reader
            function.
        """

        # Convert the log file into a DataFrame
        absname, log = BaseLog.from_file(input_fname, "Mask", *args, **kwargs)

        # Iterate over the rows of the DataFrame
        for _, row in log.iterrows():
            self.add_mask_period(row["Start_time"], row["Stop_time"])