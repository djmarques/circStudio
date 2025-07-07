import numpy as np
import pandas as pd
import pyexcel as pxl
import warnings


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


class FiltersMixin(object):
    """Mixin Class"""

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
