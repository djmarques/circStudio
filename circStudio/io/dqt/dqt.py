import pandas as pd
import os
import warnings
from ..base import Raw


class DQT(Raw):
    r"""Raw object from .csv file recorded by Daqtometers (Daqtix, Germany)

    Parameters
    ----------
    input_fname: str
        Path to the Daqtometer file.
    header_size: int, optional
        Header size (i.e. number of lines) of the raw data file.
        Default is 15.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    """

    def __init__(
        self,
        input_fname,
        header_size=15,
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # extract header and data size
        with open(input_fname) as f:
            header = [next(f) for x in range(header_size)]

        # extract information from the header
        freq = self._extract_dqt_freq(header)

        if freq > self._extract_dqt_sample_freq(header):
            warnings.warn(
                "The store rate of the DQT data is greater than the sampling "
                "rate.\nData are thus aggregated with the following settigs:\n"
                " - Binning mode: {}".format(
                    self._extract_dqt_bin_mode(header)
                ),
                UserWarning
            )

        data = pd.read_csv(
            input_fname,
            delimiter=',',
            skiprows=(header_size-1),
            header=None,
            names=['datetime','activity', 'light'],
            index_col=0,
            parse_dates=[0],
            date_format="%Y-%m-%d %H:%M:%S",
            dtype=float,
            na_values='x'
        ).asfreq(freq)

        # Convert activity from string to float
        data['activity'] = data['activity'].astype(float)

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = data.index[0]

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = data.index[-1]
            period = stop_time - start_time

        data = data[start_time:stop_time]

        # Light
        if 'light' in data.columns:
            index_light = data.loc[:, 'light']
        else:
            index_light = None

        # call __init__ function of the base class
        super().__init__(
            df=data,
            fpath=input_fname,
            start_time=start_time,
            period=period,
            frequency=freq,
            activity=data['activity'],
            light=index_light.to_frame(name='light') if index_light is not None else None
        )

    @property
    def white_light(self):
        r"""Value of the light intensity (lux)."""
        if self.light is None:
            return None
        else:
            return self.light.get_channel('whitelight')

    @classmethod
    def _match_string(cls, header, match):
        matchings = [s for s in header if match in s]
        if len(matchings) == 0:
            print('No match found for the string: {}.'.format(match))
            return None
        if len(matchings) > 1:
            print('Found multiple matchings for the string: {}'.format(match))
        else:
            return matchings[0]

    @classmethod
    def _extract_dqt_freq(cls, header):
        freqstr = cls._match_string(header=header, match='Store rate')
        return pd.Timedelta(int(freqstr.split(',')[1]), unit='s')

    @classmethod
    def _extract_dqt_sample_freq(cls, header):
        freqstr = cls._match_string(header=header, match='Sample rate')
        return pd.Timedelta(int(freqstr.split(',')[1])/0.1, unit='s')

    @classmethod
    def _extract_dqt_bin_mode(cls, header):
        modestr = cls._match_string(header=header, match='Binning mode')
        return modestr.split(',')[1]


def read_dqt(
    input_fname,
    header_size=15,
    start_time=None,
    period=None
):
    r"""Raw object from .csv file recorded by Daqtometers (Daqtix, Germany)

    Parameters
    ----------
    input_fname: str
        Path to the DQT file.
    header_size: int
        Header size (i.e. number of lines) of the raw data file. Default is 15.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).

    Returns
    -------
    raw : Instance of RawDQT
        An object containing raw DQT data
    """

    return DQT(
        input_fname=input_fname,
        header_size=header_size,
        start_time=start_time,
        period=period
    )