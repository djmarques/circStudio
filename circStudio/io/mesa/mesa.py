import pandas as pd
import os
import warnings
from ..base import Raw


class MESA(Raw):
    r"""Raw object from MESA files

    Parameters
    ----------
    input_fname: str
        Path to the MESA file.
    time_origin: datetime-like
        Time origin of the timestamps.
        Required as the MESA files do not contain date informations.
        Default is '2000-01-01'
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    intervals: dict, optional
        Map manually annotated periods to specific scores.
        If set to None, the names of the annotated periods is returned instead.
        Default is {'EXCLUDED': -1, 'ACTIVE': 1, 'REST': 0.5, 'REST-S': 0}.
    check_dayofweek: bool, optional
        If set to True, check if the day of the week reported in the original
        recording is aligned with the reconstructed index.
        Default is False.
    light_mode : str,
        Select light channel for analysis. Available modes for mesa are 'whitelight',
        'redlight', 'greenlight' and 'bluelight'. Default is 'whitelight'.
    """

    def __init__(
        self,
        input_fname,
        time_origin='2000-01-01',
        start_time=None,
        period=None,
        intervals=None,
        check_dayofweek=False,
        light_mode='whitelight'
    ):
        if intervals is None:
            intervals = {'EXCLUDED': -1, 'ACTIVE': 1, 'REST': 0.5, 'REST-S': 0}

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # read file
        data = pd.read_csv(input_fname, index_col='line')

        freq = pd.Timedelta(30, unit='s')

        # day of the week
        self._dayofweek = data['dayofweek']

        # reconstruct MESA datetime index
        date = pd.to_datetime(
            data['daybymidnight'] - 1 + data.loc[1, 'dayofweek'],
            unit='D',
            origin=time_origin
        ).astype(str)
        time = data['linetime']

        index = pd.DatetimeIndex(date + ' ' + time, freq='infer')

        data.set_index(index, inplace=True)

        if check_dayofweek:
            # Shift day of the week to match Pandas' convention (0=Monday, etc)
            dw = self.__dayofweek - 2
            if (data.index.dayofweek - dw.where(dw >= 0, dw + 7)).sum() != 0:
                warnings.warn((
                    "Specified time_origin is such that the day of the week in"
                    " the reconstructed time index is *not* aligned with the"
                    " day of the week reported in the recording."
                ))

        # set start and stop times
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

        # no wear indicator
        self._nowear = data['offwrist']

        # event marker indicator
        self._marker = data['marker']

        # LIGHT
        self._white_light = data['whitelight']

        # wake indicator
        self._wake = data['wake']

        # intervals
        if intervals is not None:
            self._intervals = data['interval'].map(intervals)
        else:
            self._intervals = data['interval']

        # call __init__ function of the base class
        super().__init__(
            df = data,
            fpath=input_fname,
            start_time=start_time,
            period=period,
            frequency=freq,
            activity=data['activity'],
            light=data[light_mode]
        )

    @property
    def marker(self):
        r"""Event marker indicator."""
        return self._marker

    @property
    def wake(self):
        r"""Awake indicator."""
        return self._wake

    @property
    def nowear(self):
        r"""Off-wrist indicator."""
        return self._nowear

    @property
    def intervals(self):
        r"""Interval type (manual rest-activty scoring)."""
        return self._intervals

    @property
    def dayofweek(self):
        r"""Day of the week (1=Sunday, 2=Monday, etc)."""
        return self._dayofweek

    @property
    def white_light(self):
        r"""Value of the white light illuminance in lux."""
        return self._white_light


def read_mesa(
    input_fname,
    time_origin='2000-01-01',
    start_time=None,
    period=None,
    intervals=None,
    check_dayofweek=False,
    light_mode='whitelight'
):
    r"""Reader function for MESA files

    Parameters
    ----------
    input_fname: str
        Path to the ActTrust file.
    time_origin: datetime-like
        Time origin of the timestamps.
        Required as the MESA files do not contain date informations.
        Default is '2000-01-01'
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e., all the data).
    intervals: dict, optional
        Map manually annotated periods to specific scores.
        If set to None, the names of the annotated periods is returned instead.
        Default is {'EXCLUDED': -1, 'ACTIVE': 1, 'REST': 0.5, 'REST-S': 0}.
    check_dayofweek: bool, optional
        If set to True, check if the day of the week reported in the original
        recording is aligned with the reconstructed index.
        Default is False.
    light_mode : str,
        Select light channel for analysis. Available modes for mesa are 'whitelight',
        'redlight', 'greenlight' and 'bluelight'. Default is 'whitelight'.

    Returns
    -------
    raw : Instance of RawMESA
        An object containing raw MESA data
    """
    if intervals is None:
        intervals = {'EXCLUDED': -1, 'ACTIVE': 1, 'REST': 0.5, 'REST-S': 0}

    return MESA(
        input_fname=input_fname,
        time_origin=time_origin,
        start_time=start_time,
        period=period,
        intervals=intervals,
        check_dayofweek=check_dayofweek,
        light_mode=light_mode
    )