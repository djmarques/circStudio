import pandas as pd
import numpy as np
import os
import sqlite3
import warnings
from ..base import Raw


class AGD(Raw):
    r"""Raw object from .agd file (recorded by Actigraph)

    Parameters
    ----------
    input_fname: str
        Path to the AGD file.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e., all the data).
    """

    def __init__(
        self,
        input_fname,
        start_time=None,
        period=None
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # create connection to the SQLITE3 .agd file in read-only mode
        connection = sqlite3.connect('file:'+input_fname+'?mode=ro', uri=True)

        # extract header and data size
        settings = pd.read_sql_query(
            "SELECT * FROM settings",
            connection,
            index_col='settingName'
        )

        # extract information from the header
        start = self._to_timestamps(
            int(settings.at['startdatetime', 'settingValue'])
        )
        freq = pd.to_timedelta(
            int(settings.at['epochlength', 'settingValue']),
            unit='s'
        )
        self._model = settings.at['devicename', 'settingValue']

        # extract proximity (wear/no-wear) information
        try:
            capsense = pd.read_sql_query(
                "SELECT state, timeStamp FROM capsense",
                connection,
                index_col='timeStamp'
            ).squeeze()
            # convert index to a date time
            capsense.index = self._to_timestamps(capsense.index)
            # set index frequency
            if 'proximityIntervalInSeconds' in settings.index:
                self.capsense = capsense.asfreq(
                    pd.to_timedelta(
                        int(
                            settings.at[
                                'proximityIntervalInSeconds',
                                'settingValue'
                            ]
                        ),
                        unit='s'
                    )
                )
            elif capsense.index.inferred_freq is not None:
                self.capsense = capsense.asfreq(capsense.index.inferred_freq)
            else:
                warnings.warn(
                    'Acquisition frequency for the wearing sensor data '
                    + '(capsense) could neither be retrieved nor inferred from'
                    + ' the data.\n Use this information at your own risk',
                    UserWarning
                )
                self.capsense = capsense
        except Exception as err:
            # TODO: specialize exception (eg;sqlite3.OperationalError)
            warnings.warn(
                'Could not find wearing sensor data (capsense): {}'.format(
                    err),
                UserWarning
            )
            self.capsense = None

        # extract acceleration and light data
        data = pd.read_sql_query(
            "SELECT * FROM data",
            connection,
            index_col='dataTimestamp'
        )
        # convert index to a date time
        data.index = self._to_timestamps(data.index)
        # set index frequency
        data = data.asfreq(freq=freq)

        # calculate the magnitude of the acceleration vector.
        data['mag'] = np.sqrt(
            data['axis1']*data['axis1']
            + data['axis2']*data['axis2']
            + data['axis3']*data['axis3']
        )

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = data.index[-1]
            period = stop_time - start_time

        data = data.loc[start_time:stop_time]

        # Extract position info:
        self._position = data.filter(regex='incline*', axis='columns')

        # call __init__ function of the base class
        super().__init__(
            df=data,
            fpath=input_fname,
            start_time=start_time,
            period=period,
            frequency=freq,
            activity=data['mag'],
            light=data['lux'].to_frame(name='whitelight') if 'lux' in data.columns else None
        )

        # Close sqlite3 connection
        connection.close()

    def _extract_position(self, column):
        if column in self._position.columns:
            return self._position.loc[:, column]
        else:
            return None

    def _extract_light_channel(self, channel):
        if self.light is None:
            return None
        else:
            return self.light.get_channel(channel)

    @property
    def white_light(self):
        r"""White light levels (in lux.)"""
        return self._extract_light_channel("whitelight")

    @property
    def model(self):
        r"""Model of the device: devicename"""
        return self._model

    @staticmethod
    def _to_timestamps(ticks):
        return pd.to_datetime(
            (ticks/10000000) - 62135596800,
            unit='s'
        )

    @property
    def incline_off(self):
        r"""Hourly positional information: inclineOff """
        return self._extract_position('inclineOff')

    @property
    def incline_standing(self):
        r"""Hourly positional information: inclineStanding """
        return self._extract_position('inclineStanding')

    @property
    def incline_sitting(self):
        r"""Hourly positional information: inclineSitting """
        return self._extract_position('inclineSitting')

    @property
    def incline_lying(self):
        r"""Hourly positional information: inclineLying """
        return self._extract_position('inclineLying')

    def incline_position(self, pos_map=None):
        r"""Reader function for raw AWD file.

        Parameters
        ----------
        pos_map: dict, optional
            Positional information map.
            Can be used to turn positional info (str) into int, for example.
            Set to None to keep original information.
            Default mapping is:
            * 'inclineOff': -1
            * 'inclineLying': 0
            * 'inclineSitting': 1
            * 'inclineStanding': 2

        Returns
        -------
        pos : Pandas.Series
            Time series with positional information.
        """
        if pos_map is None:
            pos_map = {
                'inclineOff': -1,
                'inclineLying': 0,
                'inclineSitting': 1,
                'inclineStanding': 2
            }
        pos = self._position.idxmax(axis=1)
        return pos.map(pos_map) if pos_map is not None else pos


def read_agd(input_fname, start_time=None, period=None):
    r"""Reader function for raw AWD file.

    Parameters
    ----------
    input_fname: str
        Path to the AGD file.
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
    raw : Instance of AWD
        An object containing raw AWD data
    """

    return AGD(
        input_fname=input_fname,
        start_time=start_time,
        period=period
    )