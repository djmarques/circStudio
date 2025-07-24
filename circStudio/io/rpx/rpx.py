import datetime
import io
import pandas as pd
import os
import re
import warnings
from .multilang import fields, columns, day_first
from ..base import Raw


class RPX(Raw):
    """Raw object from .CSV file (recorded by Respironics)

    Parameters
    ----------
    input_fname: str
        Path to the rpx file.
    language: str, optional
        Language of the input csv file.
        Available options are: 'ENG_UK', 'ENG_US', 'FR', 'GER'.
        Default is 'ENG_US'.
    dayfirst: bool, optional
        Whether to interpret the first value of a date as the day.
        If None, rely on the laguage:
        * ENG_US: False
        * ENG_UK or FR or GER: True
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is 'float'.
    delimiter: str, optional
        Delimiter to use when reading the input file.
        Default is ','
    decimal: str, optional
        Decimal character to use when reading the input file.
        Default is '.'
    drop_na: bool, optional
        If set to True, drop epochs where activity is NaN.
        Default is True.
    """

    def __init__(
        self,
        input_fname,
        light_mode=None,
        language='ENG_US',
        dayfirst=None,
        start_time=None,
        period=None,
        data_dtype='float',
        delimiter=',',
        decimal='.',
        drop_na=True
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)
        # check if file exists
        if not os.path.exists(input_fname):
            raise FileNotFoundError(
                "File does not exist: {}.\n Please check.".format(input_fname)
            )

        if language not in fields.keys():
            raise ValueError(
                'Language {0} not supported. Supported languages: {1}'.format(
                    language, '" or "'.join(fields.keys())
                )
            )
        else:
            self._language = language

        # read file header info
        header_offset, data_offset, header, data_available_cols = \
            self._extract_rpx_header_info(input_fname, delimiter)

        # Verify that the input file contains the needed information
        self._check_rpx_header(
            input_fname,
            data_available_cols[2:],
            [columns[self.language][k] for k in ['Date', 'Time', 'Activity']]
        )

        # Unless specified otherwise,
        # set dayfirst as a function of the language
        if dayfirst is None:
            dayfirst = day_first[language]

        # extract information from the header
        start = self._extract_rpx_start_time(header, delimiter, dayfirst)
        frequency = self._extract_rpx_frequency(header, delimiter)

        # read actigraphy data
        with open(input_fname, mode='rb') as file:
            data = file.read()
        data = data.replace(b'\r\r\n', b'\r\n')

        index_data = pd.read_csv(
            io.StringIO(data.decode('utf-8')),
            encoding='utf-8',
            skiprows=header_offset+data_offset+1,
            header=0,
            delimiter=delimiter,
            index_col=0,
            dayfirst=dayfirst,
            usecols=data_available_cols[2:],
            na_values=fields[self.language]['NAN'],
            decimal=decimal,
            dtype={
                columns[self.language]['Activity']: data_dtype,
            }
        )

        index_data['Date_Time'] = pd.to_datetime(
            index_data[columns[self.language]['Date']].str.strip() + ' ' +
            index_data[columns[self.language]['Time']].str.strip(),
            dayfirst=dayfirst
        )
        index_data.set_index('Date_Time', inplace=True)

        # verify that the start time and the first date index matches
        self._check_rpx_start_time(index_data, start)

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = start

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time

        # restrict data to start_time+period (if required)
        index_data = index_data[start_time:stop_time]

        # drop NaN (if required)
        if drop_na:
            index_data.dropna(
                subset=[columns[self.language]['Activity']],
                inplace=True
            )

        # resample the data
        index_data = index_data.asfreq(freq=pd.Timedelta(frequency))

        # Light
        index_light = None
        if light_mode is not None:
            index_light = index_data.loc[:, light_mode]

        # Off-wrist status
        self._off_wrist = self._extract_rpx_data(index_data, "Off_Wrist")
        # Sleep/Wake scoring
        self._sleep_wake = self._extract_rpx_data(index_data, 'Sleep/Wake')

        # Mobility
        self._mobility = self._extract_rpx_data(index_data, 'Mobility')

        # Interval Status
        self._interval_status = self._extract_rpx_data(
            index_data,
            'Interval Status'
        )

        # Sleep/Wake status
        self._sleep_wake_status = self._extract_rpx_data(
            index_data,
            'S/W Status'
        )

        # call __init__ function of the base class
        super().__init__(
            df=index_data,
            fpath=input_fname,
            start_time=start_time,
            period=period,
            frequency=pd.Timedelta(frequency),
            activity=index_data[columns[self.language]['Activity']],
            light=index_light if index_light is not None else None
        )

    @property
    def language(self):
        r"""Language (ENG_UK, FR, GER, etc) used to set up the device"""
        return self._language


    @property
    def off_wrist(self):
        r"""Off-wrist status (1 : device not wrist-worn)"""
        return self._off_wrist

    @property
    def sleep_wake(self):
        r"""Sleep/Wake score (0:sleep, 1:wake)."""
        return self._sleep_wake

    @property
    def mobility(self):
        r"""Mobility score (0:immobile, 1:mobile)."""
        return self._mobility

    @property
    def interval_status(self):
        r"""Interval status (ACTIVE, REST, REST-S or Excluded)."""
        return self._interval_status

    @property
    def sleep_wake_status(self):
        r"""Manually set status (Forced wake, Forced sleep or Excluded)."""
        return self._sleep_wake_status

    def _extract_rpx_header_info(self, fname, delimiter):
        # extract file header and data header
        header = []
        data_available_cols = []
        with open(fname, mode='rb') as file:
            data = file.readlines()
        for header_offset, line in enumerate(data, 1):
            if fields[self.language]['Data'] in line.decode('utf-8'):
                break
            else:
                header.append(line.decode('utf-8'))
        # Read file until the next blank line
        # First, skip blank line after section title
        # next(file)
        for data_offset, line in enumerate(data[header_offset+1:]):
            line_clean = line.replace(b'\r\r\n', b'\r\n')
            if line_clean == b'\r\n':
                break
            else:
                data_available_cols.append(
                    line_clean.decode(
                        'utf-8'
                    ).split(delimiter)[0].strip('"').rstrip(':')
                )

        return header_offset, data_offset, header, data_available_cols

    def _extract_rpx_name(self, header, delimiter):
        for line in header:
            if fields[self.language]['Name'] in line:
                name = re.sub(
                    r'[^\w\s]', '', line.split(delimiter)[1]
                ).strip()
                break
        return name

    def _extract_rpx_uuid(self, header, delimiter):
        for line in header:
            if fields[self.language]['Device_id'] in line:
                uuid = re.sub(r'[\W_]+', '', line.split(delimiter)[1])
                break
        return uuid

    def _extract_rpx_start_time(self, header, delimiter, dayfirst):
        start_time = []
        for line in header:
            if fields[self.language]['Start_date'] in line:
                start_time.append(
                    re.sub(r'[^\d./]+', '', line.split(delimiter)[1])
                )
            elif fields[self.language]['Start_time'] in line:
                start_time.append(
                    re.sub(r'[^\d.:AMP]+', '', line.split(delimiter)[1])
                )
        return pd.to_datetime(
            ' '.join(start_time),
            dayfirst=dayfirst  # (self.language in day_first)
        )

    def _extract_rpx_frequency(self, header, delimiter):
        for line in header:
            if fields[self.language]['Period'] in line:
                frequency = pd.Timedelta(
                    int(re.sub(r'([^\s\w])+', '', line.split(delimiter)[1])
                        .replace('\xa0', ' ').strip()),
                    unit='second'
                )
                break
        return frequency

    def _extract_rpx_data(self, data, column):

        if column in columns[self.language].keys():
            col_name = columns[self.language][column]
        else:
            col_name = None

        return data.loc[:, col_name] if col_name in data.columns else None


    def _extract_light_channel(self, channel):
        if self.light is None:
            return None
        else:
            return self.light.get_channel(columns[self.language][channel])

    def _check_rpx_header(self, fname, cols_available, cols_required):
        if (
            set(cols_available)
            <= set(cols_required)
        ):
            raise ValueError(
                'The data section of the input file {} '.format(fname)
                + 'does not contain the required columns.\n'
                + 'Required columns: {}.\n'.format('", "'.join(
                    cols_required)
                )
                + 'Available columns: {}.\n'.format('", "'.join(
                    cols_available)
                )
            )

    def _check_rpx_start_time(
        self, data, start_time, tolerance=datetime.timedelta(minutes=1)
    ):
        warning_msg = """
- Start time extracted from the header: {0}
- Datetime index of the first data points : {1}
do not match.
Please verify your input file.
"""
        if abs(data.index[0] - start_time) > tolerance:
            warnings.warn(
                warning_msg.format(start_time, data.index[0])
            )


def read_rpx(
    input_fname,
    language='ENG_US',
    light_mode=None,
    dayfirst=None,
    start_time=None,
    period=None,
    data_dtype='float',
    delimiter=',',
    decimal='.',
    drop_na=True
):
    """Reader function for raw Respironics file.

    Parameters
    ----------
    input_fname: str
        Path to the rpx file.
    language: str, optional
        Language of the input csv file.
        Available options are: 'ENG_UK', 'ENG_US', 'FR', 'GER'.
        Default is 'ENG_US'.
    light_mode : str
        Select the light channel for analysis. Default is None.
    dayfirst: bool, optional
        Whether to interpret the first value of a date as the day.
        If None, rely on the laguage:
        * ENG_US: False
        * ENG_UK or FR or GER: True
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    data_dtype: dtype, optional
        The dtype of the raw data.
        Default is 'float'.
    delimiter: str, optional
        Delimiter to use when reading the input file.
        Default is ','
    decimal: str, optional
        Decimal character to use when reading the input file.
        Default is '.'
    drop_na: bool, optional
        If set to True, drop epochs where activity is NaN.
        Default is True.

    Returns
    -------
    raw : Instance of RPX
        An object containing raw RPX data
    """
    return RPX(
        input_fname=input_fname,
        light_mode=light_mode,
        language=language,
        dayfirst=dayfirst,
        start_time=start_time,
        period=period,
        data_dtype=data_dtype,
        delimiter=delimiter,
        decimal=decimal,
        drop_na=drop_na
    )