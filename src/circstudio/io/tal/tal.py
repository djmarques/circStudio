import pandas as pd
import os
import re

from ..base import Raw



class TAL(Raw):
    r"""Raw object from .txt file recorded by Tempatilumi (CE Brasil)

    Parameters
    ----------
    input_fname: str
        Path to the Tempatilumi file.
    sep: str, optional
        Delimiter to use.
        Default is ";".
    frequency: str, optional
        Sampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        If None, the sampling frequency is inferred from the data. Otherwise,
        the data are resampled to the specified frequency.
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    encoding: str, optional
        Encoding to use for UTF when reading the file.
        Default is "latin-1".
    """

    def __init__(
        self,
        input_fname,
        sep=';',
        frequency=None,
        start_time=None,
        period=None,
        encoding='latin-1'
    ):

        # get absolute file path
        input_fname = os.path.abspath(input_fname)

        # extract header and data
        with open(input_fname, encoding=encoding) as f:
            header = []
            pos = 0
            cur_line = f.readline()
            while not cur_line.startswith(sep.join(["Data", " Hora"])):
                header.append(cur_line)
                pos = f.tell()
                cur_line = f.readline()
            f.seek(pos)
            index_data = pd.read_csv(
                filepath_or_buffer=f,
                skipinitialspace=True,
                sep=sep,
                index_col=False,
            )
        # Strip whitespace and combine date and time
        index_data['Date_Time'] = pd.to_datetime(
            index_data['Data'].str.strip() + ' ' + index_data['Hora'].str.strip(),
            format='%Y/%m/%d %H:%M:%S'
        )

        # Set Date_Time as index
        index_data.set_index('Date_Time', inplace=True)

        # Check column names
        # Evento	 Temperatura	 Luminosidade	 Atividade
        if 'Atividade' not in index_data.columns:
            raise ValueError(
                'The activity counts cannot be found in the data.\n'
                'Column name in file header should be "Atividade".'
            )

        self._temperature = self._extract_from_data(
            index_data, 'Temperatura'
        )

        self._events = self._extract_from_data(
            index_data, 'Evento'
        )

        if frequency is not None:
            index_data = index_data.resample(frequency).sum()
            freq = pd.Timedelta(frequency)
        elif not index_data.index.inferred_freq:
            raise ValueError(
                'The sampling frequency:\n'
                '- cannot be inferred from the data\n'
                'AND\n'
                '- is NOT explicity passed to the reader function.\n'
            )
        else:
            index_data = index_data.asfreq(index_data.index.inferred_freq)
            freq = pd.Timedelta(index_data.index.freq)

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        else:
            start_time = index_data.index[0]

        if period is not None:
            period = pd.Timedelta(period)
            stop_time = start_time+period
        else:
            stop_time = index_data.index[-1]
            period = stop_time - start_time

        index_data = index_data[start_time:stop_time]

        # Light
        index_light = self._extract_from_data(index_data, 'Luminosidade')

        # call __init__ function of the base class
        super().__init__(
            df=index_data,
            fpath=input_fname,
            start_time=start_time,
            period=period,
            frequency=freq,
            activity=index_data['Atividade'],
            light=index_light.to_frame(name='light'),
        )

    @property
    def white_light(self):
        r"""Value of the light intensity in µw/cm²."""
        if self.light is None:
            return None
        else:
            return self.light.get_channel("whitelight")

    @property
    def temperature(self):
        r"""Value of the temperature (in ° C)."""
        return self._temperature

    @property
    def events(self):
        r"""Event markers."""
        return self._events

    @classmethod
    def _extract_tal_uuid(cls, header):
        match = re.search(r'Série: (\d+)', ''.join(header))
        if not match:
            raise ValueError('UUID cannot be extracted from the file header.')
        return match.group(1)

    @classmethod
    def _extract_from_data(cls, data, key):
        if key in data.columns:
            return data[key]
        else:
            return None


def read_tal(
    input_fname,
    sep=';',
    frequency=None,
    start_time=None,
    period=None,
    encoding='latin-1'
):
    r"""Raw object from .txt file recorded by Tempatilumi (CE Brasil)

    Parameters
    ----------
    input_fname: str
        Path to the Tempatilumi file.
    sep: str, optional
        Delimiter to use.
        Default is ';'.
    frequency: str, optional
        Sampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        If None, the sampling frequency is inferred from the data. Otherwise,
        the data are resampled to the specified frequency.
        Default is None.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    encoding: str, optional
        Encoding to use for UTF when reading the file.
        Default is 'latin-1'.

    Returns
    -------
    raw : Instance of TAL
        An object containing raw TAL data
    """

    return TAL(
        input_fname=input_fname,
        sep=sep,
        frequency=frequency,
        start_time=start_time,
        period=period,
        encoding=encoding
    )