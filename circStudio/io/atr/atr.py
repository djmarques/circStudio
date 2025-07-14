import pandas as pd
import re
from ..base import Raw


class ATR(Raw):
    r"""Adaptor class for ActTrust (Condor Instruments) actigraphy logs

    Parameters
    ----------
    input_fname: str
        Path to the ActTrust file.
    activity_mode: str, optional
        Activity sampling mode.
        Available modes are: Proportional Integral Mode (PIM),  Time Above
        Threshold (TAT) and Zero Crossing Mode (ZCM).
        Default is PIM.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e all the data).
    """

    _default_modes = ["PIM", "PIMn", "TAT", "TATn", "ZCM", "ZCMn"]

    def __init__(self,
                 input_fname,
                 activity_mode='PIM',
                 light_mode='LIGHT',
                 start_time=None,
                 skip_rows=0,
                 period=None):


        # extract header and data size
        header = {}
        with open(input_fname) as fp:
            lines = fp.readlines()

            if skip_rows is not None:
                lines = lines[skip_rows:]

            if not lines:
                raise ValueError("Input file has no lines")

            first_line = lines[0]
            if not re.match(r"\+-*\+ \w+ \w+ \w+ \+-*\+", first_line):
                raise ValueError("The input file does not seem to contain the usual header.")
            for line in lines[1:]:
                if '+-------------------' in line:
                    break
                else:
                    chunks = line.strip().split(' : ')
                    if chunks:
                        header[chunks[0]] = chunks[1:]
        if not header:
            raise ValueError("The input file does not contain a header.")

        #Extract information from the header
        freq = pd.Timedelta(int(header['INTERVAL'][0]), unit='s')

        # Create a DataFrame containing actigraphy data
        data = pd.read_csv(input_fname,
            skiprows=len(header)+2+skip_rows,
            sep=';',
            parse_dates=True,
            dayfirst=True,
            index_col=[0]
        ).resample(freq).sum()

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

        # Create a new Raw object using the information extracted
        super().__init__(
            df=data,
            start_time=start_time,
            period=period,
            frequency=freq,
            activity=data[activity_mode],
            light=data[light_mode]
        )

def read_atr(input_fname, activity_mode='PIM', light_mode='LIGHT', start_time=None, period=None, skip_rows=0):
    r"""Reader function for .txt file recorded by ActTrust (Condor Instruments)

    Parameters
    ----------
    skip_rows
    light_mode: str, optional.
        Light sampling mode. By default, it uses the LIGHT channel (the other channels are not in lux).
    input_fname: str
        Path to the ActTrust file.
    activity_mode: str, optional
        Activity sampling mode.
        Available modes are: Proportional Integral Mode (PIM),  Time Above
        Threshold (TAT) and Zero Crossing Mode (ZCM).
        Default is PIM.
    start_time: datetime-like, optional
        Read data from this time.
        Default is None.
    period: str, optional
        Length of the read data.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is None (i.e., all the data).

    Returns
    -------
    raw : Instance of RawATR
        An object containing raw ATR data
    """
    return ATR(input_fname=input_fname,
               activity_mode=activity_mode,
               light_mode=light_mode,
               start_time=start_time,
               period=period,
               skip_rows=skip_rows)
