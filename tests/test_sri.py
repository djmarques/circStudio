from generate_dataset import generate_series
from generate_dataset import generate_sinewave

import pandas as pd
import circStudio
from pytest import approx

sampling_period = 60
frequency = pd.Timedelta(sampling_period, unit='s')
start_time = '01/01/2018 08:00:00'
N = 20800*sampling_period
period = pd.Timedelta(N, unit='s')

sine_wave = generate_series(
    generate_sinewave(A=100, offset=True, N=N-1),
    start=start_time,
    sampling_period=sampling_period
)

raw_sinewave = circStudio.io.Raw(
    name='raw_sinewave',
    uuid='XXXXXXXX',
    format='CUSTOM',
    axial_mode=None,
    start_time=pd.to_datetime(start_time),
    period=period,
    frequency=frequency,
    data=sine_wave,
    light=None
)


def test_sri_sinewave():

    assert raw_sinewave.SleepRegularityIndex(
        freq='5min',
        algo='Roenneberg'
        ) == approx(100, 1)
