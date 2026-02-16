import os.path as op

import circStudio
import inspect
import pandas as pd


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
atr_path = op.join(data_dir, 'test_sample_atr.txt')

# read AWD with default parameters
rawATR = circStudio.io.read_atr(atr_path)


def test_instance_atr():

    assert isinstance(rawATR, circStudio.io.atr.ATR)

def test_read_raw_atr_start_time():

    assert rawATR.start_time == pd.Timestamp('1918-01-01 09:00:00')


def test_read_raw_atr_data():

    assert len(rawATR.activity) == 4*1440