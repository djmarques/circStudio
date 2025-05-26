"""IO module for reading raw data."""

# Authors: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

# from . import awd
# from . import rpx
# from . import reader

from .base import BaseRaw
from .reader import RawReader

from .reader import read_raw
from .atr import read_raw_atr
from .awd import read_raw_awd

__all__ = [
    "BaseRaw", "RawReader",
    "read_raw",
    "read_raw_atr",
    "read_raw_awd"
]
