"""IO module for reading raw data."""

# Authors: Grégory Hammad <gregory.hammad@uliege.be>
#
# License: BSD (3-clause)

# from . import awd
# from . import rpx
# from . import reader

from .base import Raw

from .atr import read_raw_atr
from .awd import read_raw_awd

__all__ = [
    "Raw",
    "read_raw_atr",
    "read_raw_awd"
]

#Test