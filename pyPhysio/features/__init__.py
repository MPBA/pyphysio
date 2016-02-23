# coding=utf-8
"""
features package:
Contains classes with 43 of the main algorithms for the HRV analysis.
They are divided by:
TDIndexes:          Time domain indices,
FDIndexes:          Frequency domain indices,
NonLinearIndexes:   Like e.g. entropy features and Poincaré features.
"""

__author__ = 'AleB'

import TDFeatures
import FDFeatures
import NonLinearFeatures
import CacheOnlyFeatures
from TDFeatures import *
from FDFeatures import *
from NonLinearFeatures import *

__all_indexes__ = []
__all_indexes__.extend(TDFeatures.__all__)
__all_indexes__.extend(FDFeatures.__all__)
__all_indexes__.extend(NonLinearFeatures.__all__)
__all__ = ['TDFeatures', 'FDFeatures', 'NonLinearFeatures', 'CacheOnlyFeatures']
__all__.extend(__all_indexes__)


def get_available_indexes():
    return __all_indexes__


def get_available_online_indexes():
    return filter(lambda x: hasattr(x in vars(), "required_sv"), get_available_indexes())
