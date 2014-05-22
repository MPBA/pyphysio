# coding=utf-8
"""
indexes package:
Contains classes with 43 of the main algorithms for the HRV analysis.
They are divided by:
TDIndexes:          Time domain indices,
FDIndexes:          Frequency domain indices,
NonLinearIndexes:   Like e.g. entropy indexes and Poincaré indexes.
"""

__author__ = 'AleB'

import TDIndexes
import FDIndexes
import NonLinearIndexes
import BaseIndexes
from TDIndexes import *
from FDIndexes import *
from NonLinearIndexes import *

__all_indexes__ = []
__all_indexes__.extend(TDIndexes.__all__)
__all_indexes__.extend(FDIndexes.__all__)
# __all_indexes__.extend(NonLinearIndexes.__all__)
__all__ = ['TDIndexes', 'FDIndexes', 'NonLinearIndexes']
__all__.extend(__all_indexes__)
