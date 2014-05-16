__author__ = 'AleB'

# Will make every index and index directory accessible from the pyHRV namespace

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
__all_indexes__.extend(NonLinearIndexes.__all__)
__all__ = ['TDIndexes', 'FDIndexes', 'NonLinearIndexes']
__all__.extend(__all_indexes__)
