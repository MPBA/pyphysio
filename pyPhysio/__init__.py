"""
Main Package: pyHRV
Contains:
DataSeries: Data structure with cache support (for intermediate values in the indices computation).
Filters: Scripts for cleaning and normalizing data.
Files: Scripts for loading data from and saving to files in various formats (CSV and EXCEL BVP, ECG, IBI..)
+ 2 sub-packages (indexes, windowing)
"""

__author__ = "AleB"
import indexes
import windowing
import Files
import Filters
import PyHRVSettings

__all__ = ['Files', 'PyHRVSettings', 'windowing', 'indexes', 'Filters']
__all__.extend(Filters.__all__)
__all__.extend(Files.__all__)

__all__.extend(indexes.__all__)
__all__.extend(windowing.__all__)

from Files import *
from Filters import *
from PyHRVSettings import *
from indexes import *
from windowing import *
