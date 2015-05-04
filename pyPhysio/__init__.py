"""
Main Package: pyHRV
Contains:
DataSeries: Data structure with cache support (for intermediate values in the indices computation).
Filters: Scripts for cleaning and normalizing data.
Files: Scripts for loading data from and saving to files in various formats (CSV and EXCEL BVP, ECG, IBI..)
+ 2 sub-packages (indexes, windowing)
"""

__author__ = "AleB"
import features
import windowing
import Files
import Filters
import PyHRVSettings

__all__ = ['Files', 'PyHRVSettings', 'windowing', 'features', 'Filters']
__all__.extend(Filters.__all__)
__all__.extend(Files.__all__)

__all__.extend(features.__all__)
__all__.extend(windowing.__all__)

from Files import *
from Filters import *
from PyHRVSettings import *
from features import *
from windowing import *


def compute(data=None, features_list=None, params=None, windows=None):
    from pandas import TimeSeries
    if data is None or not isinstance(data, TimeSeries):
        print("The first parameter must be a pandas.TimeSeries containing the data to analyze.")
    elif features_list is None or len(features_list) == 0:
        print("The second parameter must be a list containing the features to extract e.g. [" +
              Mean.__name__ + ", " + SD.__name__ + ", " + NN50.__name__ + "].")
    else:
        if windows is None:
            if len(features_list) == 1:
                return features_list[0](data, params).value
            else:
                windows = CollectionWinGen([windowing.Window(0, len(data), data)])
        return WindowsIterator(data, windows, features_list, params).compute_all()
