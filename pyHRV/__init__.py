__author__ = "AleB"
import Files
import Filters
import PyHRVSettings
import indexes
import DataSeries as _Ds

__all__ = ['Files', 'PyHRVSettings', 'indexes', 'Filters']
__all__.extend(_Ds.__all__)
__all__.extend(Files.__all__)
__all__.extend(PyHRVSettings.__all__)
__all__.extend(indexes.__all__)
__all__.extend(Filters.__all__)


def get_available_indexes():
    return indexes.__all_indexes__


from DataSeries import *
from Files import *
from PyHRVSettings import *
from indexes import *

del _Ds
