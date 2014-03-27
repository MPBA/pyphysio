__author__ = "AleB"
import Files
import PyHRVSettings
import Indexes
import DataSeries as _Ds

__all__ = ['Files', 'PyHRVSettings', 'Indexes']
__all__.extend(_Ds.__all__)
__all__.extend(Files.__all__)
__all__.extend(PyHRVSettings.__all__)
__all__.extend(Indexes.__all__)


def get_available_indexes():
    return Indexes.__all_indexes__


from DataSeries import *
from Files import *
from PyHRVSettings import *
from Indexes import *
del _Ds
