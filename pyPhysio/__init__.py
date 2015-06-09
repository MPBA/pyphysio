from __future__ import division
__author__ = "AleB"
import features
import windowing
import Files
import Filters

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


def py_physio_log(mex, lev='', col=31):
    print(">%s\x1b[%dm%s\x1b[39m" % ("%s: " % lev if lev != '' else lev, col, mex))


def create_series(data, time_bias_in_seconds=0, time_scale_to_seconds=1.0, index=None, metadata=None):
    from pandas import Series, Int64Index, Float64Index
    from numpy import cumsum, array

    if not hasattr(data, "__len__"):
        py_physio_log("data must have a length, its type is %s instead." % type(data))
    elif not (index is None) and not hasattr(index, "__len__"):
        py_physio_log("index must have a length, its type is %s instead." % type(index))
    elif not (index is None) and not (len(index) == len(data)):
        py_physio_log("index must be of the same length of data (%d and %d)." % (len(data), len(index)))
    elif not isinstance(time_scale_to_seconds, int) and not isinstance(time_scale_to_seconds, float):
        py_physio_log("The time scale (time_scale_to_seconds) must be a number.")
    elif not isinstance(time_bias_in_seconds, int) and not isinstance(time_scale_to_seconds, float):
        py_physio_log("The time bias (time_bias_in_seconds) must be an integer or a float.")
    else:
        if metadata is not None and not isinstance(metadata, dict):
            py_physio_log("Metadata must be a dictionary e.g. {'sampling_freq': 10}")
        else:
            data = array(data)
            if index is None:
                py_physio_log("assuming data values as intervals.", 'Warning', 33)
                py_physio_log("Use time_scale_to_seconds and time_bias_in_seconds to better convert them.", col=35)
                ret = Series(data * time_scale_to_seconds, cumsum(data))
            else:
                ret = Series(data, index)
            if ret.index.is_mixed():
                py_physio_log("Every index value must of the same type.")
            elif not ret.index.is_unique:
                py_physio_log("Every index value must appear once, duplicated values found.")
            else:
                if not ret.index.is_all_dates:
                    py_physio_log("Converting index to datetime with scale %fs and bias %fs."
                                  % (time_scale_to_seconds, time_bias_in_seconds), col=35)
                    tmp = (ret.index * time_scale_to_seconds + time_bias_in_seconds) * (10**9)
                    assert isinstance(tmp, Int64Index) or isinstance(tmp, Float64Index), "Index is a %s" % type(tmp)
                    ret.index = tmp.to_datetime()
                if not ret.index.is_monotonic:
                    py_physio_log("index is not monotonic, sorting.", 'Warning', 33)
                    ret = ret.sort_index()
                return ret
    return None


def create_labels_series(times_or_data_series=None, labels=None, time_bias_in_seconds=0, time_scale_to_seconds=1,
                         is_polling=True):
    from pandas import Series
    if times_or_data_series is None:
        py_physio_log("The first parameter must be a list of times [s] or the Series of the data "
                      "with the times which the labels are related to.")
    elif labels is None:
        py_physio_log("The second parameter must be a list/array of labels names/info.")
    elif isinstance(times_or_data_series, Series):
        py_physio_log("Taking times from a Series.", col=35)
        i_create_labels_series(times_or_data_series.index, labels, time_bias_in_seconds=time_bias_in_seconds,
                               time_scale_to_seconds=time_scale_to_seconds, is_polling=is_polling)
    else:
        py_physio_log("Taking times from a %s." % type(times_or_data_series), col=35)
        i_create_labels_series(times_or_data_series, labels, time_bias_in_seconds=time_bias_in_seconds,
                               time_scale_to_seconds=time_scale_to_seconds, is_polling=is_polling)


def i_create_labels_series(times, labels, time_bias_in_seconds=0, time_scale_to_seconds=1, is_polling=True):
    x = create_series(labels, time_bias_in_seconds, time_scale_to_seconds, times)
    if is_polling:
        from pandas import Series
        w = None
        kk = []
        vv = []
        for i in xrange(len(x)):
            if w != x[i]:
                kk.append(x.index[i])
                vv.append(x[i])
                w = x[i]
        return Series(vv, index=kk)
    else:
        return create_series(labels, time_bias_in_seconds, time_scale_to_seconds, times)


def compute(data=None, features_list=None, params=None, windows=None):
    from pandas import Series

    if data is None or not isinstance(data, Series):
        py_physio_log("The first parameter must be a pandas.Series containing the data to analyze.")
    elif features_list is None or len(features_list) == 0:
        py_physio_log("The second parameter must be a list containing the features to extract e.g. [" +
                      Mean.__name__ + ", " + SD.__name__ + ", " + NN50.__name__ + "].")
    else:
        if windows is None:
            if len(features_list) == 1:  # one feature manually
                return features_list[0](data, params).value
            else:  # auto-create win
                windows = ExistingWindows([windowing.Window(0, len(data), data)])  # TODO 3: test with the new windowing
                # use iterator
        return WindowsIterator(data, windows, features_list, params).compute_all()
