# coding=utf-8
from __future__ import division

__author__ = "AleB"

from features import *
from filters import *
from segmentation import *
from SegmentationBase import Segment
from WindowsIterator import WindowsIterator


class PhUI(object):
    @staticmethod
    def a(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def i(mex):
        PhUI.p(mex, '', 35)

    @staticmethod
    def o(mex):
        PhUI.p(mex, '', 31)

    @staticmethod
    def w(mex):
        PhUI.p(mex, 'Warning: ', 33)

    @staticmethod
    def p(mex, lev, col):
        print(">%s\x1b[%dm%s\x1b[39m" % (lev, col, mex))


# TODO uses pandas
def create_series(data, time_bias_in_seconds=0, time_scale_to_seconds=1.0, index=None, metadata=None):
    from pandas import Series, Int64Index, Float64Index
    from numpy import cumsum, array

    if isinstance(data, Series):
        PhUI.o("data is already a Series, pass your_series.values as the data parameter or it is useless to use "
               "this function.")
        return data
    elif not hasattr(data, "__len__"):
        PhUI.o("data must have a length, its type is %s instead." % type(data))
    elif not (index is None) and not hasattr(index, "__len__"):
        PhUI.o("index must have a length, its type is %s instead." % type(index))
    elif not (index is None) and not (len(index) == len(data)):
        PhUI.o("index must be of the same length of data (%d and %d)." % (len(data), len(index)))
    elif not isinstance(time_scale_to_seconds, int) and not isinstance(time_scale_to_seconds, float):
        PhUI.o("The time scale (time_scale_to_seconds) must be a number.")
    elif not isinstance(time_bias_in_seconds, int) and not isinstance(time_scale_to_seconds, float):
        PhUI.o("The time bias (time_bias_in_seconds) must be an integer or a float.")
    else:
        if metadata is not None and not isinstance(metadata, dict):
            PhUI.o("Metadata must be a dictionary e.g. {'sampling_freq': 10}")
        else:
            data = array(data)
            if index is None:
                PhUI.w("assuming data values as intervals.")
                PhUI.i("Use time_scale_to_seconds and time_bias_in_seconds parameters to better convert them.")
                ret = Series(data * time_scale_to_seconds, cumsum(data))
            else:
                ret = Series(data, index)
            if ret.index.is_mixed():
                PhUI.o("Every index value must of the same type.")
            elif not ret.index.is_unique:
                PhUI.o("Every index value must appear once, duplicated values found.")
            else:
                if not ret.index.is_all_dates:
                    PhUI.i("Converting index to datetime with scale %fs and bias %fs."
                           % (time_scale_to_seconds, time_bias_in_seconds))
                    tmp = (ret.index * time_scale_to_seconds + time_bias_in_seconds) * (10 ** 9)
                    PhUI.a(isinstance(tmp, Int64Index) or isinstance(tmp, Float64Index), "Index is a %s" % type(tmp))
                    assert isinstance(tmp, Int64Index) or isinstance(tmp, Float64Index)
                    ret.index = tmp.to_datetime()
                if not ret.index.is_monotonic:
                    PhUI.w("index is not monotonic, sorting.")
                    ret = ret.sort_index()
                return ret
    return None


# TODO uses pandas
def create_labels_series(times_or_data_series=None, labels=None, time_bias_in_seconds=0, time_scale_to_seconds=1,
                         is_polling=None):
    from pandas import Series

    if times_or_data_series is None:
        PhUI.o("The first parameter must be a list of times [s] or the Series of the data "
               "with the times which the labels are related to.")
    elif labels is None:
        PhUI.o("The second parameter must be a list/array of labels names/info.")
    else:
        if is_polling is None:
            is_polling = len(labels) == len(times_or_data_series)
        PhUI.i("Assuming label mode polling.")
        if isinstance(times_or_data_series, Series):
            PhUI.i("Taking times from a Series.")
            return i_create_labels_series(times_or_data_series.index, labels, time_bias_in_seconds=time_bias_in_seconds,
                                          time_scale_to_seconds=time_scale_to_seconds, is_polling=is_polling)
        else:
            PhUI.i("Taking times from a %s." % type(times_or_data_series))
            return i_create_labels_series(times_or_data_series, labels, time_bias_in_seconds=time_bias_in_seconds,
                                          time_scale_to_seconds=time_scale_to_seconds, is_polling=is_polling)


# TODO uses pandas
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


# TODO uses pandas
def compute(data=None, features_list=None, params=None, windows=None):
    from pandas import Series
    from pyPhysio.BaseFeature import Feature

    if data is None or not isinstance(data, Series):
        PhUI.o("The first parameter must be a pandas.Series containing the data to analyze.")
    else:
        if type(features_list) is type or isinstance(features_list, Feature):
            features_list = [features_list]
        if type(features_list) is not list or len(features_list) == 0:
            PhUI.o("The second parameter must be a list containing the features to extract e.g. [" +
                   Mean.__name__ + ", " + SD.__name__ + ", " + NN50.__name__ + "].")
        else:
            if windows is None:
                if len(features_list) == 1:  # one feature manually
                    return features_list[0](data, params).value
                else:  # auto-create win
                    windows = ExistingSegments([Segment(0, len(data), data)])  # TODO 3: test with the new segmentation
                    # use iterator
            return WindowsIterator(data, windows, features_list, params).compute_all()
