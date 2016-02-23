# coding=utf-8
__author__ = 'AleB'
__all__ = ["Filters"]

import numpy as np

from ..BaseFilter import Filter

from .. import PhUI
from ..features.TDFeatures import Mean, SD


class Normalize(Filter):
    """
    Normalizes the series removing the mean (val-mean)
    """

    def get_used_params(self):
        return ['norm_type']

    class Types(object):
        Mean = 0
        MeanSd = 1
        Min = 2
        MaxMin = 3
        Custom = 4

    @classmethod
    def algorithm(cls, series, params):
        if 'norm_type' in params:
            if params['norm_type'] == Normalize.Types.Mean:
                return Normalize._mean(series)
            elif params['norm_type'] == Normalize.Types.MeanSd:
                return Normalize._mean_sd(series)
            elif params['norm_type'] == Normalize.Types.Min:
                return Normalize._min(series)
            elif params['norm_type'] == Normalize.Types.MaxMin:
                return Normalize._max_min(series)
            elif params['norm_type'] == Normalize.Types.Custom:
                assert 'norm_bias' in params, "For the custom normalization the parameter norm_bias is needed."
                assert 'norm_factor' in params, "For the custom normalization the parameter norm_factor is needed."
                return Normalize._custom(series, params['norm_bias'], params['norm_factor'])
            else:
                PhUI.w("Unrecognized normalization type in 'norm_type'.")
        else:
            PhUI.i("Assuming Mean normalization.")
        return Normalize._mean(series)

    @staticmethod
    def _mean(series):
        """
        Normalizes the series removing the mean (val-mean)
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        return TimeSeries(series - Mean.get(series))

    @staticmethod
    def _mean_sd(series):
        """
        Normalizes the series removing the mean and dividing by the standard deviation (val-mean)/sd
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, TimeSeries)
        return TimeSeries((series - Mean.get(series)) / SD.get(series))

    @staticmethod
    def _min(series):
        """
        Normalizes the series removing the minimum value (val-min)
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, TimeSeries)
        return TimeSeries(series - np.min(series))

    @staticmethod
    def _max_min(series):
        """
        Normalizes the series removing the min value and dividing by the range width (val-min)/(max-min)
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, TimeSeries)
        return TimeSeries(series / (np.max(series) - np.min(series)))

    @staticmethod
    def _custom(series, par1, par2):
        """
        Normalizes the series considering two factors ((val-par1)/par2)
        @param par1: a scale for each sample
        @param par2: second parameter: average calm-state expected bpm
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, TimeSeries)
        return TimeSeries((par1 - series) / par2)


class Outliers(Filter):
    # TODO: Only works with RR-series

    def get_used_params(self):
        return ['outliers_last', 'outliers_min', 'outliers_max', 'outliers_win_len']

    def algorithm(self, data, params):
        return Outliers._filter_outliers(data, params['outliers_last'], params['outliers_min'], params['outliers_max'],
                                         params['outliers_win_len'])

    @staticmethod
    def _filter_outliers(series, last=13, min_bpm=24, max_bpm=198, win_length=50):
        """
        Removes outliers from RR series.
        @param series: TimeSeries
        @type series: TimeSeries
        @param last: last percentage
        @type last: float
        @param min_bpm: minimum bpm to be considered valid
        @type min_bpm: float
        @param max_bpm: maximum bpm to be considered valid
        @type max_bpm: float
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, TimeSeries)
        new_series = np.array(series)
        max_rr = 60000 / min_bpm
        min_rr = 60000 / max_bpm

        # threshold initialization
        u_last = last  # 13%
        u_mean = 1.5 * u_last  # 19%

        index = 1  # discard the first
        var_pre = 100 * abs((new_series[1] - new_series[0]) / new_series[0])
        while index < len(new_series) - 1:  # pre-last
            v = new_series[max(index - win_length, 0):index]  # last win_length values avg
            m = np.mean(v)
            var_next = 100 * abs((new_series[index + 1] - new_series[index]) / new_series[index + 1])
            var_mean = 100 * abs((new_series[index] - m) / m)

            if (((var_pre < u_last) |  # last var
                    (var_next < u_last) |  # last var
                    (var_mean < u_mean)) &  # avg var
                    (new_series[index] > min_rr) & (new_series[index] < max_rr)):  # ok values
                index += 1  # ok
            else:
                new_series = np.delete(new_series, index)
        return TimeSeries(new_series)


class PeakDetection(Filter):
    @classmethod
    def algorithm(cls, data, params):
        PhUI.a('pkd_delta' in params, "The parameter 'pkd_delta is needed in order to perform a peak detection.")
        try:
            return PeakDetection._peak_detection(data, params['pkd_delta'], params['pkd_times'])
        except ValueError as e:
            PhUI.a(False, e.message)

    @staticmethod
    def _peak_detection(data, delta, times=None):
        """
        Detects peaks in the signal assuming the specified delta.
        @param data: Array of the values.
        @param delta: Differential threshold.
        @param times: Array of the times.
        @return: Tuple of lists: (max_t, min_t, max_v, min_v)
        @rtype: (list, list, list, list)
        @raise ValueError:
        """
        max_i = []
        min_i = []
        max_v = []
        min_v = []

        if times is None:
            times = np.arange(len(data))

        data = np.asarray(data)

        if len(data) != len(times):
            raise ValueError('Input vectors v and x must have same length')

        if not np.isscalar(delta):
            raise ValueError('Input argument delta must be a scalar')

        if delta <= 0:
            raise ValueError('Input argument delta must be positive')

        mn, mx = np.Inf, -np.Inf
        mn_pos, mx_pos = np.NaN, np.NaN

        look_for_max = True

        for i in np.arange(len(data)):
            this = data[i]
            if this > mx:
                mx = this
                mx_pos = times[i]
            if this < mn:
                mn = this
                mn_pos = times[i]

            if look_for_max:
                if this < mx - delta:
                    max_v.append(mx)
                    max_i.append(mx_pos)
                    mn = this
                    mn_pos = times[i]
                    look_for_max = False
            else:
                if this > mn + delta:
                    min_v.append(mn)
                    min_i.append(mn_pos)
                    mx = this
                    mx_pos = times[i]
                    look_for_max = True

        return max_i, min_i, max_v, min_v