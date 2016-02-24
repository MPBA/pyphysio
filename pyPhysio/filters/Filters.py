# coding=utf-8
from pyphysio.pyPhysio.BaseAlgorithm import Cache

__author__ = 'AleB'
__all__ = ["Filters"]

import numpy as np

from ..BaseFilter import Filter
from ..PhUI import PhUI
from .. import Signal
from ..features.TDFeatures import Mean, SD


"""
Filters are blocks that take as input a SIGNAL and gives as output another SIGNAL of the SAME NATURE.
"""


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
    def algorithm(cls, data, params):
        if 'norm_type' not in params:
            PhUI.i("Assuming Mean normalization.")
            return Normalize._mean(data)
        else:
            if params['norm_type'] == Normalize.Types.Mean:
                return Normalize._mean(data)
            elif params['norm_type'] == Normalize.Types.MeanSd:
                return Normalize._mean_sd(data)
            elif params['norm_type'] == Normalize.Types.Min:
                return Normalize._min(data)
            elif params['norm_type'] == Normalize.Types.MaxMin:
                return Normalize._max_min(data)
            elif params['norm_type'] == Normalize.Types.Custom:
                assert 'norm_bias' in params, "For the custom normalization the parameter norm_bias is needed."
                assert 'norm_range' in params, "For the custom normalization the parameter norm_range is needed."
                return Normalize._custom(data, params['norm_bias'], params['norm_range'])
            else:
                PhUI.w("Unrecognized normalization type in 'norm_type'.")

    @staticmethod
    def _mean(series):
        """
        Normalizes the series removing the mean (val-mean)
        """
        assert isinstance(series, Signal)
        series.set_values(series.get_values() - Mean.get(series))
        Cache.cache_clear(series)
        return series

    @staticmethod
    def _mean_sd(series):
        """
        Normalizes the series removing the mean and dividing by the standard deviation (val-mean)/sd
        """
        assert isinstance(series, Signal)
        series.set_values(series.get_values() - Mean.get(series) / SD.get(series))
        Cache.cache_clear(series)
        return series

    @staticmethod
    def _min(series):
        """
        Normalizes the series removing the minimum value (val-min)
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, Signal)
        series.set_values(series.get_values() - np.min(series.get_values()))
        Cache.cache_clear(series)
        return series

    @staticmethod
    def _max_min(series):
        """
        Normalizes the series removing the min value and dividing by the range width (val-min)/(max-min)
        @param series: TimeSeries
        @type series: TimeSeries
        @return: Filtered TimeSeries
        @rtype: TimeSeries
        """
        assert isinstance(series, Signal)
        series.set_values(series.get_values() / (np.max(series.get_values()) - np.min(series.get_values())))
        Cache.cache_clear(series)
        return series

    @staticmethod
    def _custom(series, bias, normalization_range):
        """
        Normalizes the series considering two factors ((val-par1)/par2)
        """
        assert isinstance(series, Signal)
        series.set_values((series.get_values() - bias) / normalization_range)
        Cache.cache_clear(series)
        return series
