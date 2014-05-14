__author__ = 'AleB'
__all__ = ["RRFilters"]
import numpy as np

from pyHRV.DataSeries import DataSeries


class DataAnalysis(object):
    pass


class RRFilters(DataAnalysis):
    """ Static class containing methods for filtering RR intervals data. """

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries object to filter
        :return: DataSeries object filtered
        """
        assert isinstance(series, DataSeries)
        return series

    @staticmethod
    def normalize_mean(series):
        """PSD estimation method:
            1) RR-mean"""
        assert isinstance(series, DataSeries)
        return DataSeries(series - np.mean(series))

    @staticmethod
    def normalize_mean_sd(series):
        """PSD estimation method:
            2) (RR-mean)/sd"""
        assert isinstance(series, DataSeries)
        return DataSeries((series - np.mean(series)) / np.std(series))

    @staticmethod
    def normalize_min(series):
        """PSD estimation method:
            3) RR - min"""
        assert isinstance(series, DataSeries)
        return DataSeries(series - np.min(series))

    @staticmethod
    def normalize_max_min(series):
        """PSD estimation method:
            4) (RR-min)/(max-min)"""
        assert isinstance(series, DataSeries)
        return DataSeries((series - np.mean(series)) / (np.max(series) - np.min(series)))

    @staticmethod
    def normalize_rr_all_mean_calm(series, rr_all, mean_calm_msec):
        """PSD estimation method:
            5) RR_ALL*RR/meanCALM"""
        assert isinstance(series, DataSeries)
        return DataSeries(rr_all * series - mean_calm_msec)
