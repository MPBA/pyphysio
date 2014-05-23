__author__ = 'AleB'
__all__ = ["RRFilters"]

import numpy as np

from pyHRV.DataSeries import DataSeries


class RRFilters(object):
    """ Static class containing methods for filtering RR intervals data. """

    @staticmethod
    def normalize_mean(series):
        """Normalizes the series removing the mean (RR-mean)"""
        assert isinstance(series, DataSeries)
        return DataSeries(series - np.mean(series))

    @staticmethod
    def normalize_mean_sd(series):
        """Normalizes the series removing the mean and dividing by the standard deviation (RR-mean)/sd"""
        assert isinstance(series, DataSeries)
        return DataSeries((series - np.mean(series)) / np.std(series) + np.mean(series))

    @staticmethod
    def normalize_min(series):
        """Normalizes the series removing the minimum value (RR-min)"""
        assert isinstance(series, DataSeries)
        return DataSeries(series - np.min(series) + np.mean(series))

    @staticmethod
    def normalize_max_min(series):
        """Normalizes the series removing the mean and dividing by the range width (RR-mean)/(max-min)"""
        assert isinstance(series, DataSeries)
        return DataSeries((series - np.mean(series)) / (np.max(series) - np.min(series)) + np.mean(series))

    @staticmethod
    def normalize_rr_all_mean_calm(series, param, mean_calm_milliseconds):
        """Normalizes the series scaling by two factors ((PAR*RR)/meanCALM)"""
        assert isinstance(series, DataSeries)
        return DataSeries(param * series - mean_calm_milliseconds)

    @staticmethod
    def filter_outliers(series, last=13, min_bpm=24, max_bpm=198, win_length=50):
        """Removes outliers from RR series"""
        assert isinstance(series, DataSeries)
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
        return DataSeries(new_series)

    @staticmethod
    def example_filter(series):
        """ Example filter method, does nothing
        :param series: DataSeries
        :return: DataSeries
        """
        assert isinstance(series, DataSeries)
        return series
