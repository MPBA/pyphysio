__author__ = 'AleB'
__all__ = ["IBIFilters"]

import numpy as np

from pyHRV.DataSeries import DataSeries
from pyHRV.indexes.TDFeatures import Mean, SD


class IBIFilters(object):
    """
    Static class containing methods for filtering IBI data.
    """

    @staticmethod
    def normalize_mean(series):
        """
        Normalizes the series removing the mean (RR-mean)
        @param series: DataSeries
        @type series: DataSeries
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return DataSeries(series - Mean.get(series))

    @staticmethod
    def normalize_mean_sd(series):
        """
        Normalizes the series removing the mean and dividing by the standard deviation (IBI-mean)/sd
        @param series: DataSeries
        @type series: DataSeries
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return DataSeries((series - Mean.get(series)) / SD.get(series))

    @staticmethod
    def normalize_min(series):
        """
        Normalizes the series removing the minimum value (IBI-min)
        @param series: DataSeries
        @type series: DataSeries
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return DataSeries(series - np.min(series))

    @staticmethod
    def normalize_max_min(series):
        """
        Normalizes the series removing the min value and dividing by the range width (IBI-min)/(max-min)
        @param series: DataSeries
        @type series: DataSeries
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return DataSeries(series / (np.max(series) - np.min(series)))

    @staticmethod
    def normalize_custom(series, par1, par2):
        """
        Normalizes the series considering two factors ((IBI-par1)/par2)
        @param par1: a scale for each sample
        @param par2: second parameter: average calm-state expected bpm
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return DataSeries((par1 - series) / par2)

    @staticmethod
    def filter_outliers(series, last=13, min_bpm=24, max_bpm=198, win_length=50):
        """
        Removes outliers from RR series.
        @param series: DataSeries
        @type series: DataSeries
        @param last: last percentage
        @type last: float
        @param min_bpm: minimum bpm to be considered valid
        @type min_bpm: float
        @param max_bpm: maximum bpm to be considered valid
        @type max_bpm: float
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
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
        """
        Example filter method, does nothing
        @param series: DataSeries
        @type series: DataSeries
        @return: Filtered DataSeries
        @rtype: DataSeries
        """
        assert isinstance(series, DataSeries)
        return series
