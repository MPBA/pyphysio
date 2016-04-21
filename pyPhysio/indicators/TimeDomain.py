# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..indicators.SupportValues import SumSV as _SumSV, LengthSV as _LengthSV, DiffsSV as _DiffsSV, MedianSV as _MedianSV
from ..filters.Filters import Diff as _Diff
import numpy as _np

__author__ = 'AleB'


class Histogram(_Indicator):
    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the Histogram data to cache
        @return: (values, bins)
        @rtype: (array, array)
        """
        
        return _np.histogram(signal, params['histogram_bins'])

    @classmethod
    def get_used_params(cls):
        return ['histogram_bins']
	
	@classmethod
    def check_params(cls, params):
        params = {
			'histogram_bins': MultiTypePar(1, [IntPar(100, 1, 'Number of bins', '>0'), VectorPar(1, 'Bin edges, including the rightmost edge')])
			}
        return params


class HistogramMax(_Indicator):
    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the Histogram's max value
        @return: (values, bins)
        @rtype: (array, array)
        """
        h, b = Histogram.get(signal, params)
        return _np.max(h)  # TODO 2 Andrea: max h or b(max h)??

    @classmethod
    def get_used_params(cls):
        return Histogram.get_used_params()


class Mean(_Indicator):
    """
    Calculates the average value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Mean, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmean(data)


class Min(_Indicator):
    """
    Calculates the minimum value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Min, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmin(data)


class Max(_Indicator):
    """
    Calculates the maximum value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Max, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmax(data)


class Range(_Indicator):
    """
    Calculates the range value of the data.
    """

    def __init__(self, params=None, **kwargs):
        super(Range, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return Max(data) - Min(data)


class Median(_Indicator):
    """
    Calculates the median of the data series.
    """
    
    def __init__(self, params=None, **kwargs):
        super(Median, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanmedian(data)


class StDev(_Indicator):
    """
    Calculates the standard deviation of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(SD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        return _np.nanstd(data)


class AUC(_Indicator):
	"""
    Calculates the Area Under the Curve of the data series.
    """

    def __init__(self, params=None, **kwargs):
        super(AUC, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
		fsamp = data.sampling_freq
        return (1/fsamp) * _np.nansum(data)


## HRV
class RMSSD(_Indicator):
    """
    Calculates the square root of the mean of the squared differences.
    """

    def __init__(self, params=None, **kwargs):
        super(RMSSD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff.get(data)
        return _np.sqrt(sum(diff ** 2) / len(diff))


class SDSD(_Indicator):
    """Calculates the standard deviation of the differences between each value and its next."""

    def __init__(self, params=None, **kwargs):
        super(SDSD, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
        diff = _Diff.get(data)
        return _np.std(diff)


class Triang(_Indicator):
    """Calculates the Triangular index."""

    def __init__(self, params=None, **kwargs):
        super(Triang, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
		min_ibi = _np.min(data)
		max_ibi = _np.max(data)
		bins = _np.arange(min_ibi, max_ibi, 1000./128)
		if len(bins)>=10:
			h, b = _Histogram.get(data, histogram_bins=bins)
			return len(data) / _np.max(h)
		else:
			# warning
			return _np.nan


class TINN(_Indicator):
    """Calculates the difference between two histogram-related indicators."""

    def __init__(self, params=None, **kwargs):
        super(TINN, self).__init__(params, kwargs)

    @classmethod
    def algorithm(cls, data, params):
		min_ibi = _np.min(data)
		max_ibi = _np.max(data)
		bins = _np.arange(min_ibi, max_ibi, 1000./128)
		if len(bins)>=10:
			h, b = _Histogram.get(data, histogram_bins=bins)
			max_h = _np.max(h)
			hist_left = _np.array(h[0:_np.argmax(h)])
			ll = len(hist_left)
			hist_right = _np.array(h[_np.argmax(h):])
			rl = len(hist_right)
			y_left = _np.array(_np.linspace(0, max_h, ll))
        
			minx = _np.Inf
			pos = 0
			for i in range(1, len(hist_left) - 1):
				curr_min = _np.sum((hist_left - y_left) ** 2)
				if curr_min < minx:
					minx = curr_min
					pos = i
				y_left[i] = 0
				y_left[i + 1:] = _np.linspace(0, max_h, ll - i - 1)
		
			n = b[pos - 1]
		
			y_right = _np.array(_np.linspace(max_h, 0, rl))
			minx = _np.Inf
			pos = 0
			for i in range(rl-1, 1, -1):
				curr_min = _np.sum((hist_right - y_right) ** 2)
				if curr_min < minx:
					minx = curr_min
					pos = i
				y_right[i - 1] = 0
				y_right[0:i - 2] = _np.linspace(max_h, 0, i - 2)
		
			m = b[_np.argmax(h) + pos + 1]
			return m-n
		else:
			#WARNING
			return _np.nan

