# coding=utf-8
from __future__ import division

from ..BaseIndicator import Indicator as _Indicator
from ..tools.Tools import PSD as PSD
import numpy as _np
from ..Parameters import Parameter as _Par

__author__ = 'AleB'


class InBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        freq, spec = PSD(**params)(data)
        
        # freq is sorted so
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])

        return freq[i_min:i_max], spec[i_min:i_max]

    _params_descriptors = {
        'freq_min': _Par(2, float, 'Lower frequency of the band', 0, lambda x: x > 0),
        'freq_max': _Par(2, float, 'Higher frequency of the band', 0, lambda x: x > 0)
    }


class PowerInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        freq, powers = InBand(**params)(data)
        return _np.sum(powers)

    _params_descriptors = InBand.get_params_descriptors()
    # TODO (feature): add normalize option (total, length)
    

class PeakInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band = InBand(**params)(data)
        return _freq_band[_np.argmax(_pow_band)]

    _params_descriptors = InBand.get_params_descriptors()
