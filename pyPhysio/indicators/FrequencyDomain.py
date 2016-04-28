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
        # TODO (Andrea): ?pass the PSD estimator instance as parameter? Yes, or only its name, discuss?
        freq, spec = PSD(params)(data)
        # freq is sorted so
        i_min = _np.searchsorted(freq, params["freq_min"])
        i_max = _np.searchsorted(freq, params["freq_max"])
        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']])

    _params_descriptors = {
        'freq_min': _Par(2, float, 'Lower frequency of the band', 0, lambda x: x > 0),
        'freq_max': _Par(2, float, 'Higher frequency of the band', 0, lambda x: x > 0)
    }


class PowerInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        frequencies, _pow_band = InBand(params)(data)
        df = frequencies[1] - frequencies[0]
        # TODO (Andrea) Decidere se e come normalizzare
        return df * _np.sum(_pow_band)

    _params_descriptors = InBand.get_params_descriptors()


class PeakInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band = InBand(params)(data)
        return _freq_band[_np.argmax(_pow_band)]

    _params_descriptors = InBand.get_params_descriptors()
