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
        assert 'freq_min' in params, "Need the parameter 'freq_min' as the lower bound of the band."
        assert 'freq_max' in params, "Need the parameter 'freq_max' as the higher bound of the band."

        # TODO (Andrea): ?pass the PSD estimator instance as parameter? Yes, or only its name, discuss?

        freq, spec = PSD.get(data, params)

        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']])

    @classmethod
    def get_used_params(cls):
        return ['freq_max', 'freq_min']

    _params_descriptors = {
        'freq_min': _Par(2, (float, int), 'Lower frequency of the band', 0, lambda x: x > 0),
        'freq_max': _Par(2, (float, int), 'Higher frequency of the band', 0, lambda x: x > 0)
    }


class PowerInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        frequencies, _pow_band = InBand.get(data, params)
        df = frequencies[1] - frequencies[0]
        # TODO (Andrea) Decidere se e come normalizzare
        return df * _np.sum(_pow_band)

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()

    _params_descriptors = {
        'freq_min': _Par(2, (float, int), 'Lower frequency of the band', 0, lambda x: x > 0),
        'freq_max': _Par(2, (float, int), 'Higher frequency of the band', 0, lambda x: x > 0)
    }


class PeakInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band = InBand.get(data, params)
        return _freq_band[_np.argmax(_pow_band)]

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()

    _params_descriptors = {
        'freq_min': _Par(2, (float, int), 'Lower frequency of the band', 0, lambda x: x > 0),
        'freq_max': _Par(2, (float, int), 'Higher frequency of the band', 0, lambda x: x > 0)
    }
