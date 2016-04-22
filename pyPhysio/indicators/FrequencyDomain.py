# coding=utf-8
from __future__ import division

import spectrum
from scipy import signal
from ..BaseIndicator import Indicator as _Indicator
from ..indicators.SupportValues import SumSV as _SumSV, LengthSV as _LengthSV, DiffsSV as _DiffsSV, MedianSV as _MedianSV
from ..filters.Filters import Diff as _Diff
from ..tools.Tools import PSD as PSD
import numpy as _np

__author__ = 'AleB'


class InBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        assert 'freq_min' in params, "Need the parameter 'freq_min' as the lower bound of the band."
        assert 'freq_max' in params, "Need the parameter 'freq_max' as the higher bound of the band."
        
        # TODO (Andrea): ?pass the PSD estimator instance as parameter? Yes, or only its name, discuss?
        # assert 'psd_method' in params, "Need the parameter 'psd_method' as the higher bound of the band."

        freq, spec = PSD.get(data, params)

        return ([freq[i] for i in xrange(len(freq)) if params['freq_min'] <= freq[i] < params['freq_max']],
                [spec[i] for i in xrange(len(spec)) if params['freq_min'] <= freq[i] < params['freq_max']])

    @classmethod
    def get_used_params(cls):
        return ['freq_max', 'freq_min']

    @classmethod
    def check_params(cls, params):
        params = {
            'freq_min': FloatPar(0, 2, 'Lower frequency of the band', '>0'),
            'freq_max': FloatPar(0, 2, 'Higher frequency of the band', '>0')
            }
        return params


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

    @classmethod
    def check_params(cls, params):
        params = {
            'freq_min': FloatPar(0, 2, 'Lower frequency of the band', '>0'),
            'freq_max': FloatPar(0, 2, 'Higher frequency of the band', '>0')
            }
        return params


class PeakInBand(_Indicator):
    @classmethod
    def algorithm(cls, data, params):
        _freq_band, _pow_band = InBand.get(data, params)
        return _freq_band[_np.argmax(_pow_band)]

    @classmethod
    def get_used_params(cls):
        return InBand.get_used_params()

    @classmethod
    def check_params(cls, params):
        params = {
            'freq_min': FloatPar(0, 2, 'Lower frequency of the band', '>0'),
            'freq_max': FloatPar(0, 2, 'Higher frequency of the band', '>0')
            }
        return params
