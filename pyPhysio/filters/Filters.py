# coding=utf-8
from __future__ import division
import numpy as _np
from scipy.signal import gaussian as _gaussian, filtfilt as _filtfilt, filter_design as _filter_design
from ..BaseFilter import Filter as _Filter
from ..Signal import EvenlySignal as _EvenlySignal, UnevenlySignal as _UnevenlySignal, Signal as _Signal
from ..Utility import PhUI as _PhUI
from ..Parameters import Parameter as _Par
from ..Utility import abstractmethod as _abstract

__author__ = 'AleB'

"""
Filters are processing steps that take as input a SIGNAL and gives as output another SIGNAL of the SAME NATURE.
"""


class Normalize(_Filter):
    """
    Normalized the input signal using the general formula: ( signal - BIAS ) / RANGE

    Parameters
    ----------
    norm_method : 
        Method for the normalization: 
        'mean' - remove the mean [ BIAS = mean(signal); RANGE = 1 ]
        'standard' - standardization [ BIAS = mean(signal); RANGE = std(signal) ]
        'min' - remove the minimum [ BIAS = min(signal); RANGE = 1 ]
        'maxmin' - maxmin normalization [ BIAS = min(signal); RANGE = ( max(signal) - min(signal ) ]
        'custom' - custom [ BIAS = bias, RANGE = range ]

    bias:
        Bias for custom normalization
    range:
        Range for custom normalization

    Returns
    -------
    Signal : the normalized signal. 

    Notes
    -----
        ...
    """

    class Types(object):
        Mean = 'mean'
        MeanSd = 'standard'
        Min = 'min'
        MaxMin = 'maxmin'
        Custom = 'custom'

    @classmethod
    def algorithm(cls, signal, params):
        from ..indicators.TimeDomain import Mean as _Mean, StDev as _StDev

        method = params['norm_method']
        if method == Normalize.Types.Mean:
            return signal - _Mean()(signal)
        elif method == Normalize.Types.MeanSd:
            return (signal - _Mean()(signal)) / _StDev()(signal)
        elif method == Normalize.Types.Min:
            return signal - _np.min(signal)
        elif method == Normalize.Types.MaxMin:
            return (signal - _np.min(signal)) / (_np.max(signal) - _np.min(signal))
        elif method == Normalize.Types.Custom:
            return (signal - params['norm_bias']) / params['norm_range']

    _params_descriptors = {
        'norm_method': _Par(2, str, 'Method for the normalization.', 'standard',
                            lambda x: x in ['mean', 'standard', 'min', 'maxmin', 'custom']),
        'norm_bias': _Par(2, float, 'Bias for custom normalization', 0, activation=lambda x, p: p['norm_method'] == 'custom'),
        'norm_range': _Par(2, float, 'Range for custom normalization', 0, activation=lambda x, p: p['norm_method'] == 'custom')
    }


class Diff(_Filter):
    """
    Computes the differences between adjacent samples.

    Parameters
    ----------
    degree : int > 0
        The degree of the differences
    
    Returns
    -------
    Signal : the differences signal. 

    Notes
    -----
    Note that the length of the returned signal is: len(input_signal) - degree
    """

    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the differences between consecutive values
        """
        if isinstance(signal, _Signal) and not isinstance(signal, _EvenlySignal):
            cls.log(
                "Computing %s on '%s' may not make sense." % (cls.__name__, signal.__class__.__name__))
        degree = params['degree']

        # TODO (Ale): Manage Time references
        sig_1 = signal[:-degree]
        sig_2 = signal[degree:]

        return sig_2 - sig_1

    _params_descriptors = {
        'degree': _Par(default=1, requirement_level=0, description='Degree of the differences', pytype=float)
    }


class IIRFilter(_Filter):
    """
    Filter the input signal using an Infinite Impulse Response filter.

    Parameters
    ----------
    fp : list
        The pass frequencies
    fs : list
        The stop frequencies
    loss : float (default=)
        Loss tolerance in the pass band
    att : float (default=)
        Minimum attenuation required in the stop band.
    ftype : str (default=)
        Type of filter.

    Returns
    -------
    Signal : the filtered signal

    Notes
    -----
    See : func:`scipy.signal.filter_design.iirdesign` for additional information
    """

    @classmethod
    def algorithm(cls, signal, params):
        fsamp = signal.get_sampling_freq()
        fp, fs, loss, att, ftype = params["fp"], params["fs"], params["loss"], params["att"], params["ftype"]

        # TODO (Ale): if A and B already exist and fsamp is not changed skip the following

        # ---------
        # TODO (new feature): check that fs and fp are meaningful
        # TODO (new feature): check if fs, fp, fsamp allow no solution for the filter
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        fs = _np.array(fs)
        # TODO: Try (nyq=0)?
        wp = fp / nyq
        ws = fs / nyq
        b, a = _filter_design.iirdesign(wp, ws, loss, att, ftype=ftype, output="ba")
                
        # TODO (new feature) Trovare metodo per capire se funziona o no
        # if _np.max(a)>BIG_NUMBER | _np.isnan(_np.sum(a)):
        #    _PhUI.w('Filter parameters allow no solution')
        #    return signal
        # ---------
        # FIXME: con _filtfilt signal perde la classe e rimane array
        # TODO (Andrea): va bene EvenlySignal?
        sig_filtered = _EvenlySignal(_filtfilt(b, a, signal), signal.get_sampling_freq(), signal.get_signal_nature(),
                                     signal.get_start_time(), signal.get_metadata())
        if _np.isnan(sig_filtered[0]):
            cls.warn(cls.__name__ + ': Filter parameters allow no solution. Returning original signal.')
            return signal
        else:
            return sig_filtered

    _params_descriptors = {
        'fp': _Par(2, list, 'The pass frequencies'),
        'fs': _Par(2, list, 'The stop frequencies'),
        'loss': _Par(1, float, 'Loss tolerance in the pass band', 0.1, lambda x: x > 0),
        'att': _Par(1, float, 'Minimum attenuation required in the stop band.', 40, lambda x: x > 0),
        'ftype': _Par(1, str, 'Type of filter', 'butter', lambda x: x in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'])
    }

    @_abstract
    def plot(self):
        # plot frequency response
        # TODO (new feature)
        # WARNING 'not implemented'
        pass

class DenoiseEDA(_Filter):
    """
    DenoiseEDA
    
    Removes noise due to sensor displacement from the EDA signal.
    
    Parameters
    ----------
    TH : float
        Threshold to detect the noise
   
    Returns
    -------
    signal_out : nparray
        De-noised signal
            
    Notes
    -----
    See paper
    """

    @classmethod
    def algorithm(cls, signal, params):
        
        threshold = params['threshold']
        win_len = params['win_len']
        #TODO: detect the long periods of drop
        noise = ConvolutionalFilter(irftype ='triang', win_len = win_len, normalize=True)(abs(_np.diff(signal)))
        
        idx_ok = _np.where(noise <= threshold)[0]
        
        if idx_ok[0] != 0:
            idx_ok = _np.r_[0, idx_ok].astype(int)
        
        if idx_ok[-1] != len(signal)-1:
            idx_ok = _np.r_[idx_ok, len(signal)-1].astype(int)
            
        denoised = _UnevenlySignal(signal[idx_ok], idx_ok, signal.get_sampling_freq(), len(signal))
        signal_out = denoised.to_evenly('linear')
        return(signal_out)

    _params_descriptors = {
        'threshold': _Par(2, float, 'Threshold to detect the noise'),
        'win_len': _Par(1, float, 'Length of the window', 2, lambda x: x > 0)
    }

    def plot(self):
        # TODO (new feature)
        # WARNING 'not implemented'
        pass


class MatchedFilter(_Filter):
    """
    Matched filter

    It generates a template using reference indexes and filters the signal.

    Parameters
    ----------
    template : array
        The template for matched filter (not reversed)

    Returns
    -------
    Signal : 
        The filtered signal

    Notes
    -----
        ...

    """

    @classmethod
    def algorithm(cls, signal, params):  # TODO (Andrea): check normalization TEST
        template = params["template"]
        filtered_signal = _np.convolve(signal, template)
        filtered_signal = filtered_signal[_np.argmax(template):]
        return filtered_signal  # TODO (Andrea): Deve tornare signal, fsamp? start_time? etc..

    _params_descriptors = {
        'template': _Par(2, list, 'The template for matched filter (not reversed)')
    }

    def plot(self):
        # TODO (new feature)
        # WARNING 'not implemented'
        pass


class ConvolutionalFilter(_Filter):
    """
    Convolution-based filter

    It filters a signal by convolution with a given impulse response function (IRF).

    Parameters
    ----------
    irftype : str
        Type of IRF to be generated. 'gauss', 'rect', 'triang', 'dgauss', 'custom'.
    normalize : boolean
        Whether to normalizes the IRF to have unitary area
    win_len : int
        Durarion of the generated IRF in seconds (if irftype is not 'custom')
    irf : array
        IRF to be used if irftype is 'custom'
    
    Returns
    -------
    Signal : 
        The filtered signal

    Notes
    -----
    ...
    """

    class Types(object):
        Same = 'none'
        Gauss = 'gauss'
        Rect = 'rect'
        Triang = 'triang'
        Dgauss = 'dgauss'
        Custom = 'custom'

    @classmethod
    def algorithm(cls, signal, params):
        irftype = params["irftype"]
        normalize = params["normalize"]

        fsamp = signal.get_sampling_freq()
        irf = None

        if irftype == 'custom':
            if 'irf' not in params:
                cls.error("'irf' parameter missing.")
                return signal
            else:
                irf = _np.array(params["irf"])
                n = len(irf)
        else:
            if 'win_len' not in params:
                cls.error("'win_len' parameter missing.")
                return signal
            else:
                n = int(params['win_len']) * fsamp

                if irftype == 'gauss':
                    std = _np.floor(n / 8)  # TODO (Andrea): error if win len < 8 samples
                    irf = _gaussian(n, std)
                elif irftype == 'rect':
                    irf = _np.ones(n)
                elif irftype == 'triang':
                    irf_1 = _np.arange(n // 2)
                    irf_2 = irf_1[-1] - _np.arange(n // 2)
                    if n % 2 == 0:
                        irf = _np.r_[irf_1, irf_2]
                    else:
                        irf = _np.r_[irf_1, irf_1[-1] + 1, irf_2]
                elif irftype == 'dgauss':
                    std = _np.round(n / 8)
                    g = _gaussian(n, std)
                    irf = _np.diff(g)
                
        # NORMALIZE
        if normalize:
            irf = irf / _np.sum(irf)  # TODO (Andrea): account fsamp? TEST

        # TODO: sicuri che dopo questa riga signal rimanga un array? No
        # TODO (Andrea): n non dovrebbe essere definita anche in caso di irftype == custom?
        signal_ = _np.r_[_np.ones(n) * signal[0], signal, _np.ones(n) * signal[-1]]  # TESTME

        signal_f = _np.convolve(signal_, irf, mode='same')
        # TODO (Andrea) va bene evenly signal?
        signal_out = _EvenlySignal(signal_f[n:-n], signal.get_sampling_freq(), signal.get_signal_nature(),
                                   signal.get_start_time(), signal.get_metadata())
        return signal_out

    _params_descriptors = {
        'irftype': _Par(1, str, 'Type of IRF to be generated.', 'gauss',
                        lambda x: x in ['gauss', 'rect', 'triang', 'dgauss', 'custom']),
        'normalize': _Par(1, bool, 'Whether to normalizes the IRF to have unitary area', True),
        # TODO (Andrea): win_len was int is it ok float?
        'win_len': _Par(2, float, "Durarion of the generated IRF in seconds (if irftype is not 'custom')", 1,
                        lambda x: x > 0, lambda x, p: p['irftype'] != 'custom'),
        'irf': _Par(2, list, "IRF to be used if irftype is 'custom'", activation=lambda x, p: p['irftype'] == 'custom')
    }

    @classmethod
    def plot(cls):
        # TODO (new feature)
        # WARNING 'not implemented'
        pass


class DeConvolutionalFilter(_Filter):
    """
    Convolution-based filter

    It filters a signal by deconvolution with a given impulse response function (IRF).

    Parameters
    ----------
    irf : array
        IRF used to deconvolve the signal
    normalize : boolean
        Whether to normalize the IRF to have unitary area

    Returns
    -------
    filtered_signal : array
        The filtered signal

    Notes
    -----
    """

    @classmethod
    def algorithm(cls, signal, params):
        irf = params["irf"]
        normalize = params["normalize"]

        # TODO (Andrea): normalize?
        if normalize:
            irf = irf / _np.sum(irf)
        l = len(signal)
        fft_signal = _np.fft.fft(signal, n=l)
        fft_irf = _np.fft.fft(irf, n=l)
        out = _np.fft.ifft(fft_signal / fft_irf)
        
        out_signal = _EvenlySignal(abs(out), signal.get_sampling_freq(), signal.get_signal_nature(), signal.get_start_time(), signal.get_metadata())

        return out_signal

    _params_descriptors = {
        'irf': _Par(2, list, 'IRF used to deconvolve the signal'),
        'normalize': _Par(1, bool, 'Whether to normalize the IRF to have unitary area', True)
    }

    @classmethod
    def plot(cls):
        # WARNING 'Not implemented'
        # TODO (new feature) plot irf
        pass


# class AdaptiveThresholding(_Filter): #TODO: possiamo nascondere per il momento
#
#     Adaptively (windowing) threshold the signal using C*std(signal) as thresholding value.
#     See Raju 2014.
#
#     Parameters
#     ----------
#     signal : array
#         The input signal
#     winlen : int
#         Size of the window
#     C : float
#         Coefficient for the threshold
#
#     Returns
#     -------
#     thresholded_signal : array
#         The thresholded signal
#
#
#     @classmethod
#     def algorithm(cls, signal, params):
#         winlen = params['win_len']
#         C = params['C']
#         winlen = int(_np.round(winlen))
#         signal = _np.array(signal)
#         signal_out = _np.zeros(len(signal))
#         for i in range(0, len(signal), winlen):
#             curr_block = signal[i:i + winlen]
#             eta = C * _np.std(curr_block)
#             curr_block[curr_block < eta] = 0
#             signal_out[i:i + winlen] = curr_block
#         return signal_out
#
#     @classmethod
#     def check_params(cls, params):
#         if 'win_len' not in params:
#             # default = 100 # GRAVE
#             pass
#         if 'C' not in params:
#             # default = 1 # OK
#             pass
#         return params
