# coding=utf-8
from __future__ import division
import numpy as _np
from scipy.signal import gaussian as _gaussian, filtfilt as _filtfilt, filter_design as _filter_design
from ..BaseFilter import Filter as _Filter
from ..PhUI import PhUI as _PhUI
from ..indicators.Indicators import Mean as _Mean, SD as _SD

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
		method = params['norm_method']
        if method == Normalize.Types.Mean:
			return Normalize._mean(signal)
		elif method == Normalize.Types.MeanSd:
			return Normalize._mean_sd(signal)
		elif method == Normalize.Types.Min:
			return Normalize._min(signal)
		elif method == Normalize.Types.MaxMin:
			return Normalize._max_min(signal)
		elif method == Normalize.Types.Custom:
			return Normalize._custom(signal, params['norm_bias'], params['norm_range'])

    @classmethod
    def _check_params(cls, params):
        params = {
			'norm_method' : ListPar('standard', 2, 'Method for the normalization.', ['mean', 'standard', 'min', 'maxmin', 'custom']),
			'norm_bias' : FloatPar(0, 2, 'Bias for custom normalization', '', 'norm_method'=='custom'),
			'norm_range' : FloatPar(0, 2, 'Range for custom normalization', '', 'norm_method'=='custom')
			}
        return params

	# TODO (Ale): Non possiamo fare a meno di definire tutte le funzioni sotto?
    @staticmethod
    def _mean(signal):
        """
        Normalizes the signal removing the mean (val-mean)
        """
        return signal - _Mean.get(signal)

    @staticmethod
    def _mean_sd(signal):
        """
        Normalizes the signal removing the mean and dividing by the standard deviation (val-mean)/sd
        """
        return signal - _Mean.get(signal) / _SD.get(signal)

    @staticmethod
    def _min(signal):
        """
        Normalizes the signal removing the minimum value (val-min)
        """
        return signal - _np.min(signal)

    @staticmethod
    def _max_min(signal):
        """
        Normalizes the signal removing the min value and dividing by the range width (val-min)/(max-min)
        """
        return (signal - _np.min(signal)) / (_np.max(signal) - _np.min(signal))

    @staticmethod
    def _custom(signal, bias, range): #TODO (Ale) Occhio che stai usando la keyword 'range'
        """
        Normalizes the signal considering two factors ((val-bias)/range)
        """
        return (signal - bias) / range


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
        assert 'degree' in params, "Need parameter 'degree'."
        
        degree = params['degree']
        
        # TODO: Manage Time references, in particular if Unevenly (to be discussed...) .. who cares?
        sig_1 = signal[:-degree]
        sig_2 = signal[degree:]

        return sig_2 - sig_1

    @classmethod
    def _check_params(cls, params):
        params = {
			'degree' : FloatPar(1, 0, 'Degree of the differences', '')
			}
        return params


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
        fsamp = signal.fsamp
        fp, fs, loss, att, ftype = params["fp"], params["fs"], params["loss"], params["att"], params["ftype"]

        # TODO (Ale): if A and B already exist and fsamp is not changed skip the following

        # ---------
        # TODO (new feature): check that fs and fp are meaningful
        # TODO (new feature): check if fs, fp, fsamp allow no solution for the filter
        nyq = 0.5 * fsamp
        fp = _np.array(fp)
        fs = _np.array(fs)
        wp = fp / nyq
        ws = fs / nyq
        b, a = _filter_design.iirdesign(wp, ws, loss, att, ftype=ftype)
        if _np.isnan(b[0]) | _np.isnan(a[0]):
            #WARNING 'Filter parameters allow no solution'
            return signal
        # ---------
        
        return _filtfilt(b, a, signal)

    @classmethod
    def check_params(cls, params):
        params = {
			'fp': VectorPar(2, 'The pass frequencies'),
			'fs': VectorPar(2, 'The stop frequencies'),
			'loss': FloatPar(0.1, 1, 'Loss tolerance in the pass band', '>0'),
			'att': FloatPar(40, 1, 'Minimum attenuation required in the stop band.', '>0'),
			'ftype': ListPar('butter', 1, 'Type of filter', ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'])
			}
        return params

    def plot(self):
        # plot frequency response
        #TODO (new feature)
        #WARNING 'not implemented'
        pass


class MatchedFilter(_Filter):
    """
    Matched filter

    It generates a template using reference indexes and filters the signal.

    Parameters
    ----------
    template : nparray
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
    def algorithm(cls, signal, template): #TODO (Andrea): check normalization TEST
		filtered_signal = _np.convolve(signal, template)
        filtered_signal = filtered_signal[_np.argmax(template):]
        return filtered_signal

    @classmethod
    def check_params(cls, params):
        params = {
			'template': VectorPar(2, 'The template for matched filter (not reversed)')
            }
        return params
	
	def plot(self):
        #TODO (new feature)
        #WARNING 'not implemented'
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
    irf : nparray
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
        normalize =  params["normalize"]
                
        fsamp = signal.sampling_freq
        
        if irftype == 'custom':
			if not 'irf' in params:
				#ERROR 'irf parameter needed'
				return(signal)
			irf =  _np.array(params["irf"])
		else:
			if not 'win_len' in params:
				#ERROR 'win_len parameter needed'
				return signal
			
			N = params['win_len'] * fsamp
			
			if irftype == 'gauss':
				std = _np.floor(N / 8)
				irf = _gaussian(N, std)
			elif irftype == 'rect':
				irf = _np.ones(N)
			elif irftype == 'triang':
				irf_1 = _np.arange(N // 2)
				irf_2 = irf_1[-1] - _np.arange(N // 2)
				if N % 2 == 0:
					irf = _np.r_[irf_1, irf_2]
				else:
					irf = _np.r_[irf_1, irf_1[-1] + 1, irf_2]
			elif irftype == 'dgauss':
				std = N // 8
				g = _gaussian(N, std)
				irf = _np.diff(g)

        # NORMALIZE
        if normalize:
            irf = irf / _np.sum(irf)  # TODO (Andrea): account fsamp? TEST
        
        #TODO (Ale): sicuri che dopo questa riga signal rimanga un in nparray?
        signal_ = _np.r_[_np.ones(N) * signal[0], signal, _np.ones(N) * signal[-1]]

        signal_f = _np.convolve(signal_, irf, mode='same')
        signal_out = signal_f[N:-N]
        return signal_out


    @classmethod
    def _check_params(cls, params):
        params = {
			'irftype': ListPar('none', 2, 'Type of IRF to be generated.', ['gauss', 'rect', 'triang', 'dgauss', 'custom']),
			'normalize': BoolPar(True, 1, 'Whether to normalizes the IRF to have unitary area'),
            'win_len': IntPar(1, 2, "Durarion of the generated IRF in seconds (if irftype is not 'custom')", '>0', 'irftype' != 'custom'),
            'irf': VectorPar(2, "IRF to be used if irftype is 'custom'")
            }
        return params

    @classmethod
    def plot(cls):
        #TODO (new feature)
        #WARNING 'not implemented'
        pass


class DeConvolutionalFilter(_Filter):
    """
    Convolution-based filter

    It filters a signal by deconvolution with a given impulse response function (IRF).

    Parameters
    ----------
    irf : nparray
        IRF used to deconvolve the signal
    normalize : boolean
        Whether to normalize the IRF to have unitary area

    Returns
    -------
    filtered_signal : nparray
        The filtered signal

    Notes
    -----
    """

    @classmethod
    def algorithm(cls, signal, params):
        irf = params["irf"]
        normalize = params["normalize"]
        
        #TODO (Andrea): normalize?
        if normalize:
            irf = irf / _np.sum(irf)
        l = len(signal)
        fft_signal = _np.fft.fft(signal, n=l)  # TODO: UNUSED var?
        fft_irf = _np.fft.fft(irf, n=l)
        out = _np.fft.ifft(fft_signal / fft_irf)

        return abs(out)

    @classmethod
    def check_params(cls, params):
        params = {
			'irf': VectorPar(2, 'IRF used to deconvolve the signal'),
			'normalize': BoolPar(True, 1, 'Whether to normalize the IRF to have unitary area')
			}
        return params

    @classmethod
    def plot(cls):
        #WARNING 'Not implemented'
        #TODO (new feature) plot irf
        pass

"""
class AdaptiveThresholding(_Filter): #TODO: possiamo nascondere per il momento
    
    Adaptively (windowing) threshold the signal using C*std(signal) as thresholding value.
    See Raju 2014.

    Parameters
    ----------
    signal : nparray
        The input signal
    winlen : int
        Size of the window
    C : float
        Coefficient for the threshold

    Returns
    -------
    thresholded_signal : nparray
        The thresholded signal
    

    @classmethod
    def algorithm(cls, signal, params):
        winlen = params['win_len']
        C = params['C']
        winlen = int(_np.round(winlen))
        signal = _np.array(signal)
        signal_out = _np.zeros(len(signal))
        for i in range(0, len(signal), winlen):
            curr_block = signal[i:i + winlen]
            eta = C * _np.std(curr_block)
            curr_block[curr_block < eta] = 0
            signal_out[i:i + winlen] = curr_block
        return signal_out

    @classmethod
    def check_params(cls, params):
        if 'win_len' not in params:
            # default = 100 # GRAVE
            pass
        if 'C' not in params:
            # default = 1 # OK
            pass
        return params
"""