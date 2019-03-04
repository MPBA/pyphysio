# coding=utf-8
from __future__ import division
import numpy as _np
from scipy.signal import welch as _welch, periodogram as _periodogram, freqz as _freqz
import scipy.optimize as _opt
from scipy import linalg as _linalg

import itertools as _itertools
from ..BaseTool import Tool as _Tool
from ..Signal import UnevenlySignal as _UnevenlySignal, EvenlySignal as _EvenlySignal


class Diff(_Tool):
    """
    Computes the differences between adjacent samples.

    Optional parameters
    -------------------
    degree : int, >0, default = 1
        Sample interval to compute the differences
    
    Returns
    -------
    signal : 
        Differences signal. 

    """

    def __init__(self, degree=1):
        assert degree > 0, "The degree value should be positive"
        _Tool.__init__(self, degree=degree)

    @classmethod
    def algorithm(cls, signal, params):
        """
        Calculates the differences between consecutive values
        """
        degree = params['degree']

        sig_1 = signal[:-degree]
        sig_2 = signal[degree:]

        out = _EvenlySignal(values=sig_2 - sig_1,
                            sampling_freq=signal.get_sampling_freq(),
                            signal_type=signal.get_signal_type(),
                            start_time=signal.get_start_time() + degree / signal.get_sampling_freq())

        return out
class PeakDetection(_Tool):
    """
    Estimate the maxima and the minima in the signal (in particular for periodic signals).

    Parameters
    ----------
    delta : float or list
        Threshold for the detection of the peaks. If it is a list it must have the same length of the signal.
        
    Optional parameters
    -------------------
    refractory : float, >=0, default = 0
        Seconds to skip after a detected paek to look for new peaks.
    start_max : boolean, default = True
        Whether to start looking for a maximum or (False) for a minimum.

    Returns
    -------
    maxp : numpy.array
        Array containing indexes of the maxima
    minp : numpy.array
        Array containing indexes of the minima
    maxv : numpy.array
        Array containing values of the maxima
    minv : numpy.array
        Array containing values of the minima
    """

    def __init__(self, delta, refractory=0, start_max=True):
        delta = _np.array(delta)
        assert delta.ndim <= 1, "Delta value should be 1 or 0-dimensional"
        assert delta.all() > 0, "Delta value/s should be positive"
        assert refractory >= 0, "Refractory value should be non negative"
        _Tool.__init__(self, delta=delta, refractory=refractory, start_max=start_max)

    @classmethod
    def algorithm(cls, signal, params):
        refractory = params['refractory']
        if refractory == 0:  # if 0 then do not skip samples
            refractory = 1
        else:  # else transform the refractory from seconds to samples
            refractory = refractory * signal.get_sampling_freq()
        look_for_max = params['start_max']
        delta = params['delta']

        minp = []
        maxp = []

        minv = []
        maxv = []

        scalar = delta.ndim == 0
        if scalar:
            d = delta

        if len(signal) < 1:
            cls.warn("Empty signal (len < 1), returning empty.")
        elif not scalar and len(delta) != len(signal):
            cls.error("delta vector's length differs from signal's one, returning empty.")
        else:
            mn_pos_candidate = mx_pos_candidate = 0
            mn_candidate = mx_candidate = signal[0]

            i_activation_min = 0
            i_activation_max = 0

            for i in range(1, len(signal)):
                sample = signal[i]
                if not scalar:
                    d = delta[i]

                if sample > mx_candidate:
                    mx_candidate = sample
                    mx_pos_candidate = i
                if sample < mn_candidate:
                    mn_candidate = sample
                    mn_pos_candidate = i

                if look_for_max:
                    if i >= i_activation_max and sample < mx_candidate - d:  # new max
                        maxp.append(mx_pos_candidate)
                        maxv.append(mx_candidate)
                        i_activation_max = i + refractory

                        mn_candidate = sample
                        mn_pos_candidate = i

                        look_for_max = False
                else:
                    if i >= i_activation_min and sample > mn_candidate + d:  # new min
                        minp.append(mn_pos_candidate)
                        minv.append(mn_candidate)
                        i_activation_min = i + refractory

                        mx_candidate = sample
                        mx_pos_candidate = i

                        look_for_max = True

        return _np.array(maxp), _np.array(minp), _np.array(maxv), _np.array(minv)


class PeakSelection(_Tool):
    """
    Identify the start and the end indexes of each peak in the signal, using derivatives.

    Parameters
    ----------
    indices : array, >=0
        Array containing indexes (first column) and values (second column) of the maxima
    win_pre : float, >0
        Duration (in seconds) of interval before the peak that is considered to find the start of the peak
    win_post : float, >0
        Duration (in seconds) of interval after the peak that is considered to find the end of the peak
    
    Returns
    -------
    starts : array
        Array containing start indexes
    ends : array
        Array containing end indexes
    """

    def __init__(self, indices, win_pre, win_post):
        indices = _np.array(indices)
        assert indices.ndim < 2, "Parameter indices has to be 1 or 0-dimensional"
        assert indices.all() >= 0, "Parameter indices contains negative values"
        assert win_pre > 0, "Window pre peak value should be positive"
        assert win_post > 0, "Window post peak value should be positive"
        _Tool.__init__(self, indices=indices, win_pre=win_pre, win_post=win_post)

    @classmethod
    def algorithm(cls, signal, params):
        i_peaks = params['indices']
        i_pre_max = int(params['win_pre'] * signal.get_sampling_freq())
        i_post_max = int(params['win_post'] * signal.get_sampling_freq())

        ZERO = 0.01

        i_start = _np.empty(len(i_peaks), int)
        i_stop = _np.empty(len(i_peaks), int)

        signal_dt = Diff()(signal)
        for i in range(len(i_peaks)):
            i_pk = int(i_peaks[i])

            if i_pk < i_pre_max:
                i_st = 0
                i_sp = i_pk + i_post_max
            elif i_pk >= len(signal_dt) - i_post_max:
                i_st = i_pk - i_pre_max
                i_sp = len(signal_dt) - 1
            else:
                i_st = i_pk - i_pre_max
                i_sp = i_pk + i_post_max

            # find START
            signal_dt_pre = signal_dt[i_st:i_pk]
            i_pre = len(signal_dt_pre) - 1

            # OR below is to allow small fluctuations (?)

            while i_pre > 0 and (signal_dt_pre[i_pre] > 0 or abs(signal_dt_pre[i_pre]) <= ZERO):
                i_pre -= 1

            i_start[i] = i_st + i_pre + 1

            # find STOP
            signal_dt_post = signal_dt[i_pk: i_sp]
            i_post = 1

            # OR below is to allow small fluctuations (?)
            while i_post < len(signal_dt_post) - 1 and (
                            signal_dt_post[i_post] < 0 or abs(signal_dt_post[i_post]) <= ZERO):
                i_post += 1

            i_stop[i] = i_pk + i_post

        return i_start, i_stop


class SignalRange(_Tool):
    """
    Estimate the local range of the signal by sliding windowing

    Parameters
    ----------
    win_len : float, >0
        Length of the window  in seconds
    win_step : float, >0
        Shift to start the next window in seconds

    Optional parameters
    -------------------    
    smooth : boolean, default=True
        Whether to convolve the result with a gaussian window

    Returns
    -------
    deltas : numpy.array
        Local range of the signal
    """

    def __init__(self, win_len, win_step, smooth=True):
        assert win_len > 0, "Window length should be positive"
        assert win_step > 0, "Window step should be positive"
        _Tool.__init__(self, win_len=win_len, win_step=win_step, smooth=smooth)

    @classmethod
    def algorithm(cls, signal, params):
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        fsamp = signal.get_sampling_freq()
        idx_len = int(win_len * fsamp)
        idx_step = int(win_step * fsamp)

        if len(signal) < idx_len:
            cls.warn("Input signal is shorter than the window length.")
            return _np.max(signal) - _np.min(signal)
        else:
            windows = _np.arange(0, len(signal) - idx_len + 1, idx_step)
            deltas = _np.zeros(len(signal))

            curr_delta = 0
            for start in windows:
                portion_curr = signal[start: start + idx_len]
                curr_delta = _np.max(portion_curr) - _np.min(portion_curr)
                deltas[start:start + idx_len] = curr_delta

            deltas[windows[-1] + idx_len:] = curr_delta

            if smooth:
                win_len = int(win_len*2*fsamp)
                deltas = _np.convolve(deltas, _np.ones(win_len)/win_len, mode='same')

            return deltas


class PSD(_Tool):
    """
    Estimate the power spectral density (PSD) of the signal.

    Parameters
    ----------
    method : str
        Method to estimate the PSD. Available methods: 'welch', 'fft', 'ar'
        
    Optional parameters
    -------------------
    
    nfft : int, >0, default=2048
        Number of samples of the PSD
    window : str, default = 'hamming'
        Type of window
    min_order : int, >0, default=18
        Minimum order of the model to be tested for psd_method='ar'
    max_order : int, >0, default=25
        Maximum order of the model to be tested for psd_method='ar'
    normalize : boolean, default = True
        Whether to normalize the PSD
    remove_mean : boolean, default = True
        Whether to remove the mean from the signal before estimating the PSD
    
    Returns
    -------
    freq : numpy.array
        Frequencies
    psd : numpy.array
        Power Spectrum Density
    """

    def __init__(self, method, nfft=2048, window='hamming', min_order=10, max_order=30, normalize=False,
                 remove_mean=True, **kwargs):
        
        _method_list = ['welch', 'fft', 'ar']
        _window_list = ['hamming', 'blackman', 'hanning', 'bartlett', 'none']

        assert method in _method_list, "Parameter method should be in " + _method_list.__repr__()
        assert nfft > 0, "nfft value should be positive"
        assert window in _window_list, "Parameter window type should be in " + _window_list.__repr__()
        if method == "ar":
            assert min_order > 0, "Minimum order for the AR method should be positive"
            assert max_order > 0, "Maximum order for the AR method should be positive"
        
        _Tool.__init__(self, method=method, nfft=nfft, window=window, min_order=min_order,
                       max_order=max_order, normalize=normalize, remove_mean=remove_mean, **kwargs)

    # TODO (Feature - Issue #15): consider point below:
    # A density spectrum considers the amplitudes per unit frequency.
    # Density spectra are used to compare spectra with different frequency resolution as the
    # magnitudes are not influenced by the resolution because it is per Hertz. The amplitude
    # spectra on the other hand depend on the chosen frequency resolution.

    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        nfft = params['nfft'] if "nfft" in params else None
        window = params['window']
        normalize = params['normalize']
        remove_mean = params['remove_mean']

        assert isinstance(signal, _EvenlySignal), "The PSD can be computed on EvenlySignals only. Consider interpolating the signal: signal.resample(fsamp)"

        fsamp = signal.get_sampling_freq()

        if remove_mean:
            signal = signal - _np.mean(signal)

        if method == 'fft':
            freqs, psd = _periodogram(signal, fs=fsamp, window = window, nfft=nfft, return_onesided=True)

        elif method == 'welch':
            freqs, psd = _welch(signal, fsamp, window=window, return_onesided=True, nfft=nfft)

        elif method == 'ar':
            cls.warn("Using AR method: results might not be comparable with other methods")
            #methods derived from: https://github.com/mpastell/pyageng
            def autocorr(x, lag=30):
                c = _np.correlate(x, x, 'full')
                mid = len(c)//2
                acov = c[mid:mid+lag]
                acor = acov/acov[0]
                return(acor)
                
            def aryw(x, order=30):
                x = x - _np.mean(x)
                ac = autocorr(x, order+1)
                R = _linalg.toeplitz(ac[:order])
                r = ac[1:order+1]
                params = _np.linalg.inv(R).dot(r)
                return(params)
                
            def AIC_yule(signal, order):
                #this is from library spectrum: https://github.com/cokelaer/spectrum
                N = len(signal)
                assert N>=order, "The number of samples in the signal should be >= to the model order"
                
                C = _np.correlate(signal, signal, mode='full')/N
                r = C[N-1:]
                
                T0  = r[0]
                T = r[1:]
                
                A = _np.zeros(order, dtype=float)
                P = T0
                
                for k in range(0, order):
                    save = T[k]
                    if k == 0:
                        temp = -save / P
                    else:
                        for j in range(0, k):
                            save = save + A[j] * T[k-j-1]
                        temp = -save / P
                    
                    P = P * (1. - temp**2.)
                    A[k] = temp
                
                    khalf = (k+1)//2
                    for j in range(0, khalf):
                        kj = k-j-1
                        save = A[j]
                        A[j] = save + temp * A[kj]
                        if j != kj:
                            A[kj] += temp*save
                
                res = N * _np.log(P) + 2*(order + 1)
                return(res)
            
            min_order = params['min_order']
            max_order = params['max_order']

            if len(signal) <= max_order:
                cls.warn("Input signal too short: try another 'method', a lower 'max_order', or a longer signal")
                return [], []

            orders = _np.arange(min_order, max_order + 1)
            aics = [AIC_yule(signal, x) for x in orders]
            best_order = orders[_np.argmin(aics)]

            params = aryw(signal, best_order)
            a = _np.concatenate([_np.ones(1), -params])
            w, P = _freqz(1, a, whole = False, worN = nfft)
            
            psd = 2*_np.abs(P)/fsamp
            
        else:
            cls.warn('Method not understood, using welch.')
            bands_w, psd = _welch(signal, fsamp, nfft=nfft, scaling = 'spectrum')

        freqs = _np.linspace(start=0, stop=fsamp / 2, num=len(psd))

        # NORMALIZE
        if normalize:
            psd /= _np.sum(psd)
        return freqs, psd


class Maxima(_Tool):
    """
    Find all local maxima in the signal

    Parameters
    ----------
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')
    method : str
        Method to detect the maxima. Available methods: 'complete' or 'windowing'. 'complete' finds all the local
         maxima, 'windowing' uses a runnning window to find the global maxima in each window.
    
    Optional parameters
    -------------------
    refractory : float, >0, default=0
        Seconds to skip after a detected maximum to look for new maxima, when method = 'complete'. 

    Returns
    -------
    idx_maxs : array
        Array containing indexes of the maxima
    val_maxs : array
        Array containing values of the maxima
    """

    def __init__(self, method='complete', refractory=0, win_len=None, win_step=None):
        assert method in ['complete', 'windowing'], "Method not valid"
        assert refractory >= 0, "Refractory time value should be positive (or 0 to deactivate)"
        
        if method == 'windowing':
            assert win_len > 0, "Window length should be positive"
            assert win_step > 0, "Window step should be positive"
            _Tool.__init__(self, method=method, refractory=refractory, win_len=win_len, win_step=win_step)
        elif method == 'complete':
            _Tool.__init__(self, method=method, refractory=refractory)
        
        

    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        if method == 'complete':
            refractory = params['refractory']
            if refractory == 0:
                refractory = 1
            else:
                refractory = refractory * signal.get_sampling_freq()
            idx_maxs = []
            prev = signal[0]
            k = 1
            while k < len(signal) - 1 - refractory:
                curr = signal[k]
                nxt = signal[k + 1]
                if (curr >= prev) and (curr >= nxt):
                    idx_maxs.append(k)
                    prev = signal[k + 1 + refractory]
                    k = k + 2 + refractory
                else:  # continue
                    prev = signal[k]
                    k += 1
            idx_maxs = _np.array(idx_maxs).astype(int)
            maxs = signal[idx_maxs]
            return idx_maxs, maxs
        elif method == 'windowing':
            fsamp = signal.get_sampling_freq()
            winlen = int(params['win_len'] * fsamp)
            winstep = int(params['win_step'] * fsamp)

            # TODO (Andrea): check that winlen > 2
            # TODO (Andrea): check that winstep >= 1

            idx_maxs = [_np.nan]
            maxs = [_np.nan]

            if winlen < len(signal):
                idx_start = _np.arange(0, len(signal) - winlen + 1, winstep)
            else:
                idx_start = [0]

            for idx_st in idx_start:
                idx_sp = idx_st + winlen
                if idx_sp > len(signal):
                    idx_sp = len(signal)
                curr_win = signal[idx_st: idx_sp]
                curr_idx_max = _np.argmax(curr_win) + idx_st
                curr_max = _np.max(curr_win)

                # peak not already detected & peak not at the beginnig/end of the window:
                if curr_idx_max != idx_maxs[-1] and curr_idx_max != idx_st and curr_idx_max != idx_sp - 1:
                    idx_maxs.append(curr_idx_max)
                    maxs.append(curr_max)
            idx_maxs = idx_maxs[1:]
            maxs = maxs[1:]
            return _np.array(idx_maxs), _np.array(maxs)


class Minima(_Tool):
    """
    Find all local minima in the signal

    Parameters
    ----------
    method : str
        Method to detect the minima. Available methods: 'complete' or 'windowing'. 'complete' finds all the local
        minima, 'windowing' uses a runnning window to find the global minima in each window.
    win_len : float, >0
        Length of window in seconds (method = 'windowing')
    win_step : float, >0
        Shift of the window to start the next window in seconds (method = 'windowing')

    Optional parameters
    -------------------
    refractory : float, >0, default = 0
        Seconds to skip after a detected minimum to look for new minima, when method = 'complete'. 

    Returns
    -------
    idx_mins : array
        Array containing indexes of the minima
    val_mins : array
        Array containing values of the minima
    """

    def __init__(self, method='complete', refractory=0, win_len=None, win_step=None):
        assert method in ['complete', 'windowing'], "Method not valid"
        assert refractory >= 0, "Refractory time value should be positive (or 0 to deactivate)"
        
        if method == 'windowing':
            assert win_len > 0, "Window length should be positive"
            assert win_step > 0, "Window step should be positive"
            _Tool.__init__(self, method=method, refractory=refractory, win_len=win_len, win_step=win_step)
        elif method == 'complete':
            _Tool.__init__(self, method=method, refractory=refractory)

    @classmethod
    def algorithm(cls, signal, params):
        idx_mins, mins = Maxima(**params)(-signal.copy())
        return idx_mins, -1 * mins


class CreateTemplate(_Tool):
    """
    Create a template for matched filtering
    
    Parameters
    ----------
    ref_indexes : array of int
        Indexes of the signals to be used as reference point to generate the template
    idx_start : int, >0
        Index of the signal to start the segmentation of the portion used to generate the template
    idx_stop : int, >0
        Index of the signal to end the segmentation of the portion used to generate the template
    smp_pre : int, >0
        Number of samples before the reference point to be used to generate the template
    smp_post : int, >0
        Number of samples after the reference point to be used to generate the template
    
    Returns
    -------
    template : numpy.array
        The template
    """

    def __init__(self, ref_indexes, smp_pre, smp_post, idx_start=0, idx_stop=None):
        ref_indexes = _np.array(ref_indexes)
        assert ref_indexes.ndim == 1, "Array of indices has to be 1-dimensional"
        assert smp_pre > 0, "Number of samples before the reference should be positive"
        assert smp_post > 0, "Number of samples after the reference should be positive"
        assert idx_start >= 0, "Start index should be non negative"
        assert idx_stop >= 0, "Stop index should be positive or 0 for end of the signal"
        _Tool.__init__(self, ref_indexes=ref_indexes, smp_pre=smp_pre, smp_post=smp_post, idx_start=idx_start,
                       idx_stop=idx_stop)

    @classmethod
    def algorithm(cls, signal, params):
        ref_indexes = params['ref_indexes']
        idx_start = params['idx_start']
        idx_end = params['idx_stop']
        if idx_end is None:
            idx_end = len(signal)
        smp_pre = params['smp_pre']
        smp_post = params['smp_post']

        sig = _np.array(signal[idx_start: idx_end])
        ref_indexes = _np.array(
            ref_indexes[_np.where((ref_indexes > idx_start) & (ref_indexes <= idx_end))[0]]) - idx_start
        total_samples = smp_pre + smp_post
        templates = _np.zeros(total_samples)

        for i in range(1, len(ref_indexes) - 1):
            idx_peak = ref_indexes[i]
            tmp = sig[idx_peak - smp_pre: idx_peak + smp_post]
            tmp = (tmp - _np.min(tmp)) / (_np.max(tmp) - _np.min(tmp))
            templates = _np.c_[templates, tmp]

        templates = templates[:, 1:]

        template = _np.mean(templates, axis=1)
        template = template - template[0]
        template = template / _np.sum(template)
        return template


class BootstrapEstimation(_Tool):
    """
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        Function to use in the bootstrapping. Must accept data as input
        
    Optional parameters
    -------------------
    
    n : int, >0, default = 100
        Number of iterations
    k : float, (0,1), default = 0.5
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        Bootstrapped estimate
    
    """

    def __init__(self, func, n=100, k=0.5):
        from types import FunctionType as Func
        assert isinstance(func, Func), "Parameter function should be a function (types.FunctionType)"
        assert n > 0, "n should be positive"
        assert 0 < k <= 1, "k should be between (0 and 1]"
        _Tool.__init__(self, func=func, n=n, k=k)

    @classmethod
    def algorithm(cls, signal, params):
        signal = _np.asarray(signal)
        l = len(signal)
        func = params['func']
        niter = int(params['n'])
        k = params['k']

        estim = []
        for i in range(niter):
            ixs = _np.arange(l)
            ixs_p = _np.random.permutation(ixs)
            sampled_data = signal[ixs_p[:int(round(k * l))]]
            curr_est = func(sampled_data)
            estim.append(curr_est)
        estim = _np.sort(estim)
        return estim[int(len(estim) / 2)]


class Durations(_Tool):
    """
    Compute durations of events starting from their start and stop indexes

    Parameters:
    -----------
    starts : array
        Start indexes along the data
    stops : array
        Stop indexes along the data

    Return:
    -------
    durations : array
        durations of the events
    """

    def __init__(self, starts, stops):
        starts = _np.array(starts)
        assert starts.ndim == 1
        stops = _np.array(stops)
        assert stops.ndim == 1
        _Tool.__init__(self, starts=starts, stops=stops)

    @classmethod
    def algorithm(cls, signal, params):
        starts = params["starts"]
        stops = params["stops"]

        fsamp = signal.get_sampling_freq()
        durations = []
        for I in range(len(starts)):
            if (stops[I] > 0) & (starts[I] >= 0):
                durations.append((stops[I] - starts[I]) / fsamp)
            else:
                durations.append(_np.nan)
        return durations


class Slopes(_Tool):
    """
    Compute rising slope of peaks

    Parameters:
    -----------
    starts : array
        Start of the peaks indexes
    peaks : array
        Peaks indexes

    Return:
    -------
    slopes : array
        Rising slopes the peaks
    """

    def __init__(self, starts, peaks):
        starts = _np.array(starts)
        assert starts.ndim == 1
        peaks = _np.array(peaks)
        assert peaks.ndim == 1
        _Tool.__init__(self, starts=starts, peaks=peaks)

    @classmethod
    def algorithm(cls, data, params):
        starts = params["starts"]
        peaks = params["peaks"]

        fsamp = data.get_sampling_freq()
        slopes = []
        for I in range(len(starts)):
            if peaks[I] > 0 & starts[I] >= 0:
                dy = data[peaks[I]] - data[starts[I]]
                dt = (peaks[I] - starts[I]) / fsamp
                slopes.append(dy / dt)
            else:
                slopes.append(_np.nan)
        return slopes


class BeatOutliers(_Tool):
    """
    Detects outliers in the IBI signal. 
    
    Optional parameters
    -------------------
    
    cache : int, >0,  default=3
        Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI
    
    Returns
    -------
    id_bad_ibi : numpy.array
        Identifiers of wrong beats
    
    Notes
    -----
    It only detects outliers. You should manually remove outliers using FixIBI
    
    """

    def __init__(self, ibi_median=0, cache=3, sensitivity=0.25):
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Tool.__init__(self, ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        cache, sensitivity, ibi_median = params["cache"], params["sensitivity"], params["ibi_median"]

        if ibi_median == 0:
            ibi_expected = float(_np.median(signal))
        else:
            ibi_expected = float(ibi_median)

        id_bad_ibi = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        for i in range(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)

            curr_ibi = ibi[i]

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            # missings.append([idx_ibi[i-1],idx_ibi[i]])

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                counter_bad = 0
            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0

        return id_bad_ibi


class FixIBI(_Tool):
    """
    Corrects the IBI series removing abnormal IBI
    
    Parameters
    ----------
    idx_bad_ibi : array
        Identifiers of abnormal beats
   
    Returns
    -------
    ibi : Unevenly Signal
        Corrected IBI
            
    """

    def __init__(self, idx_bad_ibi):
        idx_bad_ibi = _np.array(idx_bad_ibi)
        assert idx_bad_ibi.ndim == 1
        _Tool.__init__(self, id_bad_ibi=idx_bad_ibi)

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        assert isinstance(signal,
                          _UnevenlySignal), "IBI can only be represented by an UnevenlySignal, %s found." % type(signal)
        id_bad = params['id_bad_ibi']
        idx_ibi = signal.get_indices()
        ibi = signal.get_values()
        idx_ibi_nobad = _np.delete(idx_ibi, id_bad)
        ibi_nobad = _np.delete(ibi, id_bad)
        idx_ibi = idx_ibi_nobad.astype(int)
        ibi = ibi_nobad
        return _UnevenlySignal(values = ibi, 
                               sampling_freq = signal.get_sampling_freq(), 
                               start_time = signal.get_start_time(),
                               signal_type = signal.get_signal_type(), 
                               x_values=idx_ibi, x_type='indices', 
                               duration=signal.get_duration())


class BeatOptimizer(_Tool):
    """
    Optimize detection of errors in IBI estimation.
    
    Optional parameters
    -------------------

    B : float, >0, default = 0.25
        Ball radius in seconds to allow pairing between forward and backward beats
    cache : int, >0,  default = 3
        Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI

    Returns
    -------
    ibi : UnevenlySignal
        Optimized IBI signal

    Notes
    -----
        Bizzego et al., *DBD-RCO: Derivative Based Detection and Reverse Combinatorial Optimization 
        to improve heart beat detection for wearable devices for info about the algorithm*
    """

    def __init__(self, b=0.25, ibi_median=0, cache=3, sensitivity=0.25):
        assert b > 0, "Ball radius should be positive"
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Tool.__init__(self, B=b, ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        cls.error('Not implemented')
        return signal

        b, cache, sensitivity, ibi_median = params["B"], params["cache"], params["sensitivity"], params["ibi_median"]

        fsamp = signal.get_sampling_freq()

        # FORWARD
        id_bad_ibi_F = BeatOutliers(ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)(signal)

        # BACKWARD
        # reverse the signal
        idx_reverse = signal.get_indices()

        idx_reverse = -1 * (idx_reverse - idx_reverse[-1])[::-1]
        ibi_reverse = _np.diff(idx_reverse)
        ibi_reverse = _np.r_[ibi_reverse[0], ibi_reverse]

        ibi_reverse = _UnevenlySignal(ibi_reverse / fsamp, x_values=idx_reverse, x_type='indices',
                                      duration=signal.get_duration())

        id_bad_ibi_b = BeatOutliers(ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)(ibi_reverse)

        id_bad_ibi_B = -1 * (_np.array(id_bad_ibi_b) - len(ibi_reverse))[::-1]

        # idx_1
        ###
        # add indexes of idx_ibi_2 which are not in idx_ibi_1 but close enough
        b = b * fsamp
        for i_2 in range(1, len(idx_2)):
            curr_idx_2 = idx_2[i_2]
            if not (curr_idx_2 in idx_1):
                i_1 = _np.where((idx_1 >= curr_idx_2 - b) & (idx_1 <= curr_idx_2 + b))[0]
                if not len(i_1) > 0:
                    idx_1 = _np.r_[idx_1, curr_idx_2]
        idx_1 = _np.sort(idx_1)

        ###
        # create pairs for each beat
        pairs = [[0, 0]]
        for i_1 in range(1, len(idx_1)):
            curr_idx_1 = idx_1[i_1]
            if curr_idx_1 in idx_2:
                pairs.append([curr_idx_1, curr_idx_1])
            else:
                i_2 = _np.where((idx_2 >= curr_idx_1 - b) & (idx_2 <= curr_idx_1 + b))[0]
                if len(i_2) > 0:
                    i_2 = i_2[0]
                    pairs.append([curr_idx_1, idx_2[i_2]])
                else:
                    pairs.append([curr_idx_1, curr_idx_1])
        pairs = _np.array(pairs)

        ########################################
        # define zones where there are different values
        diff_idxs = pairs[:, 0] - pairs[:, 1]
        diff_idxs[diff_idxs != 0] = 1
        diff_idxs = _np.diff(diff_idxs)

        starts = _np.where(diff_idxs > 0)[0]
        stops = _np.where(diff_idxs < 0)[0]

        if len(starts) == 0:  # no differences
            # keep the 'outliers removed' version
            idx_1 = idx_1 + idx_st
            return _UnevenlySignal(ibi_1 / fsamp, sampling_freq=fsamp, signal_type="IBI",
                                   start_time=signal.get_start_time(), x_values=idx_1, x_type='indices',
                                   duration=signal.get_duration())

        if len(stops) == 0:
            stops = _np.array([starts[-1] + 1])

        if starts[0] >= stops[0]:
            stops = stops[1:]

        stops += 1

        if len(starts) > len(stops):
            stops = _np.r_[stops, starts[-1] + 1]

        # split long sequences
        new_starts = _np.copy(starts)
        new_stops = _np.copy(stops)

        add_index = 0
        lens = stops - starts
        for i in range(len(starts)):
            l = lens[i]
            if l > 10:
                curr_st = starts[i]
                curr_sp = stops[i]
                new_st = _np.arange(curr_st, curr_sp, 4)
                new_sp = new_st + 4
                new_sp[-1] = curr_sp
                new_starts = _np.delete(new_starts, i + add_index)
                new_stops = _np.delete(new_stops, i + add_index)
                new_starts = _np.insert(new_starts, i + add_index, new_st)
                new_stops = _np.insert(new_stops, i + add_index, new_sp)
                add_index = add_index + len(new_st) - 1

        starts = new_starts
        stops = new_stops

        ########################################
        # find best combination
        idx_out = _np.copy(pairs[:, 0])
        for i in range(len(starts)):
            i_st = starts[i]
            i_sp = stops[i]

            if i_sp > len(idx_out) - 1:
                i_sp = len(idx_out) - 1

            curr_portion = _np.copy(pairs[i_st - 1: i_sp + 1, :])

            best_portion = None
            best_error = _np.Inf

            combinations = list(_itertools.product([0, 1], repeat=i_sp - i_st - 1))
            for comb in combinations:
                cand_portion = _np.copy(curr_portion[:, 0])
                for k in range(len(comb)):
                    bit = comb[k]
                    cand_portion[k + 2] = curr_portion[k + 2, bit]
                cand_error = sum(abs(_np.diff(_np.diff(cand_portion))))
                if cand_error < best_error:
                    best_portion = cand_portion
                    best_error = cand_error
            idx_out[i_st - 1: i_sp + 1] = best_portion

        ###
        # finalize arrays
        idx_out = _np.array(idx_out) + idx_st
        ibi_out = _np.diff(idx_out)
        ibi_out = _np.r_[signal.get_values()[0], ibi_out / fsamp]

        return _UnevenlySignal(ibi_out, sampling_freq=signal.get_sampling_freq(), signal_type="IBI",
                               start_time=signal.get_start_time(), x_values=idx_out, x_type='indices',
                               duration=signal.get_duration())


# EDA Tools
class OptimizeBateman(_Tool):
    """
    Optimize the Bateman parameters T1 and T2.
    
    Parameters
    ----------
    delta : float
        Minimum amplitude of the peaks in the driver
        
    Optional parameters
    -------------------
    
    opt_method : str
        Method to perform the search of optimal parameters.
        Available methods:
    * 'bsh' Adaptive Simulated Annealing. Uses the Basin-Hopping algorithm.
    * 'grid' Grid search
    complete : boolean, default = True
        Whether to perform minimization after detecting the optimal parameters
    par_ranges : list, default = [0.1, 0.99, 1, 10]
        [min_T1, max_T1, min_T2, max_T2] boundaries for the Bateman parameters
    maxiter : int (Default = 99999)
        Maximum number of iterations ('asa' method)
    n_step_1 : int
        Number of steps in the grid search (paramter t1)
    n_step_2 : int
        Number of steps in the grid search (parameter t2)
    weight : str
        How the errors should be weighted before computing the loss function. ['exp', 'lin', 'none']
    min_pars : dict
        Additional parameters to pass to the minimization function (when complete = True)

    Returns
    -------
    x0 : list
        The resulting optimal parameters
    
    """

    # TODO (Feature): add **kwargs parameters for internal minimization
    def __init__(self, delta, loss_func='all', opt_method='bsh', complete=False, par_ranges=None,
                 maxiter=99999, n_step_1=10, n_step_2=10, **kwargs):
        if par_ranges is None:
            par_ranges = [0.1, 0.99, 1.5, 10]
        assert delta > 0
        assert loss_func in ['ben', 'all'], "Loss function not valid"
        assert opt_method in ['grid', 'bsh'], "Optimization method not valid"
        assert len(par_ranges) == 4
        if opt_method == "bsh":
            assert maxiter > 0
        if opt_method == "grid":
            assert n_step_1 > 0
            assert n_step_2 > 0

        _Tool.__init__(self, delta=delta, loss_func=loss_func, opt_method=opt_method, complete=complete,
                       par_ranges=par_ranges, maxiter=maxiter, n_step_1=n_step_1, n_step_2=n_step_2,
                       **kwargs)

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        opt_method = params['opt_method']
        complete = params['complete']
        par_ranges = params['par_ranges']
        maxiter = params['maxiter']
        n_step_1 = params['n_step_1']
        n_step_2 = params['n_step_2']

        # TODO (Andrea):explain "add **kwargs"

        if params['loss_func'] == 'ben':
            loss_function = OptimizeBateman._loss_benedek
        elif params['loss_func'] == 'all':
            loss_function = OptimizeBateman._loss_function_all

        if opt_method == 'grid':
            min_T1 = float(par_ranges[0])
            max_T1 = float(par_ranges[1])
    
            min_T2 = float(par_ranges[2])
            max_T2 = float(par_ranges[3])
            step_T1 = (max_T1 - min_T1) / n_step_1
            step_T2 = (max_T2 - min_T2) / n_step_2
            rranges = (slice(min_T1, max_T1 + step_T1, step_T1), slice(min_T2, max_T2 + step_T2, step_T2))
            x0, loss, grid, loss_grid = _opt.brute(loss_function, rranges,
                                                   args=(signal, delta),
                                                   full_output=True, finish=None)
            exit_code = -1

        elif opt_method == 'bsh':
            x_opt = _opt.basinhopping(loss_function, [0.75, 2.],
                                      niter=maxiter,
                                      minimizer_kwargs={
                                          "bounds": ((par_ranges[0], par_ranges[1]), (par_ranges[2], par_ranges[3])),
                                          "args": (signal, delta)},
                                      disp=False, niter_success=10)
            x0 = x_opt.x
            loss = float(x_opt.fun)

            if x_opt.minimization_failures == maxiter:
                exit_code = 1
            else:
                exit_code = 0
        else:
            cls.error("opt_method not understood")
            return None

        if complete:
            x0_min, loss_min, niter,\
            nfuncalls, warnflag, allvec = _opt.fmin(loss_function, x0,
                                                    args=(signal, delta),
                                                    full_output=True)
            return x0, x0_min, loss, loss_min, exit_code, warnflag
        else:
            return x0, loss, exit_code


    @staticmethod
    def _loss_function_all(par_bat, signal, delta):
        """
        Computes the loss for optimization of Bateman parameters.

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : Signal
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
       
        Returns
        -------
        loss : float
            The computed loss
        """

        from ..estimators.Estimators import DriverEstim as _DriverEstim
        # check if pars hit boudaries
        # if par_bat[0] < min_T1 or par_bat[0] > max_T1 or par_bat[1] < min_T2 or par_bat[1] > max_T2 or par_bat[0] >=
        #  par_bat[1]:
        #            return 10000

        REC_TIME = 20  #max recovery time
        if _np.isnan(par_bat[0]) | _np.isnan(par_bat[1]):
            return _np.Inf

        fsamp = signal.get_sampling_freq()

        driver = _DriverEstim(t1=par_bat[0], t2=par_bat[1])(signal)
        maxp, minp, ignored, ignored = PeakDetection(delta=delta, start_max=True)(driver.segment_time(driver.get_start_time(), driver.get_end_time() - REC_TIME ))
        
        if len(maxp) == 0:
            OptimizeBateman.warn('Unable to find peaks in driver signal for computation of Energy. Returning Inf')
            return _np.inf
        else:
            energy = 0

            for idx_max in maxp:
                # extract WLEN seconds after the peak
                driver_portion = driver.segment_idx(idx_max, idx_max + REC_TIME * fsamp)
                
                # find angular coefficient (average derivative)
                

                diff_y = Diff()(driver_portion)
                th_66 = _np.percentile(diff_y, 66)
                th_33 = _np.percentile(diff_y, 33)

                idx_sel_diff_y = _np.where((diff_y > th_33) & (diff_y < th_66))[0]
                diff_y_sel = diff_y.get_values()[idx_sel_diff_y]
                mean_s = BootstrapEstimation(func=_np.mean, n=10, k=0.5)(_UnevenlySignal(diff_y_sel, sampling_freq = fsamp, x_values = idx_sel_diff_y, x_type = 'indices'))
                
                # find intercept
                mean_y = BootstrapEstimation(func=_np.median, n=10, k=0.5)(driver_portion)
                b_mean_s = mean_y - mean_s * len(driver_portion) / 2

                # compute linear trend
                line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s

                # detrend
                driver_detrended = driver_portion - line_mean_s

                # normalize detrended driver
                driver_detrended /= float(_np.max(driver_detrended))

                energy_curr = (1 / fsamp) * _np.sum(driver_detrended[1:] ** 2) / (len(driver_detrended) - 1)
                energy += energy_curr
            
            energy /= len(maxp)
            OptimizeBateman.log(
                '\r\r ALL. Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(
                    energy) + '\r')
            if _np.isnan(energy):
                return _np.Inf

            return energy

    @staticmethod
    def _loss_benedek(par_bat, signal, delta):
        """
        Computes the loss for optimization of Bateman parameters according to Benedek2010 (REF).
        #TODO: insert reference

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : _Signal
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
       
        Returns
        -------
        loss : float
            The computed loss
        """

        from ..estimators.Estimators import DriverEstim as _DriverEstim

        if _np.isnan(par_bat[0]) | _np.isnan(par_bat[1]):
            return _np.Inf

        def phasic_estim_benedek(driver, delta):
            # find peaks in the driver
            fsamp = driver.get_sampling_freq()
            i_peaks, idx_min, val_max, val_min = PeakDetection(delta=delta, refractory=1, start_max=True)(driver)

            i_pre_max = 10 * fsamp
            i_post_max = 10 * fsamp

            i_start = _np.empty(len(i_peaks), int)
            i_stop = _np.empty(len(i_peaks), int)

            if len(i_peaks) == 0:
                OptimizeBateman.warn('No peaks found.')
                return driver

            for i in range(len(i_peaks)):
                i_pk = int(i_peaks[i])

                # find START
                i_st = i_pk - i_pre_max
                if i_st < 0:
                    i_st = 0

                driver_pre = driver[i_st:i_pk]
                i_pre = len(driver_pre) - 2

                while i_pre > 0 and (driver_pre[i_pre] >= driver_pre[i_pre - 1]):
                    i_pre -= 1

                i_start[i] = i_st + i_pre + 1

                # find STOP
                i_sp = i_pk + i_post_max
                if i_sp >= len(driver):
                    i_sp = len(driver) - 1

                driver_post = driver[i_pk: i_sp]
                i_post = 1

                while i_post < len(driver_post) - 2 and (driver_post[i_post] >= driver_post[i_post + 1]):
                    i_post += 1

                i_stop[i] = i_pk + i_post

            idxs_peak = _np.array([])

            for i_st, i_sp in zip(i_start, i_stop):
                idxs_peak = _np.r_[idxs_peak, _np.arange(i_st, i_sp)]

            idxs_peak = idxs_peak.astype(int)

            # generate the grid for the interpolation
            idx_grid_candidate = _np.arange(0, len(driver) - 1, 10 * fsamp)

            idx_grid = []
            for i in idx_grid_candidate:
                if i not in idxs_peak:
                    idx_grid.append(i)

            if len(idx_grid) == 0:
                idx_grid.append(0)

            if idx_grid[0] != 0:
                idx_grid = _np.r_[0, idx_grid]
            if idx_grid[-1] != len(driver) - 1:
                idx_grid = _np.r_[idx_grid, len(driver) - 1]

            driver_grid = _UnevenlySignal(driver[idx_grid], fsamp, "dEDA", driver.get_start_time(), x_values=idx_grid,
                                          x_type='indices', duration=signal.get_duration())
            if len(idx_grid) >= 4:
                tonic = driver_grid.to_evenly(kind='cubic')
            else:
                tonic = driver_grid.to_evenly(kind='linear')
            phasic = driver - tonic
            return phasic

        ALPHA = 6
        driver = _DriverEstim(t1=par_bat[0], t2=par_bat[1])(signal)

        phasic = phasic_estim_benedek(driver, delta)

        TH = float(_np.max(phasic) * 0.05)

        peaks = _np.zeros(len(phasic))
        peaks[phasic > TH] = 1

        # compute indistinctness
        d_peaks_dt = _np.diff(peaks)

        starts = _np.where(d_peaks_dt == 1)[0]
        ends = _np.where(d_peaks_dt == -1)[0]

        if (len(starts) == 0) | (len(ends) == 0):
            indist = _np.Inf
        else:
            if ends[0] < starts[0]:  # phasic starts with a peak
                starts = _np.r_[0, starts]

            if ends[-1] < starts[-1]:  # phasic ends with a peak
                ends = _np.r_[ends, len(peaks) - 1]

            # at this point is should be len(starts)==len(ends)
            indist = _np.sum((ends - starts) ** 2) / phasic.get_duration()

        # compute negativeness
        negatives = phasic[phasic < 0]
        neg = _np.sqrt(_np.mean(negatives ** 2))

        # compute loss
        LOSS = indist + ALPHA * neg
        OptimizeBateman.log(
            '\r\r BENEDEK. Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(
                LOSS) + '\r')
        return LOSS
