from __future__ import division
import numpy as _np
import asa as _asa
from scipy.signal import _welch
import scipy.optimize as _opt
from spectrum import _Periodogram, _aryule, _arma2psd, _AIC
from pyPhysio import ConvolutionalFilter as _ConvFlt, PhUI as _PhUI
from ..Parameters import Parameter as _Par
import itertools as _itertools
from ..BaseTool import Tool as _Tool
from ..Signal import UnevenlySignal as _UnevenlySignal, EvenlySignal as _EvenlySignal
from ..filters.Filters import Diff as _Diff
from ..estimators.Estimators import DriverEstim as _DriverEstim


class PeakDetection(_Tool):
    """
    Estimate the maxima and the minima in the signal (in particular for periodic signals).

    Parameters
    ----------
    delta : float or np.array
        The threshold for the detection of the peaks. If array it must have the same length of the signal.
    refractory : int
        Number of samples to skip after detection of a peak
    start_max : boolean
        Whether to start looking for a max.

    Returns
    -------
    maxs : nparray
        Array containing indexes (first column) and values (second column) of the maxima
    mins : nparray
        Array containing indexes (first column) and values (second column) of the minima
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta = params['delta']
        refractory = params['refractory']
        start_max = params['start_max']

        if len(delta) != len(signal):
            _PhUI.e("delta vector's length differs from signal's one.")
            return _np.array([[], []]), _np.array([[], []])

        mins = []
        maxs = []

        # initialization
        mn_candidate, mx_candidate = _np.Inf, -_np.Inf
        mn_pos_candidate, mx_pos_candidate = _np.nan, _np.nan

        look_for_max = start_max

        i_activation_max = 0
        i_activation_min = 0

        i = 0
        while i <= len(signal) - 1:
            this = signal[i]
            dd = delta[i]

            if this > mx_candidate:
                mx_candidate = this
                mx_pos_candidate = i
            if this < mn_candidate:
                mn_candidate = this
                mn_pos_candidate = i

            if look_for_max:
                if i >= i_activation_max and this < mx_candidate - dd:  # new max
                    maxs.append((mx_pos_candidate, mx_candidate))
                    i_activation_max = i + refractory

                    mn_candidate = this
                    mn_pos_candidate = i

                    look_for_max = False
            else:
                if i >= i_activation_min and this > mn_candidate + dd:  # new min
                    mins.append((mn_pos_candidate, mn_candidate))
                    i_activation_min = i + refractory

                    mx_candidate = this
                    mx_pos_candidate = i

                    look_for_max = True
            i += 1
        return _np.array(maxs), _np.array(mins)

    _params_descriptors = {
        'delta': _Par(2, (int, float),
                      "The threshold for the detection of the peaks.",
                      0,
                      lambda x: x > 0,
                      lambda x, y: 'deltas' not in y),
        'deltas': _Par(2, _np.array,
                       "Vector of the ranges of the signal to be used as local threshold",
                       None,
                       lambda x, y: 'delta' not in y),
        'refractory': _Par(1, _np.int64,
                           "Number of samples to skip after detection of a peak",
                           lambda x: x > 0),
        'start_max': _Par(0, bool, "Whether to start looking for a max.", True)
    }


class PeakSelection(_Tool):
    """
    Identifies the start and end indexes of each peak in the signal, using derivatives.

    Parameters
    ----------
    maxs : nparray
        Array containing indexes (first column) and values (second column) of the maxima
    pre_max : float
        Duration (in seconds) of interval before the peak that is considered to find the start of the peak
    post_max : float
        Duration (in seconds) of interval after the peak that is considered to find the end of the peak
    
    Returns
    -------
    starts : nparray
        Array containing start indexes
    ends : nparray
        Array containing end indexes
    """

    @classmethod
    def algorithm(cls, signal, params):
        maxs = params['maxs']
        i_pre_max = params['pre_max'] * signal.sampling_freq
        i_post_max = params['post_max'] * signal.sampling_freq
        i_peaks = maxs[:, 0].astype(int)

        if _np.shape(maxs)[0] == 0:
            _PhUI.e('No peaks found in the input')
            return _np.array([]), _np.array([])
        else:
            signal_dt = _Diff()(signal)
            i_start = []
            i_stop = []
            for i_pk in i_peaks:
                if (i_pk < i_pre_max) or (i_pk >= len(signal_dt) - i_post_max):
                    _PhUI.w('Peak at start/end of signal, not accounting')
                    i_start.append(_np.nan)
                    i_stop.append(_np.nan)
                else:
                    # TODO: vedere se si puo scrivere un codice meno complesso
                    # TODO: gestire se sorpasso gli intervalli e non ho trovato il minimo

                    # find START
                    i_st = i_pk - i_pre_max
                    signal_dt_pre = signal_dt[i_st: i_pk]
                    i_pre = len(signal_dt_pre) - 1

                    done = False
                    while not done:
                        if signal_dt_pre[i_pre] <= 0:
                            done = True
                        else:
                            i_pre -= 1
                            if i_pre < 0:
                                done = True
                                i_pre = 0

                    i_pre_true = i_st + i_pre + 1
                    i_start.append(i_pre_true)

                    # find STOP
                    i_sp = i_pk + i_post_max
                    signal_dt_post = signal_dt[i_pk: i_sp]
                    i_post = 0

                    done = False
                    while not done:
                        if signal_dt_post[i_post] >= 0:
                            done = True
                        else:
                            i_post += 1
                            if i_post == len(signal_dt_post):
                                done = True
                                i_post = len(signal_dt_post)

                    i_post_true = i_pk + i_post
                    i_stop.append(i_post_true)

            return _np.array(i_start).astype(int), _np.array(i_stop).astype(int)

    _params_descriptors = {
        'maxs': _Par(2, list,
                     'Array containing indexes (first column) and values (second column) of the maxima'),
        'pre_max': _Par(2, (int, float),
                        'Duration (in seconds) of interval before the peak that is considered to find the start of the peak',
                        1, lambda x: x > 0),
        'post_max': _Par(2, (int, float),
                         'Duration (in seconds) of interval after the peak that is considered to find the start of the peak',
                         1, lambda x: x > 0)
    }


class SignalRange(_Tool):
    """
    Estimate the local range of the signal by sliding windowing

    Parameters
    ----------
    win_len : int
        The length of the window (seconds)
    win_step : int
        The increment to start the next window (seconds)
    smooth : boolean
        Whether to convolve the result with a gaussian window

    Returns
    -------
    deltas : nparray
        Result of estimation of local range
    """

    @classmethod
    def algorithm(cls, signal, params):
        win_len = params['win_len']
        win_step = params['win_step']
        smooth = params['smooth']

        fsamp = signal.sampling_freq
        idx_len = int(win_len * fsamp)
        idx_step = int(win_step * fsamp)

        windows = _np.arange(0, len(signal) - idx_len, idx_step)
        deltas = _np.zeros(len(signal))
        
        # ---
        curr_delta = None
        # ---
        for start in windows:
            portion_curr = signal[start: start + idx_len]
            curr_delta = (_np.max(portion_curr) - _np.min(portion_curr))
            deltas[start:start + idx_len] = curr_delta
        # ---
        start = windows[-1]
        # ---
        deltas[start + idx_len:] = curr_delta

        if smooth:
            deltas = _ConvFlt(irftype='gauss', win_len=win_len * 2, normalize=True)(deltas)  # TODO: check sintax

        return deltas

    _params_descriptors = {
        'win_len': _Par(2, int, 'The length of the window (seconds)', 1, lambda x: x > 0),
        'win_step': _Par(2, int, 'The increment to start the next window (seconds)', 1, lambda x: x > 0),
        'smooth': _Par(1, bool, 'Whether to convolve the result with a gaussian window', True)
    }


class PSD(_Tool):
    """
    Estimate the power spectral density (PSD) of the signal.

    Parameters
    ----------
    psd_method : str
        Method to estimate the PSD
    nfft : int
        Number of samples of the PSD
    window : str
        Type of window
    remove_mean : boolean
        Whether to remove the mean from the signal
    min_order : int (default=10)
        Minimum order of the model (for psd_method='ar')
    max_order : int (defaut=25)
        Maximum order of the model (for psd_method='ar')
    normalize : boolean
        Whether to normalize the PSD

    Returns
    -------
    freq : nparray
        The frequencies
    psd : nparray
        The Power Spectrum Density
    """

    # TODO: consider point below:
    '''
    A density spectrum considers the amplitudes per unit frequency. 
    Density spectra are used to compare spectra with different frequency resolution as the 
    magnitudes are not influenced by the resolution because it is per Hertz. The amplitude 
    spectra on the other hand depend on the chosen frequency resolution.
    '''

    @classmethod
    def algorithm(cls, signal, params):
        method = params['psd_method']
        nfft = params['nfft']
        window = params['window']
        normalize = params['normalize']
        remove_mean = params['remove_mean']
        
        fsamp = signal.sampling_freq

        # TODO: check signal type.
        # TODO: if unevenly --> interpolate
        if not isinstance(signal, _EvenlySignal):
            signal = signal.to_evenly()

        l = len(signal)
        if remove_mean:
            signal = signal - _np.mean(signal)

        if window == 'hamming':
            win = _np.hamming(l)
        elif window == 'blackman':
            win = _np.blackman(l)
        elif window == 'bartlett':
            win = _np.bartlett(l)
        elif window == 'hanning':
            win = _np.hanning(l)
        elif window == 'none':
            win = _np.ones(l)
        else:
            _PhUI.w('Window type not understood, using none.')
            win = _np.ones(l)

        signal = signal * win
        if method == 'fft':
            spec_tmp = _np.abs(_np.fft.fft(signal, n=nfft)) ** 2  # FFT
            psd = spec_tmp[0:(_np.ceil(len(spec_tmp) / 2))]

        elif method == 'welch':
            bands_w, psd = _welch(signal, fsamp, nfft=nfft)

        elif method == 'ar':
            min_order = params['min_order']
            max_order = params['max_order']

            orders = range(min_order, max_order + 1)
            AICs = []
            for order in orders:
                try:
                    ar, p, k = _aryule(signal, order=order, norm='biased')
                    AICs.append(_AIC(l, p, order))
                except AssertionError:  # TODO (Andrea): check whether to start from higher orders
                    break
            best_order = orders[_np.argmin(AICs)]

            ar, p, k = _aryule(signal, best_order, norm='biased')
            psd = _arma2psd(ar, NFFT=nfft)
            psd = psd[0: _np.ceil(len(psd) / 2)]

        freqs = _np.linspace(start=0, stop=fsamp / 2, num=len(psd), endpoint=True)

        # NORMALIZE
        if normalize:
            psd /= 0.5 * fsamp * _np.sum(psd) / len(psd)
        return freqs, psd

    _method_list = ['welch', 'fft', 'ar']
    _window_list = ['hamming', 'blackman', 'hanning', 'none']

    _params_descriptors = {
        'method': _Par(1, str, 'Method to estimate the PSD', 'welch', lambda x: x in PSD._method_list),
        'min_order': _Par(1, int, 'Minimum order of the model (for method="ar")',
                          10,
                          lambda x: x > 0,
                          lambda x, y: y["method"] == "ar"),
        'nfft': _Par(1, int, 'Number of samples in the PSD', 2048, lambda x: x > 0),
        'window': _Par(1, str, 'Type of window to adapt the signal before estimation of the PSD',
                       'hamming', lambda x: x in PSD._window_list),
        'normalize': _Par(1, bool, 'Whether to normalize the PSD', True),
        'remove_mean': _Par(1, bool, 'Whether to remove the mean from the signal before estimation of the PSD', True)
    }


class Energy(_Tool):
    """
    Estimate the local energy of the signal

    Parameters
    ----------
    win_len : int
        The dimension of the window    
    win_step : int
        The increment indexes to start the next window
    smooth : boolean
        Whether to convolve the result with a gaussian window

    Returns
    -------
    energy : nparray
        Result of estimation of local energy
    """

    @classmethod
    def algorithm(cls, signal, params):
        fsamp = signal.sampling_freq
        win_len = params['win_len']
        win_step = params['win_step']
        idx_len =  win_len * fsamp
        idx_step = win_step * fsamp
        smooth = params['smooth']

        windows = _np.arange(0, len(signal) - idx_len, idx_step)

        energy = []
        curr_energy = None
        for start in windows:
            portion_curr = signal[start: start + idx_len]
            curr_energy = _np.sum(_np.power(portion_curr, 2)) / len(portion_curr)
            energy.append(curr_energy)
        energy.append(curr_energy)
        energy.insert(0, energy[0])

        idx_interp = _np.r_[0, windows + round(idx_len / 2), len(signal)]
        energy = _np.array(energy)
        # TODO: assumed ", 1," was the wanted fsamp
        # WAS: energy_out = flt.interpolate_unevenly(energy, idx_interp, 1, kind='linear')
        energy_out = _UnevenlySignal(energy, idx_interp, 1).to_evenly(kind='linear').get_y_values()

        if smooth:
            energy_out = _ConvFlt(irftype='gauss', win_len = 2 * win_len, normalize=True)(energy_out)

        return energy_out

    _params_descriptors = {
        'win_len': _Par(2, int, 'The length of the window (seconds)', 1, lambda x: x > 0),
        'win_step': _Par(2, int, 'The increment to start the next window (seconds)', 1, lambda x: x > 0),
        'smooth': _Par(1, bool, 'Whether to convolve the result with a gaussian window', True)
    }


class Maxima(_Tool):
    """
    Find all local maxima in the signal

    Parameters
    ----------
    method : 'complete' or 'windowing'
        Method to detect the maxima
    refractory : int
        Number of samples to skip after detection of a maximum (method = 'complete')
    win_len : int
        Size of window in seconds (method = 'windowing')
    win_step : int
        Increment to start the next window in seconds (method = 'windowing')

    Returns
    -------
    maxs : nparray
        Array containing indexes (first column) and values (second column) of the maxima
    """

    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        refractory = params['refractory']
        if method == 'complete':
            maxima = []
            prev = signal[0]
            k = 1
            while k < len(signal) - 1:
                curr = signal[k]
                nxt = signal[k + 1]
                if (curr >= prev) and (curr >= nxt):
                    maxima.append(k)
                    prev = signal[k + 1 + refractory]
                    k = k + 2 + refractory
                else:  # continue
                    prev = signal[k]
                    k += 1
            maxima = _np.array(maxima).astype(int)
            peaks = signal[maxima]
            return _np.c_[maxima, peaks]

        elif method == 'windowing':
            # TODO: test the algorithm
            fsamp = signal.sampling_freq
            wlen = int(params['win_len'] * fsamp)
            wstep = int(params['win_step'] * fsamp)

            idx_maxs = [0]
            maxs = [0]

            idx_start = _np.arange(0, len(signal), wstep)
            for idx_st in idx_start:
                idx_sp = idx_st + wlen
                if idx_sp > len(signal):
                    idx_sp = len(signal)
                curr_win = signal[idx_st:idx_sp]
                curr_idx_max = _np.argmax(curr_win) + idx_st
                curr_max = _np.max(curr_win)

                if curr_idx_max != idx_maxs[
                    -1] and curr_idx_max != idx_st and curr_idx_max != idx_sp - 1:  # peak not already detected & peak not at the beginnig/end of the window:
                    idx_maxs.append(curr_idx_max)
                    maxs.append(curr_max)
            idx_maxs.append(len(signal) - 1)
            maxs.append(signal[-1])
            return _np.c_[idx_maxs, maxs]

    _params_descriptors = {
        'method': _Par(2, 'Method to detect the maxima',
                       'complete',
                       lambda x: x in ['complete', 'windowing']),
        'refractory': _Par(1, 'Number of samples to skip after detection of a maximum (method = "complete")',
                           1,
                           lambda x: x > 0,
                           lambda x, p: p['method'] == 'complete'),
        'win_len': _Par(1, 'Size of window in seconds (method = "windowing")',
                        1,
                        lambda x: x > 0,
                        lambda x, p: p['method'] == 'windowing'),
        'win_step': _Par(1, 'Increment to start the next window in seconds (method = "windowing")',
                         1,
                         lambda x: x > 0,
                         lambda x, p: p['method'] == 'windowing'),
    }


class Minima(_Tool):
    """
    Find all local minima in the signal

    Parameters
    ----------
    method : 'complete' or 'windowing'
        Method to detect the minima
    refractory : int
        Number of samples to skip after detection of a maximum (method = 'complete')
    win_len : int
        Size of window (method = 'windowing')
    win_step : int
        Steps to start the next of window (method = 'windowing')

    Returns
    -------
    mins : nparray
        Array containing indexes (first column) and values (second column) of the minima
    """

    @classmethod
    def algorithm(cls, signal, params):
        method = params['method']
        refractory = params['refractory']
        if method == 'complete':
            minima = []
            prev = signal[0]
            k = 1
            while k < len(signal) - 1:
                curr = signal[k]
                nxt = signal[k + 1]
                if (curr <= prev) and (curr <= nxt):
                    minima.append(k)
                    prev = signal[k + 1 + refractory]
                    k = k + 2 + refractory
                else:  # continue
                    prev = signal[k]
                    k += 1
            minima = _np.array(minima).astype(int)
            peaks = signal[minima]
            return _np.c_[peaks, minima]

        elif method == 'windowing':
            # TODO: test the algorithm
            winlen = params['win_len']
            winstep = params['win_step']

            idx_mins = [0]
            mins = [0]

            idx_starts = _np.arange(0, len(signal), winstep)
            for idx_st in idx_starts:
                idx_sp = idx_st + winlen
                if idx_sp > len(signal):
                    idx_sp = len(signal)
                curr_win = signal[idx_st:idx_sp]
                curr_idx_min = _np.argmin(curr_win) + idx_st
                curr_min = _np.min(curr_win)

                if curr_idx_min != idx_mins[
                    -1] and curr_idx_min != idx_st and curr_idx_min != idx_sp - 1:  # peak not present & peak not at the beginnig/end of the window
                    idx_mins.append(curr_idx_min)
                    mins.append(curr_min)
            idx_mins.append(len(signal) - 1)
            mins.append(signal[-1])
            return _np.c_[idx_mins, mins]

    _params_descriptors = {
        'method': _Par(2, 'Method to detect the minima',
                       'complete',
                       lambda x: x in ['complete', 'windowing']),
        'refractory': _Par(1, 'Number of samples to skip after detection of a maximum (method = "complete")',
                           1,
                           lambda x: x > 0,
                           lambda x, p: p['method'] == 'complete'),
        'win_len': _Par(1, 'Size of window in seconds (method = "windowing")',
                        1,
                        lambda x: x > 0,
                        lambda x, p: p['method'] == 'windowing'),
        'win_step': _Par(1, 'Increment to start the next window in seconds (method = "windowing")',
                         1,
                         lambda x: x > 0,
                         lambda x, p: p['method'] == 'windowing'),
    }


class CreateTemplate(_Tool):
    """
    Create a template for matched filtering
    
    Parameters
    ----------
    ref_indexes : nparray of int
        Indexes of the signals to be used as reference point to generate the template
    idx_start : int
        Index of the signal to start the segmentation of the portion used to generate the template
    idx_end : int
        Index of the signal to end the segmentation of the portion used to generate the template
    smp_pre : int
        Number of samples before the reference point to be used to generate the template
    smp_post : int
        Number of samples after the reference point to be used to generate the template
    
    Returns
    -------
    template : nparray
        The template
    """

    @classmethod
    def algorithm(cls, signal, params):
        idx_start = params['idx_start']
        idx_end = params['idx_stop']
        smp_pre = params['smp_pre']
        smp_post = params['smp_post']
        ref_indexes = params['ref_indexes']
        sig = _np.array(signal[idx_start:idx_end])
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

    _params_descriptors = {
        'ref_indexes': _Par(2, _np.ndarray,
                            'Indexes of the signals to be used as reference point to generate the template', None),
        'idx_start': _Par(2, int,
                          'Index of the signal to start the segmentation of the portion used to generate the template',
                          None, lambda x: x > 0),
        'idx_stop': _Par(2, int,
                         'Index of the signal to end the segmentation of the portion used to generate the template',
                         None, lambda x: x > 0),
        'smp_pre': _Par(2, int, 'Number of samples before the reference point to be used to generate the template',
                        None, lambda x: x > 0),
        'smp_post': _Par(2, int, 'Number of samples after the reference point to be used to generate the template',
                         None, lambda x: x > 0)
    }


class BootstrapEstimation(_Tool):
    """
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        The estimator. Must accept data as input
    N : int
        Number of iterations
    k : float (0-1)
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        The bootstrapped estimate
    """

    @classmethod
    def algorithm(cls, signal, params):
        l = len(signal)
        func = params['func']
        niter = params['N']
        k = params['k']

        estim = []
        for i in range(niter):
            ixs = _np.arange(l)
            ixs_p = _np.random.permutation(ixs)
            sampled_data = signal[ixs_p[:round(k * l)]]
            curr_est = func(sampled_data)
            estim.append(curr_est)
        return _np.mean(estim)

    from types import FunctionType as func
    _params_descriptors = {
        'func': _Par(2, func, 'Function (accepts as input a vector and returns a scalar).'),
        'N': _Par(1, int, 'Number of iterations', 100, lambda x: x > 0),
        'k': _Par(1, (int, float), 'Portion of data to be used at each iteration', 0.5, lambda x: 0 < x < 1)
    }


### IBI Tools
class BeatOutliers(_Tool):
    """
    Detects outliers in IBI signal.
    Returns id of outliers.
    
    Parameters
    ----------
    cache : int (default = 3)
        Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, optional
        Relative variation from the current median that is accepted
    ibi_median : float, optional
        If given it is used to initialize the cache
    
    Returns
    -------
    id_bad_ibi : nparray
        Identifiers of wrong beats
    
    Notes
    -----
    It only detects outliers. You should manually remove outliers using FixIBI
    
    """

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        cache, sensitivity, ibi_median = params["cache"], params["sensitivity"], params["ibi_median"]

        if ibi_median == 0:
            ibi_expected = _np.median(signal)
        else:
            ibi_expected = ibi_median
        id_bad_ibi = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        # TODO (Ale): giusta la sintassi?
        idx_ibi = signal.x_values
        ibi = signal.y_values
        for i in range(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi[i]
            curr_ibi = ibi[i]

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            # missings.append([idx_ibi[i-1],idx_ibi[i]])

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[-cache + 1:], curr_ibi]
                counter_bad = 0  # TODO: check

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # MESSAGE 'Cache re-initialized - '+ str(curr_idx)
                counter_bad = 0

        return id_bad_ibi

    _params_descriptors = {
        'ibi_median': _Par(1, (int, float),
                           'Ibi value used to initialize the cache. If 0 (default) the ibi_median is computed on the input signal',
                           0, lambda x: x >= 0),
        'cache': _Par(1, int,
                      'Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values',
                      3, lambda x: x > 0),
        'sensitivity': _Par(1, (int, float), 'Relative variation from the current median that is accepted', 0.25,
                            lambda x: x > 0)
    }


class FixIBI(_Tool):
    """
    Corrects the IBI series removing abnormal IBI
    
    Parameters
    ----------
    id_bad_ibi : nparray
        Identifiers of abnormal beats
   
    Returns
    -------
    ibi : Unevenly Signal
        Corrected IBI
            
    """

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        assert isinstance(signal, _UnevenlySignal), "IBI can only be represented by an EvenlySignal, %s found." % type(
            signal)
        id_bad = params['id_bad_ibi']
        idx_ibi = signal.get_x_values()
        ibi = signal.get_y_values()
        idx_ibi_nobad = _np.delete(idx_ibi, id_bad)
        ibi_nobad = _np.delete(ibi, id_bad)
        idx_ibi = idx_ibi_nobad.astype(int)
        ibi = ibi_nobad
        return _UnevenlySignal(ibi, idx_ibi, signal.get_sampling_freq(), len(signal), signal.get_signal_nature(),
                               signal.get_start_time())

    _params_descriptors = {
        'id_bad_ibi': _Par(2, list, 'Identifiers of abnormal beats')
    }


class BeatOptimizer(_Tool):
    """
    Optimize detection of errors in IBI estimation.
    
    Parameters
    ----------
    B : float
        Ball radius (in seconds)
    cache : int (default=)
        Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float (default=)
        Relative variation from the current median that is accepted
    ibi_median : float (default=0, computed from input data)
        If given it is used to initialize the cache
        
    Returns
    -------
    ibi : Unevenly Signal
        Optimized IBI signal

    Notes
    -----
    ...        
    """

    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        b, cache, sensitivity, ibi_median = params["B"], params["cache"], params["sensitivity"], params["ibi_median"]

        idx_ibi = signal.indexes
        fsamp = signal.fsamp

        if ibi_median == 0:
            ibi_expected = _np.median(_Diff()(idx_ibi))
        else:
            ibi_expected = ibi_median

        idx_st = idx_ibi[0]
        idx_ibi = idx_ibi - idx_st

        ###
        # RUN FORWARD:
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_1 = [idx_ibi[0]]
        ibi_1 = []

        prev_idx = idx_ibi[0]
        for i in range(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi[i]
            curr_ibi = curr_idx - prev_idx

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_1.append(_np.nan)
                idx_1.append(curr_idx)
                counter_bad += 1
            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_1.append(curr_ibi)
                idx_1.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # message
                counter_bad = 0

        ###
        # RUN BACKWARD:
        idx_ibi_rev = idx_ibi[-1] - idx_ibi
        idx_ibi_rev = idx_ibi_rev[::-1]

        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_2 = [idx_ibi_rev[0]]
        ibi_2 = []

        prev_idx = idx_ibi_rev[0]
        for i in xrange(1, len(idx_ibi_rev)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi_rev[i]
            curr_ibi = curr_idx - prev_idx

            # print([curr_median*(1+sensitivity), curr_median*(1-sensitivity), curr_median])
            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_2.append(_np.nan)
                idx_2.append(curr_idx)
                counter_bad += 1

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_2.append(curr_ibi)
                idx_2.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # OK Message
                counter_bad = 0

        idx_2 = -1 * (_np.array(idx_2) - idx_ibi_rev[-1])
        idx_2 = idx_2[::-1]
        ibi_2 = ibi_2[::-1]

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
        pairs = []
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
        stops = _np.where(diff_idxs < 0)[0] + 1

        # TODO (Ale): is stops an array? It should
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
        ibi_out = _Diff()(idx_out)
        ibi_out = _np.r_[ibi_out[0], ibi_out]

        return _UnevenlySignal(ibi_out, idx_out, signal.get_sampling_freq(), len(signal), "IBI",
                               signal.get_start_time(), meta=signal.meta)

    _params_descriptors = {
        'B': _Par(1, (int, float), 'Ball radius (in seconds) to detect paired beats', 0.25, lambda x: x > 0),
        'ibi_median': _Par(1, int,
                           'Ibi value used to initialize the cache. If 0 (default) the ibi_median is computed on the input signal',
                           0, lambda x: x > 0),
        'cache': _Par(1, int,
                      'Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values',
                      3, lambda x: x > 0),
        'sensitivity': _Par(1, (int, float), 'Relative variation from the current median that is accepted', 0.25,
                            lambda x: x > 0)
    }


### EDA Tools
class OptimizeBateman(_Tool):
    """
    Optimize the Bateman parameters T1 and T2.
    
    Parameters
    ----------
    opt_method : 'asa' or 'grid'
        Method to perform the search of optimal parameters.
        'asa' Adaptive Simulated Annealing (CITE)
        'grid' Grid search
    complete : boolean
        Whether to perform a minimization after detecting the optimal parameters
    par_ranges : list
        [min_T1, max_T1, min_T2, max_T2] boundaries for the Bateman parameters
    maxiter : int (Default = 99999)
        Maximum number of iterations ('asa' method)
    n_step : int
        Number of increments in the grid search
    delta : float
        Minimum amplitude of the peaks in the driver
    min_pars : dict
        Additional parameters to pass to the minimization function (when complete = True)

    Returns
    -------
    x0 : list
        The resulting optimal parameters
    x0_min : list
        If complete = True, parameters resulting from the 'asa' o 'grid' search
    """

    @classmethod
    def algorithm(cls, signal, params):
        opt_method = params['opt_method']
        complete = params['complete']
        par_ranges = params['par_ranegs']
        maxiter = params['maxiter']
        n_step = params['n_step']
        delta = params['delta']
        min_pars = params['fmin_params']

        min_T1 = par_ranges[0]
        max_T1 = par_ranges[1]

        min_T2 = par_ranges[2]
        max_T2 = par_ranges[3]

        # TODO (Ale): _loss_function is the function below

        if opt_method == 'asa':
            T1 = 0.75
            T2 = 2
            if maxiter == 0:
                maxiter = 99999
            x0, loss_x0, exit_code, asa_opts = _asa.asa(_loss_function, _np.array([T1, T2]),
                                                        xmin=_np.array([min_T1, min_T2]),
                                                        xmax=_np.array([max_T1, max_T2]), full_output=True,
                                                        limit_generated=maxiter, args=(signal, delta, min_T1, max_T2))
        elif opt_method == 'grid':
            step_T1 = (max_T1 - min_T1) / n_step
            step_T2 = (max_T2 - min_T2) / n_step
            rranges = (slice(min_T1, max_T1 + step_T1, step_T1), slice(min_T2, max_T2 + step_T2, step_T2))
            x0, loss_x0, grid, loss_grid = _opt.brute(_loss_function, rranges, args=(signal, delta, min_T1, max_T2),
                                                      finish=None, full_output=True)
        else:
            _PhUI.e("opt_method not understood")
            return None

        if complete:
            x0_min, l_x0_min, niter, nfuncalls, warnflag = _opt.fmin(_loss_function, x0, args=(signal, delta),
                                                                     full_output=True,
                                                                     **min_pars)  # TODO (Ale): funziona?
            return x0_min, x0
        else:
            return x0

    @classmethod
    def _loss_function(par_bat, signal, delta, min_T1, max_T2):
        """
        Computes the loss for optimization of Bateman parameters.

        Parameters
        ----------
        par_bat : list
            Bateman parameters to be optimized
        signal : nparray
            The EDA signal
        delta : float
            Minimum amplitude of the peaks in the driver
        min_T1 : float
            Lower bound for T1
        max_T2 : float
            Upper bound for T2
        
        Returns
        -------
        loss : float
            The computed loss
        """

        # check if pars hit boudaries
        if par_bat[0] < min_T1 or par_bat[1] > max_T2 or par_bat[0] >= par_bat[1]:
            return _np.Inf  # 10000 TODO: check if it raises errors

        fsamp = signal.sampling_freq
        driver = _DriverEstim(par_bat)(signal)
        maxs, mins = PeakDetection(delta=delta, refractory=1, start_max=True)(driver)

        if len(maxs) != 0:
            idx_maxs = maxs[:, 0]
        else:
            _PhUI.w('Unable to find peaks in driver signal for computation of Energy. Returning Inf')
            return _np.Inf  # or 10000 #TODO: check if np.Inf does not raise errors

        # STAGE 1: select maxs distant from the others
        diff_maxs = _np.diff(_np.r_[idx_maxs, len(driver) - 1])
        th_diff = 15 * fsamp

        # TODO (new feature): select th such as to have enough maxs, e.g. diff_maxs_tentative = np.median(diff_maxs)
        idx_selected_maxs = _np.where(diff_maxs > th_diff)[0]
        selected_maxs = idx_maxs[idx_selected_maxs]

        energy = _np.Inf
        if len(selected_maxs) != 0:
            energy = 0
            for idx_max in selected_maxs:
                driver_portion = driver[idx_max:idx_max + 15 * fsamp]

                half = len(driver_portion) - 5 * fsamp

                y = driver_portion[half:]
                diff_y = _np.diff(y)
                th_75 = _np.percentile(diff_y, 75)
                th_25 = _np.percentile(diff_y, 25)

                idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
                diff_y_sel = diff_y[idx_sel_diff_y]

                # A2, B = _find_slope(y, half)
                mean_s = BootstrapEstimation(func=_np.mean, N=100, k=0.5)(diff_y_sel)

                mean_y = BootstrapEstimation(func=_np.median, N=100, k=0.5)(y)

                b_mean_s = mean_y - mean_s * (half + (len(driver_portion) - half) / 2)

                line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s

                driver_detrended = driver_portion - line_mean_s

                driver_detrended /= _np.max(driver_detrended)
                energy_curr = (1 / fsamp) * _np.sum(driver_detrended[fsamp:] ** 2) / (len(driver_detrended) - fsamp)

                energy += energy_curr
        else:
            _PhUI.w('Peaks found but too near. Returning Inf')
            return _np.Inf  # or 10000 #TODO: check if np.Inf does not raise errors

        _PhUI.i('Current parameters: ' + str(par_bat[0]) + ' - ' + str(par_bat[1]) + ' Loss: ' + str(energy))
        return energy

    _params_descriptors = {
        'opt_method': _Par(1, str, 'Method to perform the search of optimal parameters.', 'asa',
                           lambda x: x in ['asa', 'grid']),
        'complete': _Par(1, bool, 'Whether to perform a minimization after detecting the optimal parameters', True),
        'par_range': _Par(1, list, '[min_T1, max_T1, min_T2, max_T2] boundaries for the Bateman parameters',
                          lambda x: len(x) == 4),
        'maxiter': _Par(1, int, 'Maximum number of iterations ("asa" method).', 0, lambda x: x > 0,
                        lambda x, p: p['opt_method'] == 'asa'),
        'n_step': _Par(1, int, 'Number of increments in the grid search', 10, lambda x: x > 0,
                       lambda x, p: p['opt_method'] == 'grid'),
        'delta': _Par(2, (int, float), 'Minimum amplitude of the peaks in the driver', 0, lambda x: x > 0),
        'min_pars': _Par(0, dict, 'Additional parameters to pass to the minimization function (when complete = True)',
                         activation=lambda x, p: p['complete'])

    }