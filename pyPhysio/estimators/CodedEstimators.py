# coding=utf-8
import numpy as _np
from scipy.signal import gaussian as _gaussian, deconvolve as _deconvolve
from ..BaseEstimator import Estimator as _Estimator
from ..Signal import EvenlySignal, UnevenlySignal
from ..filters.Filters import IIRFilter, ConvolutionalFilter, DeConvolutionalFilter, Diff

__author__ = 'AleB'


### IBI ESTIMATION ###
class BeatFromBP(_Estimator):
    """Identify the beats in a Blood Pulse (BP) signal and compute the IBIs.
    Optimized to identify the percussion peak.
    
    Limitations: works only on 'BP' type signal.
    
    Based on two stages:
    1) Identification of candidate beats
    2) Identification of the peak for each beat, using the derivative of the signal

    Parameters
    ----------
    bpm_max : int (>0)
        Maximal expected heart rate (in beats per minute)
    method : 'dt' or 'lp'
        Method to identify the beats in the first stage
    sigma : float (>0)
        Amplitude of the gaussian for the 'dt' method

    Returns
    -------
    ibi : pyphysio.UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
        See paper ...
    """
    @classmethod
    def get_signal_type(cls):
        return ['BVP']
    
    @classmethod
    def algorithm(cls, signal, params):  # FIX others
        bpm_max = params["bpm_max"]
        method = params["method"]
        sigma = params["sigma"]

        # checn nature signal is 'BP'
        fmax = bpm_max / 60

        fsamp = signal.sampling_freq

        inactive = int(round(fsamp / fmax))

        # STAGE 1 - EXTRACT BEAT POSITION SIGNAL
        if method == 'dt':  # (derivative gaussian)
            sigma = my_assert(sigma, params)
            len_model = sigma * 8
            M = round(len_model * fsamp)
            sigma = round(sigma * fsamp)

            gauss_dt = ConvolutionalFilter(sigma * 8, sigma)
            signal_f = gauss_dt(signal)

        elif method == 'lp':  # (lowpass)
            signal_f = IIRFilter(1.2 * fmax, 5 * fmax)(signal)  # TODO: formato della chiamata (param=val, param2=val)

        # TODO: once implemented estimate_delta, pass fmax and fmax/3 (fsamp is in delta_f)
        deltas = tll.estimate_delta(signal_f, fsamp / fmax,
                                    3 * fsamp / fmax) * 0.5  ### Tools  # TODO: tll non è definito

        # detection of peaks
        maxp, minp = tll.peakdet(signal_f, deltas, refractory=inactive)  ### Tools
        idx_d = maxp[:, 0]

        if idx_d[0] == 0:
            idx_d = idx_d[1:]

        # STAGE 3 - IDENTIFY PEAKS using the signal derivative
        dxdt = Diff()(signal)
        true_peaks = []

        WIN = 0.25 * fsamp
        for idx_beat in idx_d:
            start_ = idx_beat - WIN
            if start_ < 0:
                start_ = 0

            # select portion of derivative where to search
            obs = dxdt[start_:idx_beat]
            peak_obs = _np.argmax(obs)
            true_obs = dxdt[start_ + peak_obs: idx_beat]

            # find the 'first minimum' (zero) the derivative (peak)
            # TODO: once implemented localMinima, pass only 0.05 (fsamp in true_obs) - CHECK
            mins = tll.localMinima(abs(true_obs), 0.05 * fsamp)  ### Tools
            idx_mins = mins[:, 0]
            if len(idx_mins) >= 2:
                peak = idx_mins[1]
                true_peaks.append(start_ + peak_obs + peak)
            else:
                # TODO : decide how to manage warnings
                # action_message('Peak not found; idx_beat: '+str(idx_beat), RuntimeWarning)
                pass

        # STAGE 4 - FINALIZE computing IBI and fixing indexes
        # TODO: decide whether to remove first beat
        ibi_values = Diff()(true_peaks) / fsamp  # TODO: Aggiungere il segnale alla chiamata Diff
        ibi_values = _np.r_[ibi_values[0], ibi_values]
        idx_ibi = _np.array(true_peaks)
        ibi = UnevenlySignal(ibi_values, idx_ibi, 'IBI', signal.start_time, meta=signal.meta)
        return (ibi)

    @classmethod
    def check_params(cls, params):
        if 'bpm_max' not in params | params['bpm_max'] < 0:
            # action_param(params, BPM_MAX=0)
            pass
        if 'method' not in params | params['sigma'] not in ['lp', 'dt']:
            # action_param(params, METHOD ='lp')
            pass
        if 'sigma' not in params | params['sigma'] < 0:
            # action_param(params, SIGMA=0.05)            
            pass
        return params

    @staticmethod
    def _generate_gaussian_derivative(M, S):
        g = _gaussian(M, S)
        gaussian_derivative_model = _np.diff(g)

        return gaussian_derivative_model


class BeatFromECG(_Estimator):
    """
    Identify the beats in an ECG signal and compute the IBIs.
    
    Parameters
    ----------
    bpm_max : int (>0)
        Maximal expected heart rate (in beats per minute)
        If not given the peak detection will run without refractory period.
    delta : float or nd.array (>0 len(delta)=len(signal)
        Threshold for the peak detection.
        If not given it will be computed.
    
    Returns
    -------
    idx_ibi : nparray
        Indexes of the input signal corresponding to R peaks
    ibi : nparray
        Inter beat interval values at R peaks
    
    Notes
    -----
    See paper ...
    """


    @classmethod
    def algorithm(cls, signal, params):
        bpm_max, delta, delta_pars_dict = params["bpm_max"], params["delta"], params["delta_pars_dict"]

        # checn nature signal is 'ECG'

        fmax = bpm_max / 60

        fsamp = signal.fsamp

        inactive = int(round(fsamp / fmax))

        if delta is None:
            delta = tll.estimate_delta(signal, **delta_pars_dict)  ## Tools

        maxp, minp = tll.peakdet(signal, delta, refractory=inactive)

        # detection of peaks
        maxp, minp = tll.peakdet(signal_f, deltas, refractory=inactive)  ### Tools
        idx_d = maxp[:, 0]

        if idx_d[0] == 0:
            idx_d = idx_d[1:]

        # TODO: decide whether to remove first beat
        ibi_values = Diff()(true_peaks) / fsamp
        ibi_values = _np.r_[ibi_values[0], ibi_values]
        ibi = UnevenlySignal(ibi_values, idx_ibi, 'IBI', fsamp, signal.start_time, meta=signal.meta)
        return ibi

    @classmethod
    def check_params(cls, params):
        if 'bpm_max' not in params | params['bpm_max'] < 0:
            # action_param(params, BPM_MAX=0)
            pass
        if 'delta' not in params:
            # default = None # OK
            # requires additional parameters:
            # idx_len = fsamp*2
            # idx_stp = fsamp/2
            pass
        return params


class BeatOutliers(_Estimator):
    """
    Detects outliers in IBI signal.
    Returns id of outliers.
    NOTE: It only detects outliers. You should manually remove outliers using FixIBI

    Parameters
    ----------
    cache : int, optional
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
    See paper ...
        
    Raises
    ------
    RuntimeWarning: whenever the cache memory is reinitialized    
    """

    @classmethod
    def algorithm(cls, signal, params):
        remove, cache, sensitivity, ibi_median = params["remove"], params["cache"], params["sensitivity"], params[
            "ibi_median"]
        # check signal is IBI
        if ibi_median == 0:
            ibi_expected = _np.median(signal)
        else:
            ibi_expected = ibi_median
        id_bad_ibi = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        for i in range(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi[i]
            curr_ibi = ibi[i]

            if (curr_ibi > curr_median * (1 + sensitivity)):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            # missings.append([idx_ibi[i-1],idx_ibi[i]])

            elif (curr_ibi < curr_median * (1 - sensitivity)):  # abnormal peak:
                id_bad_ibi.append(i)  # append ibi id to the list of bad ibi
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[-cache + 1:], curr_ibi]
                counter_bad = 0  # TODO: check

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                #action_message('Cache re-initialized - '+ str(curr_idx), RuntimeWarning) # OK
                counter_bad = 0

        # return(id_bad_ibi, missings)
        if remove:
            #fixIBI
            return signal
        else:
            return id_bad_ibi


    @classmethod
    def check_params(cls, params):
        if 'ibi_median' not in params:
            # default ibi_median == 0 # OK
            pass
        if 'cache' not in params:
            # default cache=5 # OK
            pass
        if 'sensitivity' not in params:
            # defualt sensitivity = 0.3 # OK
            pass
        if 'remove' not in params:
            # default = True # OK
            pass
        return params


class _FixIBI(_Estimator):
    """
    Corrects the IBI series removing abnormal IBI
    
    Parameters
    ----------
    idx_ibi : nparray
        Indexes of beat instants in original signal
    ibi : nparray
        Inter beat interval (IBI) values corresponding to beat instants
    id_bad_ibi : nparray
        Identifiers of abnormal beats
   
    Returns
    -------
    idx_ibi : nparray
        Corrected indexes of beat instants in original signal
    ibi : nparray
        Corrected IBI values corresponding to beat instants
            
    Notes
    -----
    See paper
    """

    # TODO: correct


    idx_ibi_nobad = np.delete(idx_ibi, id_bad)
    ibi_nobad = np.delete(ibi, id_bad)
    idx_ibi = idx_ibi_nobad.astype(int)
    ibi = ibi_nobad
    return (idx_ibi, ibi)


class BeatOptimizer(_Estimator):
    """
    Optimize detection of errors in IBI estimation.
    
    Parameters
    ----------
    B : float
        Ball radius (in seconds)
    cache : int, optional
        Nuber of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, optional
        Relative variation from the current median that is accepted
    ibi_median : float, optional
        If given it is used to initialize the cache
        
    Returns
    -------
    idx_ibi, ibi : 
        Optimized IBI signal

    Notes
    -----
    See paper ...
        
    Raises
    ------
    RuntimeWarning: whenever the cache memory is reinitialized    
    """

    @classmethod
    def algorithm(cls, signal, params):
        B, cache, sensitivity, ibi_median = params["B"], params["cache"], params["sensitivity"], params["ibi_median"]
        # check signal is IBI
        if ibi_median == 0:
            ibi_expected = _np.median(Diff()(idx_ibi))  # TODO: idx_ibi non definito
        else:
            ibi_expected = ibi_median

        idx_ibi = signal.indexes
        fsamp = signal.fsamp
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

            # print([curr_median*(1+sensitivity), curr_median*(1-sensitivity), curr_median])
            if (curr_ibi > curr_median * (1 + sensitivity)):  # abnormal peak:
                prev_idx = curr_idx
                ibi_1.append(_np.nan)
                idx_1.append(curr_idx)
                counter_bad += 1

            elif (curr_ibi < curr_median * (1 - sensitivity)):  # abnormal peak:
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

        ########################################
        # RUN BACKWARD:
        idx_ibi_rev = -1 * (idx_ibi - idx_ibi[-1])
        idx_ibi_rev = idx_ibi_rev[::-1]

        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_2 = [idx_ibi_rev[0]]
        ibi_2 = []

        prev_idx = idx_ibi_rev[0]
        for i in range(1, len(idx_ibi_rev)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi_rev[i]
            curr_ibi = curr_idx - prev_idx

            # print([curr_median*(1+sensitivity), curr_median*(1-sensitivity), curr_median])
            if (curr_ibi > curr_median * (1 + sensitivity)):  # abnormal peak:
                prev_idx = curr_idx
                ibi_2.append(_np.nan)
                idx_2.append(curr_idx)
                counter_bad += 1

            elif (curr_ibi < curr_median * (1 - sensitivity)):  # abnormal peak:
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

        ########################################
        # add indexes of idx_ibi_2 which are not in idx_ibi_1 but close enough
        B = B * fsamp
        for i_2 in range(1, len(idx_2)):
            curr_idx_2 = idx_2[i_2]
            if not (curr_idx_2 in idx_1):
                i_1 = _np.where((idx_1 >= curr_idx_2 - B) & (idx_1 <= curr_idx_2 + B))[0]
                if not len(i_1) > 0:
                    idx_1 = _np.r_[idx_1, curr_idx_2]
        idx_1 = _np.sort(idx_1)

        ########################################
        # create pairs for each beat
        pairs = []
        for i_1 in range(1, len(idx_1)):
            curr_idx_1 = idx_1[i_1]
            if (curr_idx_1 in idx_2):
                pairs.append([curr_idx_1, curr_idx_1])
            else:
                i_2 = _np.where((idx_2 >= curr_idx_1 - B) & (idx_2 <= curr_idx_1 + B))[0]
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

            combinations = list(itertools.product([0, 1], repeat=i_sp - i_st - 1))  # TODO: intertools non definito
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


        ########################################
        # finalize arrays
        idx_out = _np.array(idx_out) + idx_st
        ibi_out = Diff()(idx_out)
        # TODO: decide whether to remove first beat
        ibi_out = _np.r_[ibi_out[0], ibi_out]
        ibi = UnevenlySignal(ibi_out, idx_out, 'IBI', signal.start_time, meta=signal.meta)
        return (ibi)


    @classmethod
    def check_params(cls, params):
        if not 'B' in params:
            # default B = 0.25 # OK
            pass
        if not 'ibi_median' in params:
            # default ibi_median == 0 # OK
            pass
        if not 'cache' in params:
            # default cache=5 # OK
            pass
        if not 'sensitivity' in params:
            # defualt sensitivity = 0.3 # OK
            pass
        return (params)


# TODO: fixIBI in Tools

### PHASIC ESTIMATION ###

class DriverEstim(_Estimator):
    """
    Estimates the driver of an EDA signal according to Benedek 2010

    The estimation uses a deconvolution using a bateman function as Impulsive Response Function.
    Based on the bateman function:

    if len(par_bat) = 2: :math:`b = e^{-t/T1} - e^{-t/T2}`

    if len(par_bat) = 4: :math:`b = e^{-t/T1} - (1-A)e^{-t/T2} - Ae^{-t/T3}`

    Parameters
    ----------
    par_bat: list (T1, T2)
        Parameters of the bateman function
    method : 'fft' or 'sps'
        Method to compute the deconvolution, Default 'fft'

    Returns
    -------
    driver : EvenlySignal
        The driver function

    Notes
    -----
    See Benedek2010
    See Paper ...
    """

    @classmethod
    def algorithm(cls, signal, params):
        par_bat = params['par_bat']
        method = params['method']
        fsamp = signal.fsamp

        bateman = DriverEstim._gen_bateman(fsamp, par_bat)
        idx_max_bat = _np.argmax(bateman)
        bateman_first_half = bateman[0:idx_max_bat + 1]
        bateman_first_half = signal[0] * (bateman_first_half - _np.min(bateman_first_half)) / (
            _np.max(bateman_first_half) - _np.min(bateman_first_half))

        bateman_second_half = bateman[idx_max_bat:]
        bateman_second_half = signal[-1] * (bateman_second_half - _np.min(bateman_second_half)) / (
            _np.max(bateman_second_half) - _np.min(bateman_second_half))
        signal_in = _np.r_[bateman_first_half, signal, bateman_second_half]
        if method == 'fft':
            dec_filt = DeConvolutionalFilter(irf=bateman)(signal_in)
        elif method == 'sps':
            driver, _ = _deconvolve(signal_in, bateman[1:])

        driver = driver[idx_max_bat + 1: idx_max_bat + len(signal)]
        # gaussian smoothing (s=200 ms)
        degree = int(_np.ceil(0.2 * fsamp))

        # Gaussian smoothing
        driver = ConvolutionalFilter(irftype='gauss', N=degree * 8 + 1)

        driver = EvenlySignal(driver, signal_nature="dEDA", start_time=signal.start_time, meta=signal.meta)
        return driver

    @classmethod
    def check_params(cls, params):
        if 'par_bat' not in params or len(params['par_bat']) != 2:
            # default par_bat = (0.75, 2) # GRAVE
            pass
        if 'method' not in params:
            # default method = 'fft' # OK
            pass
        return params

    @staticmethod
    def _gen_bateman(fsamp, par_bat):
        """
        Generates the bateman function:

        if len(par_bat) = 2: :math:`b = e^{-t/T1} - e^{-t/T2}`

        if len(par_bat) = 4: :math:`b = e^{-t/T1} - (1-A)e^{-t/T2} - Ae^{-t/T3}`

        Parameters
        ----------
        fsamp : float
            The sampling frequency
        par_bat: list (T1, T2)
            Parameters of the bateman function

        Returns
        -------
        bateman : nparray
            The bateman function
        """

        idx_T1 = par_bat[0] * fsamp
        idx_T2 = par_bat[1] * fsamp
        len_bat = idx_T2 * 10
        idx_bat = _np.arange(len_bat)
        bateman = -1 * (_np.exp(-idx_bat / idx_T1) - _np.exp(-idx_bat / idx_T2))

        # normalize
        bateman = bateman / ((1 / fsamp) * _np.sum(bateman))
        return bateman


class PhasicEstim(_Estimator):
    """
    Estimates the phasic and tonic components of a driver.

    It uses a detection algorithm based on the derivative of the driver.

    Parameters
    ----------
    delta : float
        Minimum amplitude of the peaks in the driver
    grid_size : float
        Sampling size of the interpolation grid
    i_pre_max : int
        Samples before the peak that are considered to find the start of the peak
    i_post_max : int
        Samples after the peak that are considered to find the end of the peak

    Returns
    -------
    tonic : nparray
        The tonic component
    phasic : nparray
        The phasic component
    driver_no_peak : nparray
        The "de-peaked" driver signal used to generate the interpolation grid
    """

    @classmethod
    def algorithm(cls, signal, params):
        delta, grid_size, i_pre_max, i_post_max = params["delta"], params["grid_size"], params["i_pre_max"], params[
            "i_post_max"]

        ZERO = 0

        fsamp = signal.fsamp
        if i_pre_max is None:
            i_pre_max = _np.round(2 * fsamp)

        if i_post_max is None:
            i_post_max = _np.round(2 * fsamp)

        max_driv, tmp_ = tll.peakdet(driver, delta)  # TODO: Tools, driver non definito

        if _np.shape(max_driv)[0] == 0:
            # action_message('Unable to find peaks in the driver signal.', RuntimeWarning)  # GRAVE
            return _np.zeros_like(driver), driver, _np.zeros_like(driver)  # TODO fix

        driver_dt = Diff()(driver)

        idx_pre = []
        idx_post = []

        for i in idx_peaks:
            # find START
            i_st = i - i_pre_max
            if i_st < 0:
                i_st = 0

            driver_dt_pre = driver_dt[i_st: i]
            i_pre = len(driver_dt_pre) - 1
            Done = False
            while not Done:
                if driver_dt_pre[i_pre] <= ZERO:
                    Done = True
                else:
                    i_pre = i_pre - 1
                    if i_pre < 0:
                        Done = True
                        i_pre = 0
            i_pre_true = i_st + i_pre
            idx_pre.append(i_pre_true)

            # find STOP
            i_sp = i + i_post_max
            if i_sp >= len(driver_dt):
                i_sp = len(driver_dt)

            driver_dt_post = driver_dt[i: i_sp]
            i_post = 0
            Done = False
            while not Done:
                if driver_dt_post[i_post] >= ZERO:
                    Done = True
                else:
                    i_post = i_post + 1
                    if i_post >= len(driver_dt_post):
                        Done = True
                        i_post = len(driver_dt_post)
            i_post_true = i + i_post
            idx_post.append(i_post_true)

        driver_no_peak = _np.copy(driver)  # TODO: check

        for I in range(len(idx_pre)):
            i_st = idx_pre[I]
            i_sp = idx_post[I]

            idx_base = _np.arange(i_sp - i_st)

            coeff = (driver[i_sp] - driver[i_st]) / len(idx_base)

            driver_base = idx_base * coeff + driver[i_st]

            driver_no_peak[i_st:i_sp] = driver_base

        idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * fsamp)
        idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]
        driver_grid = driver_no_peak[idx_grid]

        # spline interpolation to obtain the tonic component
        tonic = flt.interpolate_unevenly(driver_grid, idx_grid, 1)  # TODO: fix...
        tonic = _np.r_[tonic, driver[-1]]
        phasic = driver - tonic

        # TODO: fix ...
        return phasic, tonic, driver_no_peak


    @classmethod
    def check_params(cls, params):
        if not 'delta' in params:
            # default delta = 0.01 # GRAVE
            pass
        if not 'grid_size' in params:
            # default grid_size = 1 # OK
            pass
        if not 'idx_pre_max' in params:
            # defults = None # OK
            pass
        if not 'idx_post_pax' in params:
            # defaul = None # Ok
            pass
        return params

# TODO: optimize in Tools

