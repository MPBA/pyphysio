import numpy as _np
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate        
EXT_LIB_DIR = '/home/andrea/Trento/CODICE/libraries'

#TODO: ADD SEED

def compute_AAFT_surrogates(data, angle):
    # create Gaussian data
    gaussian = _np.random.randn(data.shape[0])
    gaussian.sort(axis = 0)

    # rescale data
    ranks = data.argsort(axis = 0).argsort(axis = 0)
    rescaled_data = gaussian[ranks]

    # transform the time series to Fourier domain
    xf = _np.fft.rfft(rescaled_data, axis = 0)
     
    # randomise the time series with random phases     
    cxf = xf * _np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = _np.fft.irfft(cxf, n = data.shape[0], axis = 0)

    # rescale back to amplitude distribution of original data
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)
    ranks = ft_surr.argsort(axis = 0).argsort(axis = 0)

    rescaled_data = sorted_original[ranks]
    
    return(rescaled_data)

def compute_IAAFT_surrogates(data, n_iters=10):
    a = _np.fft.rfft(_np.random.rand(data.shape[0]), axis = 0)
    angle = _np.random.uniform(0, 2 * _np.pi, (a.shape[0],))
    angle[0] = 0
    xf = _np.fft.rfft(data, axis = 0)
    xf_amps = _np.abs(xf)
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)

    # starting point
    R = compute_AAFT_surrogates(data, angle)

    # iterate
    for _ in range(n_iters):
        r_fft = _np.fft.rfft(R, axis = 0)
        r_phases = r_fft / _np.abs(r_fft)

        s = _np.fft.irfft(xf_amps * r_phases, n = data.shape[0], axis = 0)

        ranks = s.argsort(axis = 0).argsort(axis = 0)
        R = sorted_original[ranks]

    return(R)

class DTWDistance(object):
    def __init__(self, method='Euclidean',step='asymmetric', wtype='sakoechiba',openend=True, openbegin=True, wsize=5):
        import rpy2.robjects.numpy2ri
        from rpy2.robjects.packages import importr
        
        # Set up our R namespaces
        self.R = rpy2.robjects.r
        self.DTW = importr('dtw')
        self.method = method
        self.step = step
        self.dtwstep = getattr(self.DTW, self.step)
        self.wtype = wtype
        self.openend = openend
        self.openbegin = openbegin
        self.wsize = wsize
    
    def compute(self, data1, data2, N=0):        
        # Calculate the alignment vector and corresponding distance
        x = list(data1)
        y = list(data2)
    
        alignment = self.R.dtw(x, y, dist_method=self.method, step_pattern=self.dtwstep, window_type=self.wtype, keep_internals=False, distance_only=True, open_end=self.openend, open_begin=self.openbegin, **{'window.size': self.wsize})
        dist = alignment.rx('distance')[0][0]
        
        if N>0:
            dtw_surrogate = DTWDistance(self.method, self.step, self.wtype, self.openend, self.openbegin, self.wsize)
            
            dist_permuted = []
            y_orig = _np.array(y)
            for i in range(N):
                y = list(compute_IAAFT_surrogates(y_orig))
                dist_surrogate = dtw_surrogate.compute(x, y, N=0)
                dist_permuted.append(dist_surrogate)
            
            dist_permuted = _np.array(dist_permuted)
            p_value = _np.sum(dist_permuted<dist)/len(dist_permuted)
            return(dist, p_value)
        
        return(dist)

class CrossCorrDistance(object):
    def __init__(self, lag = 10, normalize = True):
        self.lag = 10
        self.normalize = normalize
    
    def compute(self, x, y, N=0):
        
        c = correlate(x, y)
        
        idx_0 = int(_np.ceil(len(c)/2))
        c_lag = c[idx_0-self.lag: idx_0+self.lag+1]
        
        dist = _np.max(c_lag)
        if self.normalize:
            dist = dist/len(x)
        
        if N>0:
            c_surrogate = CrossCorrDistance(self.lag, self.normalize)
            
            dist_permuted = []
            y_orig = _np.array(y).copy()
            for i in range(N):
                y = compute_IAAFT_surrogates(y_orig)
                dist_surrogate = c_surrogate.compute(x, y, N=0)
                dist_permuted.append(dist_surrogate)
            
            dist_permuted = _np.array(dist_permuted)
            p_value = _np.sum(dist_permuted<dist)/len(dist_permuted)
            return(dist, p_value)
        
        return(dist)

class MIDistance(object):
    def __init__(self, K=4):
        import jpype
        jarLocation = f'{EXT_LIB_DIR}/infodynamics-dist-1.4/infodynamics.jar'
        
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
        MIcalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
        self.MIcalc = MIcalcClass()
        # 2. Set any properties to non-default values:
        self.MIcalc.setProperty("NORMALISE", "true")
        self.MIcalc.setProperty("k", str(K))
        self.K = K
        
        
    def compute(self, data1, data2, N=0):
        self.MIcalc.initialise()
        self.MIcalc.setObservations(data1, data2)
        MI = self.MIcalc.computeAverageLocalOfObservations()
    
        if N>0:
            mi_surrogate = MIDistance(self.K)
            
            MI_permuted = []
            y_orig = _np.array(data2)
            for i in range(N):
                y = compute_IAAFT_surrogates(y_orig)
                dist_surrogate = mi_surrogate.compute(data1, y, N=0)
                MI_permuted.append(dist_surrogate)
            
            MI_permuted = _np.array(MI_permuted)
            p_value = _np.sum(MI_permuted<MI)/len(MI_permuted)
            return(MI, p_value)
        return(MI)

class PearsonDistance(object):
    def compute(self, data1, data2, N=0):        
        # Calculate the alignment vector and corresponding distance
        x = list(data1)
        y = list(data2)
    
        r, p = pearsonr(x, y)
        
        if N>0:
            pearson_surrogate = PearsonDistance()
            
            r_permuted = []
            y_orig = _np.array(y)
            for i in range(N):
                y = list(compute_IAAFT_surrogates(y_orig))
                r_surrogate, _ = pearson_surrogate.compute(x, y, N=0)
                r_permuted.append(r_surrogate)
            
            r_permuted = _np.array(r_permuted)
            p_value = _np.sum(r_permuted<r)/len(r_permuted)
            return(r, p, p_value)
        
        return(r, p)

class SpearmanDistance(object):
    def compute(self, data1, data2, N=0):        
        # Calculate the alignment vector and corresponding distance
        x = list(data1)
        y = list(data2)
    
        r, p = spearmanr(x, y)
        
        if N>0:
            pearson_surrogate = PearsonDistance()
            
            r_permuted = []
            y_orig = _np.array(y)
            for i in range(N):
                y = list(compute_IAAFT_surrogates(y_orig))
                r_surrogate, _ = pearson_surrogate.compute(x, y, N=0)
                r_permuted.append(r_surrogate)
            
            r_permuted = _np.array(r_permuted)
            p_value = _np.sum(r_permuted<r)/len(r_permuted)
            return(r, p, p_value)
        
        return(r, p)        

def compute_TDS(x, y, winlen=None, winstep=None, nsamp=0):
    if winlen is None:
        winlen = len(x)
        winstep = len(x)

    idxs_cross_start = _np.arange(0, len(y) - winstep + 1, winstep)
    idx_t0 = int(_np.round(winlen - 1))

    delays = []
    for idx_start in idxs_cross_start:
        idx_end = idx_start + winlen

        # 1. Obtain segments
        x_v = x[idx_start:idx_end]
        y_v = y[idx_start:idx_end]

        # 2. Normalize
        x_v_norm = ph.Normalize(norm_method='standard')(x_v)
        y_v_norm = ph.Normalize(norm_method='standard')(y_v)

        # 3. Compute cross-correlation
        cross = _np.correlate(x_v_norm, y_v_norm, mode='full')

        # 4. Find delay where abs(cross) is max
        idx_max = _np.argmax(abs(cross))
        idx_max -= idx_t0
        delays.append(idx_max)

    delays = _np.array(delays)

    # 5. Compute stability 
    stability = _np.repeat(_np.nan, len(delays))
    # select interval
    if nsamp == 0:
        nsamp = len(delays)

    for idx_start in _np.arange(len(delays) - nsamp + 1):
        idx_end = idx_start + nsamp
        curr_delays = delays[idx_start:idx_end]
        curr_delays = curr_delays - _np.mean(curr_delays)
        energy = _np.sqrt(_np.sum(curr_delays ** 2))
        stability[idx_start] = energy
    return stability, delays

'''
def code_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    
from scipy.stats import pearsonr

   
# ===========================================================================
# Tools for computation of coupling metrics
# TODO: Accept M-signals in input - matrices NxM

EXTLIB_DIR = '/home/andrea/Trento/CODICE/workspaces/pyphysio/ext_lib/'
RP_dir = EXTLIB_DIR + 'commandline_recurrence_plot/'
ID_dir = EXTLIB_DIR = 'infodynamics-dist-1.3/'
jarLocation = '/home/andrea/Trento/CODICE/libraries/infodynamics-dist-1.3/infodynamics.jar'


def compute_PC(x, y):
    r, p = pearsonr(x, y)
    return r, p


def compute_TE(x, y, history=1, kernelwidth=0.5):
    # Create a TE calculator and run it:
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
    teCalc.initialise(history,
                      kernelwidth)  # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(x), jpype.JArray(jpype.JDouble, 1)(y))
    result = teCalc.computeAverageLocalOfObservations()
    return result


def compute_MI(x, y, K=4, N=1000):
    shuff_MI = _np.array([_np.random.permutation(len(x)) for i in range(N)])

    # Create a MI calculator and run it:
    MIcalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    MIcalc = MIcalcClass()
    # 2. Set any properties to non-default values:
    MIcalc.setProperty("NORMALISE", "true")
    MIcalc.setProperty("k", str(K))
    
    # 3. Initialise the calculator for (re-)use:
    MIcalc.initialise()
    
    # 4. Supply the sample data:
    MIcalc.setObservations(x, y)
    
    # 5. Compute the estimate:
    MI = MIcalc.computeAverageLocalOfObservations()
    
    # 6. Compute significance:
#    jpype.JArray(jpype.JInt, 1)(shuff_MI)
    MI_p = MIcalc.computeSignificance(jpype.JArray(jpype.JInt, 2)(shuff_MI.tolist())).pValue
    
    return(MI, MI_p)


def compute_CRQA(x, y, embed_m, embed_t, th):
    cwd = os.getcwd()
    os.chdir(RP_dir)
    # TODO: Check whether directories already exist
    random_dir = code_generator(6)
    os.mkdir(random_dir)
    os.chdir(random_dir)
    _np.savetxt('x', x)
    _np.savetxt('y', y)
    os.chdir(RP_dir)
    os.system('./rp -i ' + random_dir + '/x -j ' + random_dir + '/y -m ' + str(embed_m) + ' -t ' + str(
        embed_t) + ' -e ' + str(th) + ' -o ' + random_dir + '/outmeas')
    try:
        meas = _np.loadtxt(random_dir + '/outmeas')
        os.chdir(random_dir)
        os.remove('x')
        os.remove('y')
        os.remove('outmeas')
        os.chdir(RP_dir)
        os.rmdir(random_dir)
        os.chdir(cwd)
        # labels = ['#RR', 'DET', 'DET/RR', 'LAM', 'LAM/DET', 'L_max', 'L', 'L_entr', 'DIV', 'V_max', 'TT', 'V_entr', 'T1', 'T2', 'W_max', 'W_mean', 'W_entr', 'W_prob', 'F_min']
        return meas
    except Exception as e:
        print('Unable to compute RP metrics: %s' % e.message)
        os.chdir(random_dir)
        os.remove('x')
        os.remove('y')
        os.chdir(RP_dir)
        os.rmdir(random_dir)
        os.chdir(cwd)
        return _np.repeat(_np.nan, 19)


def compute_TDS(x, y, winlen=None, winstep=None, nsamp=0):
    if winlen is None:
        winlen = len(x)
        winstep = len(x)

    idxs_cross_start = _np.arange(0, len(y) - winstep + 1, winstep)
    idx_t0 = int(_np.round(winlen - 1))

    delays = []
    for idx_start in idxs_cross_start:
        idx_end = idx_start + winlen

        # 1. Obtain segments
        x_v = x[idx_start:idx_end]
        y_v = y[idx_start:idx_end]

        # 2. Normalize
        x_v_norm = ph.Normalize(norm_method='standard')(x_v)
        y_v_norm = ph.Normalize(norm_method='standard')(y_v)

        # 3. Compute cross-correlation
        cross = _np.correlate(x_v_norm, y_v_norm, mode='full')

        # 4. Find delay where abs(cross) is max
        idx_max = _np.argmax(abs(cross))
        idx_max -= idx_t0
        delays.append(idx_max)

    delays = _np.array(delays)

    # 5. Compute stability 
    stability = _np.repeat(_np.nan, len(delays))
    # select interval
    if nsamp == 0:
        nsamp = len(delays)

    for idx_start in _np.arange(len(delays) - nsamp + 1):
        idx_end = idx_start + nsamp
        curr_delays = delays[idx_start:idx_end]
        curr_delays = curr_delays - _np.mean(curr_delays)
        energy = _np.sqrt(_np.sum(curr_delays ** 2))
        stability[idx_start] = energy
    return stability, delays


def compute_MIC(x, y, alpha=0.6, c=15, all_metrics=False):
    from minepy import MINE
    mine = MINE(alpha, c)
    mine.compute_score(x, y)
    if all_metrics:
        return mine.mic(), mine
    else:
        return mine.mic()


def compute_DTW(x, y, N=1000, method="Euclidean", step='asymmetric', wtype="none", keep=False, distanceonly=True, openend=True, openbegin=True, wsize=5):  # TODO: complete
    # TODO: check that rpy2 is installed
    # TODO: check that dtw is installed
    step = getattr(DTW, step, 'asymmetric')
    
    # Calculate the alignment vector and corresponding distance
    x = list(x)
    y = list(y)

    alignment = R.dtw(x, y, dist_method=method, step_pattern=step, window_type=wtype, keep_internals=keep, distance_only=distanceonly, open_end=openend, open_begin=openbegin, **{'window.size': wsize})
    dist = alignment.rx('distance')[0][0]
    
    dist_permuted = []
    y_orig = _np.array(y)

    for i in range(N):
        y = list(compute_IAAFT_surrogates(y_orig))
        alignment = R.dtw(x, y, dist_method=method, step_pattern=step, window_type=wtype, keep_internals=keep,
                          distance_only=distanceonly, open_end=openend, open_begin=openbegin, **{'window.size': wsize})
        dist_permuted.append(alignment.rx('distance')[0][0])
    
    dist_permuted = _np.array(dist_permuted)
    p_value = _np.sum(dist_permuted<dist)/len(dist_permuted)
   
    if distanceonly:
        return(dist, p_value)
    else:
        normDist = alignment.rx('normalizedDistance')[0][0]
        index1 = _np.array(alignment.rx('index1')[0])
        index2 = _np.array(alignment.rx('index2')[0])
        index1s = _np.array(alignment.rx('index1s')[0])
        index2s = _np.array(alignment.rx('index2s')[0])
        N = alignment.rx('N')[0][0]
        M = alignment.rx('M')[0][0]
        #    openEnd = alignment.rx('openEnd')[0][0]
        #    openBegin = alignment.rx('openBegin')[0][0]
        #    windowFunction = alignment.rx('windowFunction()')[0]
        jmin = alignment.rx('jmin')[0][0]
        stepsTaken = _np.array(alignment.rx('stepsTaken')[0])
        costMatrix = _np.array(alignment.rx('costMatrix')[0])
        out = {'dist': dist, 'normDist': normDist, 'costMatrix': costMatrix, 'index1': index1, 'index2': index2,
               'index1s': index1s, 'index2s': index2s, 'N': N, 'M': M, 'jmin': jmin, 'stepsTaken': stepsTaken}
        return(dist, p_value, out)
'''