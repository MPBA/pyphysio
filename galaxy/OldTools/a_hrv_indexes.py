from __future__ import division
from numpy import *
from scipy import interpolate

import spectrum as spct



#####################################
## NOTE
# PSD estimation: AR method
# INTERPOLATION: cubic spline
# ! ! !
# RR in milliseconds!
#####################################


# MAIN FUNCTION
# calls sub functions, each one for calculating similar type indexes
def calculate_hrv_indexes(rr, f_interp=4, return_labels=False):
    # comment lines below to avoid calculation of an index type
    TDindexes, TDlabels = calculate_td_indexes(rr)  # TD indexes
    FDindexes, FDlabels = calculate_fd_indexes(rr, f_interp)  # FD indexes
    NLindexes, NLlabels = calculate_non_lin_indexes(rr)  # Non linear indexes
    POINindexes, POINlabels = calculate_poin_indexes(rr)  # Poincare' plot indexes
    Hindex, Hlabel = hurst(rr)  # hurst
    PFDindex, PFDlabel = pfd(rr)  # petrosian fractal dimension
    DFAindex, DFAlabel = dfa(rr)  # detrended fluctation analysis

    # remove not calculated indexes
    indexes = np.hstack((TDindexes, FDindexes, NLindexes, POINindexes, Hindex, PFDindex, DFAindex))

    if return_labels:
        labels = np.hstack((TDlabels, FDlabels, NLlabels, POINlabels, Hlabel, PFDlabel, DFAlabel))
        return indexes, labels
    else:
        return indexes


def calculate_td_indexes(rr):
    # calculates Time domain indexes
    RRmean = np.mean(rr)
    RRSTD = np.std(rr)

    RRDiffs = np.diff(rr)

    RRDiffs50 = [x for x in np.abs(RRDiffs) if x > 50]
    pNN50 = 100.0 * len(RRDiffs50) / len(RRDiffs)
    RRDiffs25 = [x for x in np.abs(RRDiffs) if x > 25]
    pNN25 = 100.0 * len(RRDiffs25) / len(RRDiffs)
    RRDiffs10 = [x for x in np.abs(RRDiffs) if x > 10]
    pNN10 = 100.0 * len(RRDiffs10) / len(RRDiffs)

    RMSSD = np.sqrt(sum(RRDiffs ** 2) / (len(RRDiffs) - 1))
    SDSD = np.std(RRDiffs)

    labels = np.array(['RRmean', 'STD', 'pNN50', 'pNN25', 'pNN10', 'RMSSD', 'SDSD'], dtype='S10')

    return [RRmean, RRSTD, pNN50, pNN25, pNN10, RMSSD, SDSD], labels


def calculate_fd_indexes(rr, f_interp):
    def _power(spec, freq, fmin, fmax):
        #returns power in band
        band = np.array([spec[i] for i in xrange(len(spec)) if freq[i] >= fmin and freq[i] < fmax])
        powerinband = np.sum(band) / len(spec)
        return powerinband

    def interpolate_rr(_rr, _f_interp):
        # returns cubic spline interpolated array with sample rate = Finterp
        step = 1 / _f_interp
        BT = np.cumsum(_rr)
        xmin = BT[0]
        xmax = BT[-1]
        BT = np.insert(BT, 0, 0)
        BT = np.append(BT, BT[-1] + 1)
        _rr = np.insert(_rr, 0, 0)
        _rr = np.append(_rr, _rr[-1])

        tck = interpolate.splrep(BT, _rr)
        BT_interp = np.arange(xmin, xmax, step)
        RR_interp = interpolate.splev(BT_interp, tck)
        return RR_interp, BT_interp

    rr /= 1000  # RR in seconds
    RR_interp, BT_interp = interpolate_rr(rr, f_interp)
    RR_interp = RR_interp - np.mean(RR_interp)

    freqs = np.arange(0, 2, 0.0001)

    # calculates AR coefficients
    AR, P, k = spct.arburg(RR_interp * 1000, 16)  # burg

    # estimates PSD from AR coefficients
    spec = spct.arma2psd(AR, T=0.25, NFFT=2 * len(freqs))
    spec = spec[0:len(spec) / 2]

    # calculates power in different bands
    VLF = _power(spec, freqs, 0, 0.04)
    LF = _power(spec, freqs, 0.04, 0.15)
    HF = _power(spec, freqs, 0.15, 0.4)
    Total = _power(spec, freqs, 0, 2)
    LFHF = LF / HF
    nVLF = VLF / Total
    nLF = LF / Total
    nHF = HF / Total

    LFn = LF / (HF + LF)
    HFn = HF / (HF + LF)
    Power = [VLF, HF, LF]

    Power_Ratio = Power / sum(Power)
    # Power_Ratio=spec/sum(spec) # uncomment to calculate Spectral Entropy using all frequencies
    Spectral_Entropy = 0
    lenPower = 0  # tengo conto delle bande che ho utilizzato
    for i in xrange(0, len(Power_Ratio)):
        if Power_Ratio[i] > 0:  # potrei avere VLF=0
            Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
            lenPower += 1
    Spectral_Entropy /= np.log(lenPower)  # al posto di len(Power_Ratio) perche' magari non ho usato VLF

    labels = np.array(['VLF', 'LF', 'HF', 'Total', 'nVLF', 'nLF', 'nHF', 'LFn', 'HFn', 'LFHF', 'SpecEn'], dtype='S10')

    return [VLF, LF, HF, Total, nVLF, nLF, nHF, LFn, HFn, LFHF, Spectral_Entropy], labels


def calculate_non_lin_indexes(rr):
    #calculates non linear HRV indexes

    def build_takens_vector(rr, m):
        #returns embedded matrix
        n = len(rr)
        numelem = n - m + 1
        RRExp = np.zeros([numelem, m])
        for i in range(numelem):
            RRExp[i, :] = rr[i:i + m]
        return RRExp

    def avg_integral_correlation(RR, m, r):
        # calculates distances among embedded patterns
        from scipy.spatial.distance import cdist

        RRExp = build_takens_vector(RR, m)
        numelem = RRExp.shape[0]
        mutualDistance = cdist(RRExp, RRExp, 'chebyshev')
        Cmr = np.zeros(numelem)
        for i in range(numelem):
            vector = mutualDistance[i]
            Cmr[i] = float((vector <= r).sum()) / numelem

        Phi = (np.log(Cmr)).sum() / len(Cmr)

        return Phi

    def calculate_ap_en(rr, m=2,
                        r=0.2):  # m = lunghezza pattern [kubios m = 2], r=soglia per somiglianza [Kubios r = 0.2]??
        # calculates Approximate Entropy
        # m= pattern length, r=tolerance (coefficient for std)
        r = r * np.std(rr)
        Phi1 = avg_integral_correlation(rr, m, r)
        Phi2 = avg_integral_correlation(rr, m + 1, r)
        ApEn = Phi1 - Phi2

        return ApEn

    def samp_entropy(X, M=2, R=0.2):
        # calculates Sample Entropy
        def in_range(Template, Scroll, Distance):
            for i in range(0, len(Template)):
                if abs(Template[i] - Scroll[i]) > Distance:
                    return False
            return True

        def embed_seq(X, Tau, D):
            N = len(X)

            if D * Tau > N:
                print "Cannot build such a matrix, because D * Tau > N"
                exit()

            if Tau < 1:
                print "Tau has to be at least 1"
                exit()

            Y = np.zeros((N - (D - 1) * Tau, D))
            for i in xrange(0, N - (D - 1) * Tau):
                for j in xrange(0, D):
                    Y[i][j] = X[i + j * Tau]
            return Y

        R *= np.std(rr)
        N = len(X)

        Em = embed_seq(X, 1, M)
        Emp = embed_seq(X, 1, M + 1)

        Cm, Cmp = np.zeros(N - M - 1) + 1e-100, np.zeros(N - M - 1) + 1e-100
        # in case there is 0 after counting. Log(0) is undefined.

        for i in xrange(0, N - M):
            for j in xrange(i + 1, N - M):  # no self-match
                #			if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1
                if in_range(Em[i], Em[j], R):
                    Cm[i] += 1
                    #			if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                    if abs(Emp[i][-1] - Emp[j][-1]) <= R:  # check last one
                        Cmp[i] += 1

        Samp_En = log(sum(Cm) / sum(Cmp))

        return Samp_En

    def calculate_frac_dim(RR, m=8, Cra=0.005, Crb=0.75):  # was m=2
        from scipy.spatial.distance import pdist
        from scipy.stats.mstats import mquantiles

        RRExp = build_takens_vector(RR, m)

        mutualDistance = pdist(RRExp, 'chebyshev')

        numelem = len(mutualDistance)

        rr = mquantiles(mutualDistance, prob=[Cra, Crb])
        ra = rr[0]
        rb = rr[1]

        if numelem != 0:
            Cmra = float(((mutualDistance <= ra).sum())) / numelem
            Cmrb = float(((mutualDistance <= rb).sum())) / numelem
            return np.float(np.log(Cmrb) - np.log(Cmra)) / (np.log(rb) - np.log(ra))
        else:
            return np.nan

    ApEn = calculate_ap_en(rr)  ##
    SampEn = samp_entropy(rr)
    FracDim = calculate_frac_dim(rr)  ##

    RRExp = build_takens_vector(rr, 2)
    W = np.linalg.svd(RRExp, compute_uv=0)
    W /= sum(W)
    SVDEn = -1 * sum(W * log(W))

    FI = 0
    for i in xrange(0, len(W) - 1):  # from 1 to M
        FI += ((W[i + 1] - W[i]) ** 2) / (W[i])

    labels = ['ApproxEntropy', 'SampleEntropy', 'FractalDimension', 'SVDEntropy', 'FI']

    return [ApEn, SampEn, FracDim, SVDEn, FI], labels


def calculate_poin_indexes(rr):
    # calculates Poincare' indexes
    xdata, ydata = rr[:-1], rr[1:]
    sd1 = np.std((xdata - ydata) / np.sqrt(2.0), ddof=1)
    sd2 = np.std((xdata + ydata) / np.sqrt(2.0), ddof=1)
    sd12 = sd1 / sd2
    sEll = sd1 * sd2 * np.pi
    labels = ['sd1', 'sd2', 'sd12', 'sEll']

    return [sd1, sd2, sd12, sEll], labels


def hurst(x):
    #calculates hurst exponent
    N = len(x)
    T = np.array([float(i) for i in xrange(1, N + 1)])
    Y = np.cumsum(x)
    Ave_T = Y / T

    S_T = np.zeros((N))
    R_T = np.zeros((N))
    for i in xrange(N):
        S_T[i] = std(x[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.max(X_T[:i + 1]) - np.min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = np.linalg.lstsq(n[1:], R_S[1:])[0]
    return H[0], 'hurst'


def pfd(x, d=None):
    #calculates petrosian fractal dimension
    if d is None:  ## Xin Liu
        d = np.diff(x)
    N_delta = 0;  #number of sign changes in derivative of the signal
    for i in xrange(1, len(d)):
        if d[i] * d[i - 1] < 0:
            N_delta += 1
    n = len(x)
    return np.float(log10(n) / (log10(n) + log10(n / n + 0.4 * N_delta))), 'pfd'


def dfa(x, ave=None):
    #calculates Detrended Fluctuation Analysis splitted in alpha1 (short term) and alpha2 (long term)
    x = array(x)
    if ave is None:
        ave = mean(x)
    Y = cumsum(x)
    Y -= ave

    lunghezza = len(x)
    lMax = np.floor(lunghezza / 4)
    L = np.arange(4, 17, 4)
    F = zeros(len(L))  # F(n) of different given box length n
    for i in xrange(0, len(L)):
        n = int(L[i])  # for each box length L[i]
        for j in xrange(0, len(x), n):  # for each box
            if j + n < len(x):
                c = range(j, j + n)
                c = vstack([c, ones(n)]).T  # coordinates of time in the box
                y = Y[j:j + n]  # the value of data in the box
                F[i] += np.linalg.lstsq(c, y)[1]  # add residue in this box
        F[i] /= ((len(x) / n) * n)
    F = np.sqrt(F)
    try:
        Alpha1 = np.linalg.lstsq(vstack([log(L), ones(len(L))]).T, log(F))[0][0]
    except ValueError:
        Alpha1 = np.nan

    lMax = np.min([64, len(x)])
    L = np.arange(4, lMax + 1, 4)
    F = zeros(len(L))  # F(n) of different given box length n
    for i in xrange(0, len(L)):
        n = int(L[i])  # for each box length L[i]
        for j in xrange(0, len(x), n):  # for each box
            if j + n < len(x):
                c = range(j, j + n)
                c = vstack([c, ones(n)]).T  # coordinates of time in the box
                y = Y[j:j + n]  # the value of data in the box
                F[i] += np.linalg.lstsq(c, y)[1]  # add residue in this box
        F[i] /= ((len(x) / n) * n)
    F = np.sqrt(F)
    try:
        Alpha2 = np.linalg.lstsq(vstack([log(L), ones(len(L))]).T, log(F))[0][0]
    except ValueError:
        Alpha2 = np.nan

    return [Alpha1, Alpha2], ['alpha1', 'alpha2']
