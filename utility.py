import numpy as np
from scipy import interpolate, signal


# supplementary methods
def power(spec, freq, fmin, fmax):
    #returns power in band
    band = np.array([spec[i] for i in range(len(spec)) if freq[i] >= fmin and freq[i] < fmax])
    powerinband = np.sum(band) / len(spec)
    return powerinband


def InterpolateRR(RR, Finterp):
    # returns cubic spline interpolated array with sample rate = Finterp
    step = 1 / Finterp
    BT = np.cumsum(RR)
    xmin = BT[0]
    xmax = BT[-1]
    BT = np.insert(BT, 0, 0)
    BT = np.append(BT, BT[-1] + 1)
    RR = np.insert(RR, 0, 0)
    RR = np.append(RR, RR[-1])

    tck = interpolate.splrep(BT, RR)
    BT_interp = np.arange(xmin, xmax, step)
    RR_interp = interpolate.splev(BT_interp, tck)
    return RR_interp, BT_interp


def BuildTakensVector(RR, m):
    #creo embedded matrix
    #righe = Uj
    N = len(RR)
    numelem = N - m + 1
    RRExp = np.zeros([numelem, m])
    for i in range(numelem):
        RRExp[i, :] = RR[i:i + m]
    return RRExp


def smoothTriangle(data, degree):
    triangle = np.array(range(degree) + [degree] + range(degree)[::-1]) + 1
    smoothed = data[0:degree]
    for i in range(degree, np.size(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed = np.append(smoothed, sum(point) / sum(triangle))
    smoothed = np.insert(smoothed, -1, data[-(degree * 2):])
    return smoothed


def peakdet(v, delta, x=None):
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def lowpass_filter(X, fs, Wp):
    nyq = 0.5 * fs
    wp = Wp / nyq  # pass band Hz
    ws = wp + 0.5
    N, wn = buttord(wp, ws, 5, 30)  # calcola ordine per il filtro e la finestra delle bande
    [bFilt, aFilt] = butter(N, wn, btype='lowpass')  # calcola coefficienti filtro
    sig = filtfilt(bFilt, aFilt, X)  # filtro il segnale BVP
    return sig


def highpass_filter(X, fs, Wp):
    nyq = 0.5 * fs
    wp = Wp / nyq # pass band Hz
    ws = wp - 0.01
    N, wn = buttord(wp, ws, 5, 30)  # calcola ordine per il filtro e la finestra delle bande
    [bFilt, aFilt] = butter(N, wn, btype='highpass')  # calcola coefficienti filtro
    sig = filtfilt(bFilt, aFilt, X)  # filtro il segnale BVP
    return sig
