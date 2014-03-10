from __future__ import division
from scipy.signal import butter, buttord, bessel, filtfilt
import numpy as np
from scipy import fft, arange
import sys


def smooth_triangle(data, degree):
    triangle = np.array(range(degree)+[degree]+range(degree)[::-1])+1
    smoothed = data[0:degree]
    for i in range(degree, np.size(data)-degree*2):
        point=data[i:i+len(triangle)]*triangle
        smoothed= np.append(smoothed, sum(point)/sum(triangle))
    smoothed=np.insert(smoothed, -1,  data[-(degree*2):])
    return smoothed

def plotSpectrum(y,Fs):
    import matplotlib.pyplot as plt
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(np.floor(n/2)))] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(int(np.floor(n/2)))]
    
    plt.plot(frq, abs(Y), 'r') # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()

def peakdet(v, delta, x = None):
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
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
                
    return np.array(maxtab), np.array(mintab)

def lowpass_filter(X, fs, Wp):
    nyq = 0.5 * fs
    wp=Wp/nyq # pass band Hz
    ws=wp+0.5
    N, wn = buttord(wp, ws, 5, 30) # calcola ordine per il filtro e la finestra delle bande
    [bFilt, aFilt] = butter(N, wn, btype='lowpass') # calcola coefficienti filtro
    sig=filtfilt(bFilt,aFilt,X) # filtro il segnale BVP
    return sig
    
def highpass_filter(X, fs, Wp):
    nyq = 0.5 * fs
    wp=Wp/nyq # pass band Hz
    ws=wp-0.01
    N, wn = buttord(wp, ws, 5, 30) # calcola ordine per il filtro e la finestra delle bande
    [bFilt, aFilt] = butter(N, wn, btype='highpass') # calcola coefficienti filtro
    sig=filtfilt(bFilt,aFilt,X) # filtro il segnale BVP
    return sig
