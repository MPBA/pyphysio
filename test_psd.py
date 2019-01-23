import numpy as np
import matplotlib.pyplot as plt
import pyphysio as ph

def fft(signal, fsamp, NFFT):
    f, pwd = ph.PSD('fft', nfft=NFFT, normalize=False)(signal.resample(fsamp))
    idx_band = np.where((f>0.35) & (f<0.45))[0]
    df = np.diff(f)[0]
    plt.plot(f, pwd)
    return(f, pwd, np.sum(pwd[idx_band])*df)

def welch(signal, fsamp, NFFT):
    f, pwd = ph.PSD('welch', nfft=NFFT, normalize=False)(signal.resample(fsamp))
    idx_band = np.where((f>0.35) & (f<0.45))[0]
    df = np.diff(f)[0]
    plt.plot(f, pwd)
    return(f, pwd, np.sum(pwd[idx_band])*df)
    
def ar(signal, fsamp, NFFT):
    f, pwd = ph.PSD('ar', nfft=NFFT, normalize=False)(signal.resample(fsamp))
    idx_band = np.where((f>0.35) & (f<0.45))[0]
    df = np.diff(f)[0]
    plt.plot(f, pwd)
    return(f, pwd, np.sum(pwd[idx_band])*df)

def test_methods(signal, fsamp, NFFT):
    f_f, pwd_f, inband_f = fft(signal, fsamp, NFFT)
    f_w, pwd_w, inband_w = welch(signal, fsamp, NFFT)
    f_a, pwd_a, inband_a = ar(signal, fsamp, NFFT)
    print(inband_f, inband_w, inband_a)
    
#%%
fsamp = 5
NFFT = 1024

signal = np.sin(np.linspace(0, 500, NFFT)) + 0.5*np.cos(np.linspace(0, 750, NFFT)) + np.random.randn(NFFT)*0.2 + 0.2*np.cos(np.linspace(0, 10000, NFFT))
#plt.plot(signal[:250])
signal = ph.EvenlySignal(signal, fsamp)

#%%
test_methods(signal, 5, 1024)

test_methods(signal, 10, 2048)

test_methods(signal, 20, 1024)

