from __future__ import division
__author__ = 'andrea'
import cmath
from PyHRV.utility import *

np.random.seed()
RR=np.random.uniform(500,1000,100)
RR=RR/1000

# TODO: normalize to nsamples
# TODO: test initialization
# TODO: manage interpolation

vlf_th=0.04
lf_th=0.15
hf_th=0.4

def root_unity(fsampl, N):
    freqs=np.arange(0,fsampl,fsampl/N)
    root_unity=[]
    for k in range(len(freqs)):
        root_unity.append(cmath.exp(2j*np.pi*k/L)) #k is freq or index of coefficien?

    return root_unity, freqs

def online_fft(old, new, fft_coeff, fsamp):
    # we could return only the updated PSD (P) and then use normal methods to compute Indexes
    roots, freqs = root_unity(fsamp, len(fft_coeff)) # cacheable, no need to recalculate every time.
    vlf=0
    lf=0
    hf=0
    total=0
    vlf_max=0
    lf_max=0
    hf_max=0

    for index in range(len(freqs)):
        fft_coeff[index] = fft_coeff[index] + new - old

        # factor1 = fft_coeff[index].real * root_unity[index].real
        # factor2 = fft_coeff[index].imag * root_unity[index].imag
        # factor3 = (fft_coeff[index].real + fft_coeff[index].imag) * (root_unity[index].real + root_unity[index].imag)
        # fft_coeff[index] = (factor1 - factor2) + 1j*(factor3 - factor1 - factor2)

        fft_coeff[index]=fft_coeff[index]*roots[index]
        P = fft_coeff[index].real**2 + fft_coeff[index].imag**2

        if (freqs[index] > 0) & (freqs[index] <= vlf_th):
            vlf += P
            if P>vlf_max: vlf_p=freqs[index]
        elif (freqs[index] > vlf_th) & (freqs[index] <= lf_th):
            lf +=P
            if P>lf_max: lf_p=freqs[index]
        elif (freqs[index] > lf_th) & (freqs[index] <= hf_th):
            hf +=P
            if P>hf_max: hf_p=freqs[index]
        total +=P
    return vlf,lf,hf,total # list P?