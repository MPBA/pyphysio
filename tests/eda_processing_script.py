from __future__ import division
import numpy as _np
import os
import matplotlib.pyplot as plt
import pickle # per salvare dati in binario
import gzip # per comprimere i dati salvati
import pyphysio as ph

def phasic_estim_benedek(driver, delta):
    #find peaks in the driver
    fsamp = driver.get_sampling_freq()
    i_peaks, idx_min, val_max, val_min = ph.PeakDetection(delta=delta, refractory=1, start_max=True)(driver)
    
    i_pre_max = 10 * fsamp
    i_post_max = 10 * fsamp
        
    
    i_start = np.empty(len(i_peaks), int)
    i_stop = np.empty(len(i_peaks), int)

    if len(i_peaks)==0:
        print('No peaks found.')
        return driver
        
    for i in xrange(len(i_peaks)):
        i_pk = int(i_peaks[i])

        # find START
        i_st = i_pk - i_pre_max
        if i_st < 0:
            i_st=0

        driver_pre = driver[i_st:i_pk]
        i_pre = len(driver_pre) -2

        while i_pre > 0 and (driver_pre[i_pre] >= driver_pre[i_pre-1]):
            i_pre -= 1

        i_start[i] = i_st + i_pre + 1
        

        # find STOP
        i_sp = i_pk + i_post_max
        if i_sp >= len(driver):
            i_sp = len(driver) - 1
            
        driver_post = driver[i_pk: i_sp]
        i_post = 1

        while i_post < len(driver_post)-2 and (driver_post[i_post] >= driver_post[i_post+1]):
            i_post += 1

        i_stop[i] = i_pk + i_post
    
    idxs_peak = np.array([])
    
    for i_st, i_sp in zip(i_start, i_stop):
        idxs_peak = np.r_[idxs_peak, np.arange(i_st, i_sp)]
    
    idxs_peak = idxs_peak.astype(int)
    
    #generate the grid for the interpolation
    idx_grid_candidate = np.arange(0, len(driver) - 1, 10 * fsamp)

    idx_grid = []
    for i in idx_grid_candidate:
        if i not in idxs_peak:
            idx_grid.append(i)
    
    if len(idx_grid)==0:
        idx_grid.append(0)
    
    if idx_grid[0] != 0:
        idx_grid = np.r_[0, idx_grid]
    if idx_grid[-1] != len(driver) - 1:
        idx_grid = np.r_[idx_grid, len(driver) - 1]

    driver_grid = ph.UnevenlySignal(driver[idx_grid], fsamp, "dEDA", driver.get_start_time(), x_values = idx_grid, x_type = 'indices')
    if len(idx_grid)>=4:
        tonic = driver_grid.to_evenly(kind='cubic')
    else:
        tonic = driver_grid.to_evenly(kind='linear')
    phasic = driver - tonic
    return phasic

def _gen_bateman(fsamp, par_bat):
    """
    Generates the bateman function:

    :math:`b = e^{-t/T1} - e^{-t/T2}`

    Parameters
    ----------
    fsamp : float
        Sampling frequency
    par_bat: list (T1, T2)
        Parameters of the bateman function

    Returns
    -------
    bateman : array
        The bateman function
    """

    idx_T1 = par_bat[0] * fsamp
    idx_T2 = par_bat[1] * fsamp
    len_bat = idx_T2 * 10
    idx_bat = _np.arange(len_bat)
    bateman = _np.exp(-idx_bat / idx_T2) - _np.exp(-idx_bat / idx_T1)
    
    # normalize
    bateman = fsamp * bateman / _np.sum(bateman)
    return bateman


# 0 = ecg_bg  
# 1 = eda_bg  
# 2 = bvp_bg  
# 3 = resp_bg  
# 4 = trg_bg  
# 5 = bvp_e4  
# 6 = eda_e4  
# 7 = acc_e4  
# 8 = temp_e4  
# 9 = ecg_cb  
# 10 = acc_cb  

# define constants
FSAMP = 1024
DELTA = 0.02

SEGMENT = 'stimuli'
SIGNAL, ID_SIGNAL = 'eda_BG', 1

DATADIR = '/home/andrea/Trento/Lavori/Dataset_ABP/portions/'+SEGMENT+'/'
PAR_DIR = '/home/andrea/Trento/Lavori/Dataset_ABP/bateman_parameters/stimuli/eda_BG/'
OUT_DIR = '/home/andrea/Trento/Lavori/Dataset_ABP/PHA/'+SEGMENT+'/'+SIGNAL+'/'

subs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']

SUB = '01'

print('===============')
print(SUB)
print('===============')
os.chdir(DATADIR)

# load eda data
f = gzip.open(SUB + '_'+SEGMENT+'.pckl', 'rb')
data = pickle.load(f)
f.close()

eda = ph.EvenlySignal(data[:,ID_SIGNAL], FSAMP, 'EDA')

# downsampling
eda = eda.resample(8)
FSAMP = 8
#filtering
eda = ph.IIRFilter(fp=0.8, fs=1.1)(eda)

T1 = 0.75
T2 = 2

par_bat = [T1, T2]


### ESTIMATE DRIVER
bateman = _gen_bateman(FSAMP, par_bat)

idx_max_bat = _np.argmax(bateman)

# Prepare the input signal to avoid starting/ending peaks in the driver
bateman_first_half = bateman[0:idx_max_bat + 1]
bateman_first_half = eda[0] * (bateman_first_half - _np.min(bateman_first_half)) / (_np.max(bateman_first_half) - _np.min(bateman_first_half))

bateman_second_half = bateman[idx_max_bat:]
bateman_second_half = eda[-1] * (bateman_second_half - _np.min(bateman_second_half)) / (_np.max(bateman_second_half) - _np.min(bateman_second_half))

signal_in = _np.r_[bateman_first_half, eda.get_values(), bateman_second_half]
signal_in = ph.EvenlySignal(signal_in, FSAMP)

# deconvolution
driver = ph.DeConvolutionalFilter(irf=bateman, normalize=True, deconv_method='fft')(signal_in)
driver = driver[idx_max_bat + 1: idx_max_bat + len(eda)]

# gaussian smoothing
driver = ph.ConvolutionalFilter(irftype='gauss', win_len=_np.max([0.2, 1/FSAMP])*8, normalize=True)(driver)
driver = ph.EvenlySignal(driver, FSAMP, "dEDA", eda.get_start_time())


# ESTIMATE PHASIC
grid_size = 1
pre_max = 2
post_max = 2

#find peaks in the driver
idx_max, idx_min, val_max, val_min = ph.PeakDetection(delta=DELTA, refractory=1, start_max=True)(driver)

#identify start and stop of the peak
idx_pre, idx_post = ph.PeakSelection(maxs=idx_max, pre_max=pre_max, post_max=post_max)(driver)

# Linear interpolation to substitute the peaks
driver_no_peak = _np.copy(driver)
for I in range(len(idx_pre)):
    i_st = idx_pre[I]
    i_sp = idx_post[I]

    if _np.isnan(i_st)==False and _np.isnan(i_sp)==False:
        idx_base = _np.arange(i_sp - i_st)
        coeff = (driver[i_sp] - driver[i_st]) / len(idx_base)
        driver_base = idx_base * coeff + driver[i_st]
        driver_no_peak[i_st:i_sp] = driver_base

#generate the grid for the interpolation
idx_grid = _np.arange(0, len(driver_no_peak) - 1, grid_size * FSAMP)
idx_grid = _np.r_[idx_grid, len(driver_no_peak) - 1]

driver_grid = ph.UnevenlySignal(driver_no_peak[idx_grid], FSAMP, "dEDA", driver.get_start_time(), x_values = idx_grid, x_type = 'indices')
tonic = driver_grid.to_evenly(kind='cubic')

phasic = driver - tonic
