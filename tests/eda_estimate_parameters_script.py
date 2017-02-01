from __future__ import division
import numpy as _np
import os
import matplotlib.pyplot as plt
import pickle # per salvare dati in binario
import gzip # per comprimere i dati salvati
import pyPhysio as ph

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

WLEN = 10

driver = ph.DriverEstim(T1=0.75, T2=2)(eda)

maxp, minp, ignored, ignored = ph.PeakDetection(delta=DELTA, start_max=True)(driver)

driver.plot()
plt.vlines(driver.get_times()[maxp], np.min(driver), np.max(driver))

for i in range(len(maxp)):
    c_i = maxp[i]
    plt.text(driver.get_times()[c_i], 1, str(i))

# PEAK BASED
#402 doube peak
idx_max = maxp[402]
    
driver_portion = driver.segment_idx(idx_max,idx_max + WLEN * fsamp)
maxp, minp, ignored, minv = ph.PeakDetection(delta=DELTA/10, start_max=True)(driver_portion)

y = driver_portion[minp]
t = minp

diff_y = (y[1:] - y[:-1])/(t[1:] - t[:-1])

th_75 = _np.percentile(diff_y, 75)
th_25 = _np.percentile(diff_y, 25)

idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
diff_y_sel = diff_y[idx_sel_diff_y]

mean_s = ph.BootstrapEstimation(func=_np.mean, N=10, k=0.5)(diff_y_sel)

mean_y = ph.BootstrapEstimation(func=_np.median, N=10, k=0.5)(y)

b_mean_s = mean_y - mean_s * len(driver_portion)/2

line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s

#colors = ['violet', 'cyan', 'red', 'yellow', 'blue']
colors = ['#6c71c4', '#2aa198', '#dc322f', '#b58900', '#268bd2']

plt.figure(figsize=(7,5))
plt.style.use('ggplot')
plt.plot(driver.get_times(), driver, linewidth = 1, color = '#2aa198')
plt.plot(driver_portion.get_times(), driver_portion, linewidth=2.5, color = '#268bd2')

plt.plot(driver_portion.get_times()[minp], minv, '^', color = '#dc322f', markersize=10)
plt.plot(driver_portion.get_times(), line_mean_s, '--', linewidth=2.5, color = '#b58900')
plt.xlim((2282, 2295))
plt.ylim((2.32, 2.49))
plt.ylabel('EDA [$\mu$S]')
plt.xlabel('Time [s]')
plt.legend(['driver', 'selected portion', 'local minima', 'linear support'])
plt.tight_layout()


# PEAK 10
#402 doube peak
idx_max = maxp[219]
    
driver_portion = driver.segment_idx(idx_max,idx_max + WLEN * fsamp)
#extract final 5 seconds
half = len(driver_portion) - 5 * fsamp

#estimate slope to detrend driver
y = driver_portion[half:]
diff_y = _np.diff(y)
th_75 = _np.percentile(diff_y, 70)
th_25 = _np.percentile(diff_y, 30)

idx_sel_diff_y = _np.where((diff_y > th_25) & (diff_y < th_75))[0]
diff_y_sel = diff_y[idx_sel_diff_y]

mean_s = ph.BootstrapEstimation(func=_np.mean, N=10, k=0.5)(diff_y_sel)

mean_y = ph.BootstrapEstimation(func=_np.median, N=10, k=0.5)(y)

b_mean_s = mean_y - mean_s * (half + (len(driver_portion) - half) / 2)

line_mean_s = mean_s * _np.arange(len(driver_portion)) + b_mean_s


plt.figure(figsize=(7,5))
plt.style.use('ggplot')
plt.plot(driver.get_times(), driver, linewidth = 1, color = '#2aa198')
plt.plot(driver_portion.get_times(), driver_portion, linewidth=2.5, color = '#268bd2')

plt.plot(driver_portion.get_times()[half:], y, color = '#dc322f', linewidth=2.5)
plt.plot(driver_portion.get_times(), line_mean_s, '--', linewidth=2.5, color = '#b58900')
plt.xlim((1245, 1257))
plt.ylim((1.82, 2.02))
plt.ylabel('EDA [$\mu$S]')
plt.xlabel('Time [s]')
plt.legend(['driver', 'selected portion', 'last 5 seconds', 'linear support'])
plt.tight_layout()