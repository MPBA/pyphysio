# pipeline HRV analysis Esposito
# WINDSIZE/2 -> WINDSIZE
# ab:

from __future__ import division

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import optparse


def smooth(x, window_len=6, window='hanning'):
	import numpy as np
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len < 3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s = np.r_[2 * x[0] - x[window_len - 1:: -1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
        else:  
                w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len:-window_len + 1]


usage = "Usage: Labelizer [options]"
parser = optparse.OptionParser(usage=usage)

#directory where to find files
parser.add_option("-i", "--inputdir", 
                  action="store", type="string",
                  dest="input_dir", help="Input Dir")

# directory where to save HRVindexes files
parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")
# window lenght
parser.add_option("-w", "--windows",
                  action="store", type="int", default='40', 
                  dest="window_size", help="Window length [s]")

# jump lenght
parser.add_option("-j", "--jump",
                  action="store", type="int", default='0',
                  dest="jump", help="Moving interval in seconds for centering new windows set (0=each beat)")


(options, args) = parser.parse_args()
INPUTDIR = options.input_dir
OUTDIR = options.output_dir
WINSIZE = options.window_size
BEATJUMP = options.jump

print "Labelize"

subjects = os.listdir(INPUTDIR)
tot = len(subjects)
i = 0

subjects = os.listdir(INPUTDIR)

for subject in subjects:  # Each file
    i += 1
    rows = []

    FILENAME = os.path.join(INPUTDIR, subject)
    print "Processing " + subject + " (" + str(i) + "/" + str(tot) + ").."
    dataSUBJ = pd.read_csv(FILENAME, sep='\t', quotechar='"')
    RR = np.array(dataSUBJ['IBI'])  # millisec
    BT = np.cumsum(RR)  # millisec
    BT /= 1000.  # sec

    crySUBJ = np.array(dataSUBJ['cry'])
    sleepSUBJ = np.array(dataSUBJ['sleep'])
    calmSUBJ = abs(sleepSUBJ - 1) - crySUBJ
    threeSUBJ = abs(sleepSUBJ - 1) + crySUBJ
    distanceCRY_avanti = np.zeros(len(threeSUBJ))
    distanceCRY_indietro = np.zeros(len(threeSUBJ))
    
    distanceSL_avanti = np.zeros(len(threeSUBJ))
    distanceSL_indietro = np.zeros(len(threeSUBJ))
    
    distanceSUBJ = np.zeros(len(threeSUBJ))

    prevSession = crySUBJ[0]
    tStart_session = np.nan
    for jj in range(len(threeSUBJ)):
        currentSession = crySUBJ[jj]
        if not currentSession == prevSession:
            tStart_session = BT[jj - 1]
            prevSession = currentSession
        distanceCRY_avanti[jj] = BT[jj] - tStart_session

    nextSession = crySUBJ[-1]
    tEnd_session = np.nan
    for jj in range(len(crySUBJ) - 1, -1, -1):
        currentSession = crySUBJ[jj]
        if not currentSession == nextSession:
            tEnd_session = BT[jj + 1]
            nextSession = currentSession
        distanceCRY_indietro[jj] = tEnd_session - BT[jj]

    prevSession = sleepSUBJ[0]
    tStart_session = np.nan
    for jj in range(len(threeSUBJ)):
        currentSession = sleepSUBJ[jj]
        if not currentSession == prevSession:
            tStart_session = BT[jj - 1]
            prevSession = currentSession
        distanceSL_avanti[jj] = BT[jj] - tStart_session

    nextSession = sleepSUBJ[-1]
    tEnd_session = np.nan
    for jj in range(len(crySUBJ) - 1, -1, -1):
        currentSession = sleepSUBJ[jj]
        if not currentSession == nextSession:
            tEnd_session = BT[jj + 1]
            nextSession = currentSession
        distanceSL_indietro[jj] = tEnd_session - BT[jj]

    for jj in range(len(threeSUBJ)):
        currDistCRY = np.nanmin([distanceCRY_avanti[jj], distanceCRY_indietro[jj]])
        currDistSL = np.nanmin([distanceSL_avanti[jj], distanceSL_indietro[jj]])
        if sleepSUBJ[jj] == 1:
            currDist = currDistCRY + currDistSL + 1
        else:
            currDist = currDistCRY + 1

        currDist = np.log(currDist)

        if crySUBJ[jj] == 1:
            A = -1
        else:
            A = 1

        distanceSUBJ[jj] = A * currDist

    sit = np.array(dataSUBJ['sit'], dtype='str')
    sitSUBJ = np.zeros(len(sit), dtype='float')

    for k in xrange(len(sit)):
        if sit[k] == 'SH' or sit[k] == 'HS':
            sitSUBJ[k] = 1
        if sit[k] == 'WH' or sit[k] == 'HW':
            sitSUBJ[k] = 2
    
    sitSUBJ_smooth = smooth(sitSUBJ, 6, 'hanning')
    movingSUBJ = sitSUBJ_smooth % 1
    
    
    
    labels = ['win_size', 'ind_start', 'ind_center', 'ind_final', 't_start', 't_center', 't_final',
    'cry_start', 'cry_center', 'cry_final', 'calm_start', 'calm_center', 'calm_final', 
    'sleep_start', 'sleep_center', 'sleep_final', 'three_start', 'three_center', 'three_final', 
    'sit_start', 'sit_center', 'sit_final', 'sit_trans_start', 'sit_trans_center', 'sit_trans_final',
    'trans_start', 'trans_center', 'trans_final',
    'distance_start', 'distance_center', 'distance_final',
    'calm_perc', 'cry_perc', 'sleep_perc']
    
    first = True
    tStart = BT[0]
    tFinal = BT[-1] - np.ceil(WINSIZE)
    while tStart < tFinal:
        # select beats in window
        IND_tStart = np.argmin(abs(BT - tStart))
    
        IND_tEnd = np.argmin(abs(BT - (tStart + WINSIZE)))
        tEnd = BT[IND_tEnd]
        
        IND_tCenter = np.argmin(abs(BT - (tStart + WINSIZE / 2)))
        tCenter = BT[IND_tCenter]
        
        BT_window = BT[IND_tStart:IND_tEnd]
        
        len_window = len(BT_window)
        
        CRY_window = crySUBJ[IND_tStart:IND_tEnd]
        percCRY = sum(CRY_window) / len_window
        
        CALM_window = calmSUBJ[IND_tStart:IND_tEnd]
        percCALM = sum(CALM_window) / len_window
        
        SLEEP_window = sleepSUBJ[IND_tStart:IND_tEnd]
        percSLEEP = sum(SLEEP_window) / len_window
        
        LAB = [WINSIZE,
        IND_tStart,            IND_tCenter,            IND_tEnd, 
        tStart,                tCenter,                tEnd,
        crySUBJ[IND_tStart],   crySUBJ[IND_tCenter],   crySUBJ[IND_tEnd], 
        calmSUBJ[IND_tStart],  calmSUBJ[IND_tCenter],  calmSUBJ[IND_tEnd], 
        sleepSUBJ[IND_tStart], sleepSUBJ[IND_tCenter], sleepSUBJ[IND_tEnd], 
        threeSUBJ[IND_tStart], threeSUBJ[IND_tCenter], threeSUBJ[IND_tEnd],
        sitSUBJ[IND_tStart],   sitSUBJ[IND_tCenter],   sitSUBJ[IND_tEnd],
        sitSUBJ_smooth[IND_tStart],   sitSUBJ_smooth[IND_tCenter],   sitSUBJ_smooth[IND_tEnd],
        movingSUBJ[IND_tStart], movingSUBJ[IND_tCenter], movingSUBJ[IND_tEnd],
        distanceSUBJ[IND_tStart], distanceSUBJ[IND_tCenter], distanceSUBJ[IND_tEnd],
        percCRY, percCALM, percSLEEP]
        
        rows.append(LAB)
        if BEATJUMP == 0:
            tStart = BT[IND_tStart + 1]
        else:
            tStart += BEATJUMP

    n, e = os.path.splitext(subject)
    pd.DataFrame(rows, columns=labels).to_csv(os.path.join(OUTDIR, n + "_LAB" + e), sep='\t')

