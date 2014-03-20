# pipeline HRV analysis Esposito
# extracts HRV indexes from a RRinterval series using multiple moving windows
# ab: da finire?
## TODO: aggiunegere opzione di 'distanza' dal transitorio

from __future__ import division
import optparse

import pandas as pd

from Galaxy.OldTools.a_hrv_indexes import *

usage = "Usage: HRV analysis [options]"
parser = optparse.OptionParser(usage=usage)

#directory where to find files
parser.add_option("-i", "--inputdir", 
                  action="store", type="string",
                  dest="input_dir", help="Input Dir")

#directory where to save HRVindexes files
parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")
# parameters of windows length
parser.add_option("-w", "--windows",
                  action="store", type="string", default='40,40,40', 
                  dest="windows_opt", help="Windowing options: Min length, Max length, step")
# jump length
parser.add_option("-j", "--jump",
                  action="store", type="int", default='20', 
                  dest="jump", help="Moving interval in seconds for centering new windows set (0=each beat)")

# more options...
parser.add_option("-f", "--fsamp",
                  action="store", type="int", default='4', 
                  dest="samplingFreq", help="Sampling frequency")

parser.add_option("-p", "--pds",
                  action="store", type="string", default='ar', 
                  dest="psd_mode", help="PSD estimation method")
                  
(options, args) = parser.parse_args()
INPUTDIR = options.input_dir
OUTDIR = options.output_dir
BEATJUMP = options.jump
FSAMP = options.samplingFreq
PSD_MODE = options.psd_mode
WINDOPT = options.windows_opt.split(',')

WINMIN = int(WINDOPT[0])
WINMAX = int(WINDOPT[1])
WINSTEP = int(WINDOPT[2])

subjects = os.listdir(INPUTDIR)

WINSIZE = range(WINMIN, WINMAX+1, WINSTEP)  # Windowsize vector

for subject in subjects:  # Each file
    rows = []

    # Load subject data
    dataSUBJ = pd.read_csv(subject, sep='\t', quotechar='"')
    RR = np.array(dataSUBJ['IBI'])  # milliseconds
    BT = np.cumsum(RR)  # milliseconds
    BT /= 1000  # to seconds
    
    # Cycle vars
    tStart = BT[0]
    tFinal = BT[-1]-np.ceil(WINMAX/2)
    Labels_HRV = ['ind_start', 't_start']
    
    while tStart < tFinal:
        HRV = [IND_tStart, tStart]

        writeRow = True
        # for each windowsize
        for window in WINSIZE:
            # select beats in window
            ### BT_session, RR_session == ?
            IND_tStart = np.argmin(abs(BT_session-tStart))
            IND_tEnd = np.argmin(abs(BT_session-(tStart+window)))
            RR_window = RR_session[IND_tStart:IND_tEnd]
            
            if len(RR_window) >= 50:  # Does not calculate HRV indexes if n_beats <50
            
                # HRV indexes calculation
                HRV_window, Labels_window = calculate_hrv_indexes(RR_window, return_labels=True)

                # add windsize to label
                for i in xrange(0, len(subjects)):
                    Labels_window[i] = Labels_window[i]+"_"+str(window)
                    
                # join with old windsize HRV and labels    
                HRV = np.hstack([HRV, HRV_window])

                ### Labels_HRV prima si chiamava Labels in questa riga ma non era assegnato
                Labels_HRV = np.hstack([Labels_HRV, Labels_window])
            else:
                writeRow = False
            
        if writeRow:
            rows.append(HRV)
            
        # update tStart
        if BEATJUMP == 0:
            tStart = BT[IND_tStart+1]
        else:
            tStart += BEATJUMP
        
    pd.DataFrame(rows, columns=Labels_HRV).to_csv(os.path.join(OUTDIR, subject + '_HRV.txt'), sep='\t')
