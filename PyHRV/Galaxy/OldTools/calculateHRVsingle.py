# pipeline HRV analysis Esposito
# extracts HRV indexes from a RRinterval series using single moving window
# ab:

import optparse

import pandas as pd

from PyHRV.Galaxy.OldTools.a_hrv_indexes import *

usage = "Usage: HRV analysis [options]"
parser = optparse.OptionParser(usage=usage)

# directory where to find files
parser.add_option("-i", "--inputdir", 
                  action="store", type="string",
                  dest="input_dir", help="Input Dir")

# directory where to save HRV Indexes files
parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")
# window length
parser.add_option("-w", "--windows",
                  action="store", type="int", default='40', 
                  dest="window_size", help="Window length [s]")

# jump length
parser.add_option("-j", "--jump",
                  action="store", type="int", default='20', 
                  dest="jump", help="Moving interval in seconds for centering new windows set (0=each beat)")

# more options...
parser.add_option("-f", "--fsamp",
                  action="store", type="int", default='4', 
                  dest="samplingFreq", help="Sampling frequency for PSD estimation")

### PSD o PDS?
parser.add_option("-p", "--psd",
                  action="store", type="string", default='ar', 
                  dest="psd_mode", help="PSD estimation method")
                  

(options, args) = parser.parse_args()
INPUTDIR = options.input_dir
OUTDIR = options.output_dir
WINSIZE = options.window_size
BEATJUMP = options.jump
FSAMP = options.samplingFreq
PSD_MODE = options.psd_mode

print "Calculate HRV - Single Window"

subjects = os.listdir(INPUTDIR)
tot = len(subjects)
i = 0
first = True
labels_HRV = ['ind_start', 'ind_center', 'ind_final', 't_start', 't_center', 't_final', 'win_size']

for subject in subjects:  # Each file
    i += 1
    rows = []
    FILENAME = os.path.join(INPUTDIR, subject)
    print "Processing " + subject + " (" + str(i) + "/" + str(tot) + ").."
    # Initialize data file
    # Load subject data
    dataSUBJ = pd.read_csv(FILENAME, sep='\t', quotechar='"')
    RR = np.array(dataSUBJ['IBI'])  # milliseconds
    BT = np.cumsum(RR)  # milliseconds
    BT /= 1000.  # to seconds
    
    # initialization for cycle
    tStart = BT[0]
    tFinal = BT[-1] - np.ceil(WINSIZE / 2.)

    while tStart < tFinal:
        # find session indexes and instants
        IND_tStart = np.argmin(abs(BT - tStart))
        IND_tEnd = np.argmin(abs(BT - (tStart + WINSIZE)))
        tEnd = BT[IND_tEnd]
        
        IND_tCenter = np.argmin(abs(BT - (tStart + WINSIZE / 2.)))
        tCenter = BT[IND_tCenter]

        # select data in window
        RR_window = RR[IND_tStart:IND_tEnd]
        BT_window = BT[IND_tStart:IND_tEnd]

        # calculate HRV indexes
        if first:
            HRV_window, labels_window = calculate_hrv_indexes(RR_window, return_labels=True)
            labels_HRV = np.hstack([labels_HRV, labels_window])
            first = False
        else:
            HRV_window = calculate_hrv_indexes(RR_window)
        
        # write file
        HRV = np.hstack([[IND_tStart, IND_tCenter, IND_tEnd, tStart, tCenter, tEnd, WINSIZE], HRV_window])
        rows.append(HRV)
        
        # update tStart
        if BEATJUMP == 0:
            tStart = BT[IND_tStart + 1]
        else:
            tStart += BEATJUMP

    n, e = os.path.splitext(subject)
    pd.DataFrame(rows, columns=labels_HRV).to_csv(os.path.join(OUTDIR, n + "_HRV" + e), sep='\t')
