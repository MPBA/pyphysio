# pipeline HRV analysis Esposito
# normalize RR interval series
# ab:

from __future__ import division

import numpy as np
import os
import pandas as pd
import optparse

usage = "Usage: HRV analysis [options]"
parser = optparse.OptionParser(usage=usage)

# directory where to find files
parser.add_option("-i", "--inputdir",
                  action="store", type="string",
                  dest="input_dir", help="Input Dir")

# directory where to save HRVindexes files
parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")
# normalization method
parser.add_option("-m", "--mode",
                  action="store", type='int', default='1',
                  dest='norm_mode', help="PSD estimation method:"
                                         " 1) RR-mean;"
                                         " 2) (RR-mean)/sd;"
                                         " 3) RR - min;"
                                         " 4) (RR-min)/(max-min);"
                                         " 5) RRALL*RR/meanCALM")

parser.add_option("-p", "--par",
                  action="store", type='int', default='420',
                  dest='norm_par', help='Parameter for normalization (used with mode=5)')

(options, args) = parser.parse_args()
INPUTDIR = options.input_dir
OUTDIR = options.output_dir
MODE = options.norm_mode
PAR = options.norm_par

print "Normalize RR"

subjects = os.listdir(INPUTDIR)
tot = len(subjects)
i = 0

for subject in subjects:  # For each file
    i += 1
    FILENAME = os.path.join(INPUTDIR, subject)
    print "Processing " + subject + " (" + str(i) + "/" + str(tot) + ").."
    # Load subject data
    dataSUBJ = pd.read_csv(FILENAME, sep='\t', quotechar='"')
    RR = np.array(dataSUBJ['IBI'])  # msec
    crySUBJ = np.array(dataSUBJ['cry'])
    sleepSUBJ = np.array(dataSUBJ['sleep'])
    calmSUBJ = abs(sleepSUBJ - 1) - crySUBJ

    RRmean = np.mean(RR)
    RRmeanCALM = np.mean(RR[calmSUBJ == 1])
    RRstd = np.std(RR)
    RRmin = min(RR)
    RRmax = max(RR)

    if MODE == 1:  # x-mean
        RRout = RR - RRmean

    elif MODE == 2:  # (x-mean)/std
        RRout = (RR - RRmean) / RRstd

    elif MODE == 3:  # x - min
        RRout = RR - RRmin

    elif MODE == 4:  # (x - min)/(max-min)
        RRout = (RR - RRmin) / (RRmax - RRmin)

    elif MODE == 5:  # PAR*x/meanCALM
        RRout = (PAR * RR) / RRmeanCALM

    dataSUBJ['IBI'] = RRout
    dataSUBJ.to_csv(os.path.join(OUTDIR, subject), sep='\t')
