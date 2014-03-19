# pipeline HRV analysis Esposito
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

# window lenght
parser.add_option("-s", "--status",
                  action="store", type="str", default='MCOT,LF', 
                  dest="status", help="Stati to remove (comma separated)")

(options, args) = parser.parse_args()
INPUTDIR = options.input_dir
OUTDIR = options.output_dir

STATUS = options.status.split(',')

print "Normalize RR"

subjects = os.listdir(INPUTDIR)
tot = len(subjects)
i = 0

for subject in subjects:  # Each file
    i += 1
    rows = []
    FILENAME = os.path.join(INPUTDIR, subject)
    print "Processing " + subject + " (" + str(i) + "/" + str(tot) + ").."
    # Initialize data file    
    # Load subject data
    dataSUBJ = pd.read_csv(FILENAME, sep='\t', quotechar='"')
    sit = np.array(dataSUBJ['sit'], dtype='str')

    rowOK = []
    session = 1
    for k in xrange(len(sit)):
        currentStatus = sit[k]
        remove = False
        for stat in STATUS:
            if currentStatus == stat:
                remove = True
        if remove:
            lista_pd = dataSUBJ[rowOK, :]
            lista_pd.to_csv(os.path.join(OUTDIR, subject+'_'+str(session)+'.txt'), sep='\t')
            rowOK = []
            session += 1
