from __future__ import division
import numpy as np
from pyphysio.tools.Tools import *
import os
import string
import random

from scipy.stats import pearsonr

def code_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#===========================================================================
# Tools for computatino of couppling metrics
# TODO: Accept M-signals in input - matrixes NxM

EXTLIB_DIR = '/home/andrea/Trento/CODICE/workspaces/pyphysio/ext_lib/'
RP_dir = EXTLIB_DIR + 'commandline_recurrence_plot/'
ID_dir = EXTLIB_DIR = 'infodynamics-dist-1.3/'
jarLocation = '/home/andrea/Trento/CODICE/workspaces/pyphysio/ext_lib/infodynamics-dist-1.3/infodynamics.jar'

def compute_PC(x,y):
    r, p = pearsonr(x,y)
    return(r,p)
    

def compute_TE(x,y, history=1, kernelwidth = 0.5):
    import jpype
    print('Joseph T. Lizier, "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", Frontiers in Robotics and AI 1:11, 2014')
    
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    # Create a TE calculator and run it:
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
    teCalc.initialise(history, kernelwidth) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(x), jpype.JArray(jpype.JDouble, 1)(y))
    result = teCalc.computeAverageLocalOfObservations()
    return(result)
    
def compute_MI(x,y, history=1, kernelwidth = 0.5):
    import jpype
    print('Joseph T. Lizier, "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", Frontiers in Robotics and AI 1:11, 2014')
    
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)    
    # Create a MI calculator and run it:
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
    teCalc.initialise() # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(x), jpype.JArray(jpype.JDouble, 1)(y))
    result = teCalc.computeAverageLocalOfObservations()
    return(result)

def compute_CRQA(x, y, embed_m, embed_t, th):
    cwd = os.getcwd()
    os.chdir(RP_dir)
    # TODO: Check whether directories already exist
    random_dir = code_generator(6)
    os.mkdir(random_dir)
    os.chdir(random_dir)
    np.savetxt('x', x)
    np.savetxt('y', y)
    os.chdir(RP_dir)
    os.system('./rp -i '+random_dir+'/x -j '+random_dir+'/y -m '+str(embed_m)+' -t '+str(embed_t)+' -e '+str(th)+' -o '+random_dir+'/outmeas')
    try:
        meas = np.loadtxt(random_dir+'/outmeas')
        os.chdir(random_dir)
        os.remove('x')
        os.remove('y')
        os.remove('outmeas')
        os.chdir(RP_dir)
        os.rmdir(random_dir)
        os.chdir(cwd)
    #    labels = ['#RR', 'DET', 'DET/RR', 'LAM', 'LAM/DET', 'L_max', 'L', 'L_entr', 'DIV', 'V_max', 'TT', 'V_entr', 'T1', 'T2', 'W_max', 'W_mean', 'W_entr', 'W_prob', 'F_min']
        return(meas)
    except:
        print('Unable to compute RP metrics')
        os.chdir(random_dir)
        os.remove('x')
        os.remove('y')
        os.chdir(RP_dir)
        os.rmdir(random_dir)
        os.chdir(cwd)
        return(np.repeat(np.nan, 19))
            

def compute_TDS(x, y, winlen=None, winstep=None, nsamp=0):
    if winlen is None:
        winlen = len(x)
        winstep = len(x)
        
    idxs_cross_start = np.arange(0, len(y) - winstep +1, winstep)
    idx_t0 = int(np.round(winlen-1))
    idx_start = idxs_cross_start[0]

    delays = []
    for idx_start in idxs_cross_start:
        idx_end = idx_start + winlen
        
        # 1. Obtain segments
        x_v = x[idx_start:idx_end]
        y_v = y[idx_start:idx_end]
        
        # 2. Normalize
        x_v_norm = ph.Normalize(norm_method = 'standard')(x_v)
        y_v_norm = ph.Normalize(norm_method = 'standard')(y_v)
        
        # 3. Compute cross-correlation
        cross = np.correlate(x_v_norm, y_v_norm, mode = 'full')
        
        # 4. Find delay where abs(cross) is max
        idx_max = np.argmax(abs(cross))
        idx_max = idx_max -idx_t0
        delays.append(idx_max)
        
    delays = np.array(delays)
    
    # 5. Compute stability 
    stability = np.repeat(np.nan, len(delays))
    # select interval
    if nsamp == 0:
        nsamp = len(delays)
        
    for idx_start in np.arange(len(delays) - nsamp + 1):
        idx_end = idx_start + nsamp
        curr_delays = delays[idx_start:idx_end]
        curr_delays = curr_delays - np.mean(curr_delays)
        energy = np.sqrt(np.sum(curr_delays**2))
        stability[idx_start] = energy
    return(stability, delays)

def compute_MIC(x, y, alpha = 0.6, c=15, all_metrics=False):
    from minepy import MINE
    mine = MINE(alpha, c)
    mine.compute_score(x, y)
    if all_metrics:
        return(mine.mic(), mine)
    else:
        return(mine.mic())

def compute_DTW(x,y, method="Euclidean", step='symmetric2', wtype="none", keep=True, distanceonly=False, openend=False, openbegin=False, wsize=5): # TODO: complete
    # TODO: check that rpy2 is installed
    # TODO: check that dtw is installed
    try:
        import rpy2.robjects.numpy2ri
        from rpy2.robjects.packages import importr

        # Set up our R namespaces
        R = rpy2.robjects.r
        DTW = importr('dtw')
        
    except:
        print('Errors in importing ''dtw'' R package. Check.')
        return({})
    
    # convert step_patterns
    if step == 'symmetric1':
        step = DTW.symmetric1
    elif step == 'symmetric2':
        step = DTW.symmetric2
    elif step == 'asymmetric':
        step = DTW.asymmetric
    elif step == 'symmetricP0':
        step = DTW.symmetricP0
    elif step == 'asymmetricP0':
        step = DTW.asymmetricP0
    elif step == 'symmetricP05':
        step = DTW.symmetricP05
    elif step == 'asymmetricP05':
        step = DTW.asymmetricP05
    elif step == 'symmetricP1':
        step = DTW.symmetricP1
    elif step == 'asymmetricP1':
        step = DTW.asymmetricP1
    elif step == 'symmetricP2':
        step = DTW.symmetricP2
    elif step == 'asymmetricP2':
        step = DTW.asymmetricP2
    elif step == 'typeIa':
        step = DTW.typeIa
    elif step == 'typeIb':
        step = DTW.typeIb
    elif step == 'typeIc':
        step = DTW.typeIc
    elif step == 'typeId':
        step = DTW.typeId
    elif step == 'typeIas':
        step = DTW.typeIas
    elif step == 'typeIbs':
        step = DTW.typeIbs
    elif step == 'typeIcs':
        step = DTW.typeIcs
    elif step == 'typeIds':
        step = DTW.typeIds
    elif step == 'typeIIa':
        step = DTW.typeIIa
    elif step == 'typeIIb':
        step = DTW.typeIIb
    elif step == 'typeIIc':
        step = DTW.typeIIc
    elif step == 'typeIId':
        step = DTW.typeIId
    elif step == 'typeIIIc':
        step = DTW.typeIIIc
    elif step == 'typeIVc':
        step = DTW.typeIVc
    elif step == 'mori2006':
        step = DTW.mori2006
    elif step == 'rigid':
        step = DTW.rigid
    else:
        print('Step pattern not implemented, using default: symmetric2')
        step = DTW.symmetric2
   
    # Calculate the alignment vector and corresponding distance
    x = list(x)
    y = list(y)
    
    alignment = R.dtw(x,y, dist_method=method, step_pattern=step, window_type=wtype, keep_internals = keep, distance_only = distanceonly, open_end=openend, open_begin=openbegin, **{'window.size':wsize})
    dist = alignment.rx('distance')[0][0]
    normDist = alignment.rx('normalizedDistance')[0][0]
    index1 = np.array(alignment.rx('index1')[0])
    index2 = np.array(alignment.rx('index2')[0])
    index1s = np.array(alignment.rx('index1s')[0])
    index2s = np.array(alignment.rx('index2s')[0])
    N = alignment.rx('N')[0][0]
    M = alignment.rx('M')[0][0]
    #    openEnd = alignment.rx('openEnd')[0][0]
    #    openBegin = alignment.rx('openBegin')[0][0]
    #    windowFunction = alignment.rx('windowFunction()')[0]
    jmin = alignment.rx('jmin')[0][0]
    stepsTaken = np.array(alignment.rx('stepsTaken')[0])
    costMatrix = np.array(alignment.rx('costMatrix')[0])
    out = {'dist':dist, 'normDist':normDist, 'costMatrix':costMatrix, 'index1':index1, 'index2':index2, 'index1s':index1s, 'index2s':index2s, 'N':N, 'M':M, 'jmin':jmin, 'stepsTaken':stepsTaken}
    return(dist, out)