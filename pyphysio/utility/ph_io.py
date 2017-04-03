from __future__ import division
import pyphysio as ph
import pickle
import gzip

def to_pickle(signal, filename):
    f = gzip.open(gzip.open(filename, mode="wb"))
    pickle.dump(signal.p, f)
    f.close()
    print('saved in: '+os.getcwd())

def from_pickle(filename):
    f = gzip.open(filename, 'rb')
    sig, sig_ = pickle.load(f)
    sig._pyphysio = sig_
    f.close()
    return(sig)
