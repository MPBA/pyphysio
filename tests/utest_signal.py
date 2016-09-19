# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:01:23 2016

@author: andrea
"""

from __future__ import division
import numpy as np
import pyPhysio as ph

FSAMP = 10
TSTART = 10
TYPE = ''

data = np.arange(1000)

s = ph.EvenlySignal(data, FSAMP, TYPE, TSTART)

print(s)

s.get_duration() # OK

s.get_indices() # OK

s.get_times() ## TODO: non tiene conto dello start time
## TODO: just one non serve

s.get_values() # OK

s.get_signal_nature() # OK

s.get_sampling_freq() # OK

s.get_start_time() # OK

s.get_end_time() # TODO: ritorna un dt=1/fsamp in piu' 

s.get_metadata()

s.plot() # TODO: dovrebbe mettere sulle x il tempo

s_ = s.getslice(10, 23) # TODO: non funziona, serve?

# TEST SLICING
s_ = s[10:23]

s_.get_duration() # OK

s_.get_indices() # OK

s_.get_times() # OK

s_.get_values() # OK

s_.get_signal_nature() # OK

s_.get_sampling_freq() # OK

s_.get_start_time() # TODO: dovrebbe dare lo start della slice

s_.get_end_time() # TODO: non corrisponde

s_.get_metadata()

s_.plot()