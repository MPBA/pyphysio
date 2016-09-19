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

data = np.array([45, 90, 120])
indices = np.array([5, 10, 14])

# TODO: decidere se indices sono tempi o indici di array
s = ph.UnevenlySignal(data, indices, FSAMP, 15, TYPE, TSTART)
# TODO: original length se non data e' il massimo indice + 1
# TODO: original sampling frequency se non data e' 1
# TODO: check sempre
# TODO: check che indices siano interi
# TODO: mettere asevents = True/False per usare timestamps invece che indici (da discutere)


###############
## TEST GETS ATTRIBUTES
###############

s.get_duration() # OK

s.get_indices() # OK

s.get_times() ## TODO: non funziona, dipende da decisioni prima
## TODO: just one non serve

s.get_values() # OK

s.get_signal_nature() # OK

s.get_sampling_freq() # OK

s.get_start_time() # TODO: vedi decisioni prima

s.get_end_time() # TODO: vedi decisioni prima

s.get_metadata()

s.plot() # TODO: dovrebbe mettere sulle x il tempo

# TEST SLICING
s_ = s[:2]

s_.get_duration() # TODO: risultato errato

s_.get_indices() # OK

s_.get_times() # Vedi sopra

s_.get_values() # OK

s_.get_signal_nature() # OK

s_.get_sampling_freq() # OK

s_.get_start_time() # TODO: dovrebbe dare lo start della slice

s_.get_end_time() # TODO: non corrisponde

s_.get_metadata()

s_.plot() #OK

###############
## TEST to_evenly
###############
s_evenly = s.to_evenly() # TODO: remove prints
s_evenly.get_duration() # 1/fsamp in piu'
s_evenly.get_indices() # OK
s_evenly.get_times() # vedi sopra e non tiene conto dello start time
s_evenly.get_values() # OK
s_evenly.get_signal_nature() # OK
s_evenly.get_sampling_freq() # OK
s_evenly.get_start_time() # OK
s_evenly.get_end_time() # TODO: ritorna un dt=1/fsamp in piu' 
s_evenly.get_metadata()
s_evenly.plot() # TODO: dovrebbe mettere sulle x il tempo