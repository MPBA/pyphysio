from __future__ import division
import numpy as np
import pyphysio as ph

FSAMP = 10
TSTART = 10
TYPE = ''

data = np.array([45, 90, 120])
indices = np.array([5, 10, 14])

# TODO: decidere se indices sono tempi o indici di array
s = ph.UnevenlySignal(data, FSAMP, TYPE, TSTART, indices = indices)
# TODO: original length se non data e' il massimo indice + 1
# TODO: original sampling frequency se non data e' 1
# TODO: check sempre
# TODO: check che indices siano interi
# TODO: mettere asevents = True/False per usare timestamps invece che indici (da discutere)


###############
## TEST GETS ATTRIBUTES
###############
assert len(s) == 3, "Wrong length"
assert s.get_times()[0] == 10.5, "Error in get_times() / start"  #OK
assert s.get_times()[-1] == 11.4, "Error in get_times() / end"  #OK
assert len(s.get_times()) == len(data), "Error in get_times() / length"  #OK

assert s.get_values()[0] == 45, "Error in get_values() / start"  #OK
assert s.get_values()[-1] == 120, "Error in get_values() / end"  #OK
assert len(s.get_values()) == len(data), "Error in get_values() / length"  #OK

assert s.get_signal_nature() == '', "Error in get_signal_nature" # OK
assert s.get_sampling_freq() == 10, "Error in get_sampling_freq" # OK
assert s.get_start_time() == 10
assert s.get_end_time() == 11.4

# TEST SEGMENTATION
print('Testing time segmentation')
s_ = s.segment_time(10.5, 11)
assert len(s_) == 1, "Wrong length"
assert s_.get_duration() == 0, "Duration not correct" # OK
assert s_.get_times()[0] == s_.get_start_time(), "Error in get_times() / start"  #OK
assert s_.get_times()[-1] == s_.get_end_time(), "Error in get_times() / end"  #OK
assert s_.get_times()[-1] == s_.get_times()[0]

assert len(s_.get_times()) == len(s_), "Error in get_times() / length"  #OK

assert s_.get_values()[0] == 45, "Error in get_values() / start"  #OK
assert s_.get_values()[-1] == 45, "Error in get_values() / end"  #OK

assert s_.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_.get_sampling_freq() == s.get_sampling_freq(), "Error in get_sampling_freq" # OK

print('Testing index segmentation')
s_ = s.segment_idx(8,20)
assert len(s_) == 2, "Wrong length"
assert s_.get_times()[0] == s_.get_start_time(), "Error in get_times() / start"  #OK
assert s_.get_times()[-1] == s_.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_.get_times()) == len(s_), "Error in get_times() / length"  #OK

assert s_.get_values()[0] == 90, "Error in get_values() / start"  #OK
assert s_.get_values()[-1] == 120, "Error in get_values() / end"  #OK

assert s_.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_.get_sampling_freq() == s.get_sampling_freq(), "Error in get_sampling_freq" # OK

###############
## TEST TO EVENLY
###############
print('testing to evenly')

s_evenly = s.to_evenly(kind = 'linear') # TODO: remove prints
assert len(s_evenly) == 10, "Wrong length"
assert s_evenly.get_times()[0] == s_evenly.get_start_time(), "Error in get_times() / start"  #OK
assert s_evenly.get_times()[-1] == s_evenly.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_evenly.get_times()) == len(s_evenly), "Error in get_times() / length"  #OK

assert s_evenly.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_evenly.get_sampling_freq() == FSAMP, "Error in get_sampling_freq" # OK