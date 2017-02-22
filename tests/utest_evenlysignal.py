# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:01:23 2016

@author: andrea
"""

from __future__ import division
import numpy as np

import pyphysio as ph

FSAMP = 10
TSTART = 5
TYPE = ''

data = np.arange(1000)
s = ph.EvenlySignal(data, FSAMP, TYPE, TSTART)

###############
## TEST GETS ATTRIBUTES
###############

assert len(s) == 1000, "Wrong length"
assert s.get_duration() == 99.9, "Duration not correct" # OK
assert s.get_times()[0] ==5, "Error in get_times() / start"  #OK
assert s.get_times()[-1] == 104.9, "Error in get_times() / end"  #OK
assert len(s.get_times()) == len(data), "Error in get_times() / length"  #OK

assert s.get_values()[0] == 0, "Error in get_values() / start"  #OK
assert s.get_values()[-1] == 999, "Error in get_values() / end"  #OK
assert len(s.get_values()) == len(data), "Error in get_values() / length"  #OK

assert s.get_signal_nature() == '', "Error in get_signal_nature" # OK
assert s.get_sampling_freq() == 10, "Error in get_sampling_freq" # OK
assert s.get_start_time() == 5
assert s.get_end_time() == 104.9

# TEST SEGMENTATION
print('Testing time segmentation')
s_ = s.segment_time(7.45, 10)
assert len(s_) == 25, "Wrong length"
assert s_.get_times()[0] == s_.get_start_time(), "Error in get_times() / start"  #OK
assert s_.get_times()[-1] == s_.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_.get_times()) == len(s_), "Error in get_times() / length"  #OK

assert s_.get_values()[0] == 25, "Error in get_values() / start"  #OK
assert s_.get_values()[-1] == 49, "Error in get_values() / end"  #OK

assert s_.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_.get_sampling_freq() == s.get_sampling_freq(), "Error in get_sampling_freq" # OK


# TEST SEGMENTATION
print('Testing index segmentation')
s_ = s.segment_idx(200, 300)
assert len(s_) == 100, "Wrong length"
assert s_.get_times()[0] == s_.get_start_time(), "Error in get_times() / start"  #OK
assert s_.get_times()[-1] == s_.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_.get_times()) == len(s_), "Error in get_times() / length"  #OK

assert s_.get_values()[0] == 200, "Error in get_values() / start"  #OK
assert s_.get_values()[-1] == 299, "Error in get_values() / end"  #OK

assert s_.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_.get_sampling_freq() == s.get_sampling_freq(), "Error in get_sampling_freq" # OK

###############
## TEST RESAMPLE
###############
print('testing downsampling')
s_down = s.resample(5)
assert len(s_down) == 500, "Wrong length"
assert s_down.get_times()[0] == s_down.get_start_time(), "Error in get_times() / start"  #OK
assert s_down.get_times()[-1] == s_down.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_down.get_times()) == len(s_down), "Error in get_times() / length"  #OK

assert s_down.get_values()[0] == 0, "Error in get_values() / start"  #OK
assert s_down.get_values()[-1] == 998, "Error in get_values() / end"  #OK
assert len(s_down.get_values()) == len(s_down), "Error in get_values() / length"  #OK

assert s_down.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_down.get_sampling_freq() == 5, "Error in get_sampling_freq" # OK

print('testing oversampling')

s_up = s.resample(15)
assert len(s_up) == 1500, "Wrong length"
assert s_up.get_times()[0] == s_up.get_start_time(), "Error in get_times() / start"  #OK
assert s_up.get_times()[-1] == s_up.get_end_time(), "Error in get_times() / end"  #OK
assert len(s_up.get_times()) == len(s_up), "Error in get_times() / length"  #OK

assert s_up.get_values()[0] == 0, "Error in get_values() / start"  #OK
assert s_up.get_values()[-1] == 999, "Error in get_values() / end"  #OK
assert len(s_up.get_values()) == len(s_up), "Error in get_values() / length"  #OK

assert s_up.get_signal_nature() == s.get_signal_nature(), "Error in get_signal_nature" # OK
assert s_up.get_sampling_freq() == 15, "Error in get_sampling_freq" # OK
