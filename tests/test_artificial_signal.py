# coding=utf-8
from __future__ import division
import numpy as np
import pyPhysio as ph

FSAMP = 10
TSTART = 10
TYPE = ''

data = np.arange(1000)

s = ph.EvenlySignal(data, FSAMP, TYPE, TSTART)

assert(s.get_duration() >= 99.9)

assert(s.get_times()[0] == TSTART)
assert(s.get_times()[-1] == TSTART+s.get_duration())

s.get_values() # OK

s.get_signal_nature() # OK

s.get_sampling_freq() # OK

s.get_start_time() # OK

s.get_end_time() # TODO: ritorna un dt=1/fsamp in piu'  /// see utest_evenlysignal.py

s.get_metadata()

s.plot() # TODO: dovrebbe mettere sulle x il tempo /// see utest_evenlysignal.py

# TEST SLICING
s_ = s[10:23]

s_.get_duration() # OK

s_.get_indices() # OK

s_.get_times() # OK

s_.get_values() # OK

s_.get_signal_nature() # OK

s_.get_sampling_freq() # OK

s_.get_start_time() # TODO: dovrebbe dare lo start della slice /// see utest_evenlysignal.py

s_.get_end_time() # TODO: non corrisponde /// see utest_evenlysignal.py

s_.get_metadata()

s_.plot()