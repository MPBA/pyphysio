# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:01:23 2016

@author: andrea
"""

from __future__ import division
import numpy as np

from context import ph

FSAMP = 10
TSTART = 10
TYPE = ''

data = np.arange(1000)

s = ph.EvenlySignal(data, FSAMP, TYPE, TSTART)

###############
## TEST GETS ATTRIBUTES
###############

s.get_duration()  # TODO: 1/fsamp in piu' /// a me risulta 100.0 che è 1000 samples a 10 Hz, non è giusto?
s.get_indices()  # OK
s.get_times()  ## TODO: non tiene conto dello start time /// esatto perché sono salvati così, serve un bel po di tempo e/o memoria in più per averli con lo start_time.
## TODO: just one non serve /// serve a me internamente e non ci sonon motivi per nasconderlo
s.get_values()  # OK
s.get_signal_nature()  # OK
s.get_sampling_freq()  # OK
s.get_start_time()  # OK
s.get_end_time()  # TODO: ritorna un dt=1/fsamp in piu' /// vedi duration
s.get_metadata()

s.plot()

# TEST SLICING
s_ = s[10:23]
s_.get_duration()  # OK
s_.get_indices()  # OK
s_.get_times()  # OK
s_.get_values()  # OK
s_.get_signal_nature()  # OK
s_.get_sampling_freq()  # OK
s_.get_start_time()  # TODO: dovrebbe dare lo start della slice /// (issue2)
s_.get_end_time()  # TODO: non corrisponde /// (issue2)
s_.get_metadata()
s_.plot()

###############
## TEST RESAMPLE
###############
s_down = s.resample(5)
s_down.get_duration()  # OK
s_down.get_indices()  # OK
s_down.get_times()  # OK
s_down.get_values()  # OK
s_down.get_signal_nature()  # OK
s_down.get_sampling_freq()  # OK
s_down.get_start_time()  # OK
s_down.get_end_time()  # TODO: ritorna un dt=1/fsamp in piu' /// (vedi get_duration)
s_down.get_metadata()
s_down.plot()

s_up = s.resample(15)
s_up.get_duration()  # OK
s_up.get_indices()  # OK
s_up.get_times()  # OK
s_up.get_values()  # OK
s_up.get_signal_nature()  # OK
s_up.get_sampling_freq()  # OK
s_up.get_start_time()  # OK
s_up.get_end_time()  # TODO: ritorna un dt=1/fsamp in piu' /// (vedi get_duration)
s_up.get_metadata()
s_up.plot()
