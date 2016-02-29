# coding=utf-8
__author__ = 'AleB'

# Libraries are imported
# bvp_np is an ndarray with the data in polling format
# bvp_sf is the sampling frequency of the data
# bvp_st is the start timestamp of the series
import pyphysio.pyPhysio as ph

sig = ph.EvenlySignal(bvp_np, bvp_sf, bvp_st)
sig.metadata.set_anag("Tito", "Livio", "SUB0524", 42)

# I have to resample it
sig = ph.resample(sig, 1024, "spline")
# > Ok, done in 254ms

# Need to remove what I don't need
# Let's setup a filter
fil_bp = ph.Filters.frequency(low=50, hi=80, attenuation=40)
# Oh! Not 40, 50!
fil_bp.set(attenuatino=50)
# > Warning: the parameter 'attenuatino' is not used by the algorithm!

# Ops, typo!
fil_bp.set(attenuation=50)

# Let's filter
filtered = fil_bp(sig)
# > Ok, done in 656ms

# Let's have a look at what I've done
sig.plot('this is the original')
filtered.plot('this is filtered 50Hz-80Hz att. 50')

# Ok I like it
ibi_est = ph.Estimator.ibi(delta=0.5, fmax=180)

ibis = ibi_est(filtered)
# > Ok, done in 112ms
