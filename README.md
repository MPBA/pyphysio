# PyPhysio

PyPhysio is a library of state of art algorithms for the analysis of physiological signals.
It contains the implementations of the most important algorithms for the analysis of physiological data like ECG, BVP, EDA and inertial.
The algorithms are implemented on top of a framework that provides caching to optimize feature extraction pipelines.

### Signals
To allow optimization, two wrapper classes for signals are provided:
- **EvenlySignal**: wraps a signal sampled with a constant frequency, specifying its _values_ (samples) and its _sampling_freq_
- **UnevenlySignal**: wraps a signal where the distance between samples is not constant, specifying the _values_, the original _sampling_freq_ and the _x_values_ that can be (_x_type_) _'instants'_ or _'indices'_ wrt the original signal.

### Classes of Algorithms
Every algorithm is available under the main module name e.g.

import pyphysio as ph
ph.IIRFilter(...)

however they are divided into the following groups:

- **estimators**: from a signal produce a signal of a different nature
- **filters**: from a signal produce a filtered signal of the same nature
- **indicators**: from a signal produce a value
- **segmentation**: from a signal produce a series of segments
- **tools**: from a signal produce arbitrary data

### Example
    import pyphysio as ph
    from tests.context import Assets
     
    sig = ph.EvenlySignal(Assets.bvp(), 1024)
     
    # I have to resample it
    sig = ph.resample(sig, 250, "spline")
     
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
     
    # Let's have a look at what I've done
    sig.plot('b')
    filtered.plot('r')
     
    # Ok I like it
    ibi_est = ph.Estimator.ibi(delta=0.5, fmax=180)
     
    ibis = ibi_est(filtered)
    ibis.plot("|")
     