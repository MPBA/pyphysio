#The latest version of the library can be found at <https://gitlab.com/a.bizzego/pyphysio>
##With many new features, such as: support for multi-channel/multi-component data with parallelization (e.g. EEG, fNIRS), novel signal processing algorithms, signal quality indicators, etc.

**Please note that this repository is not actively maintained anymore.**


---

# pyphysio

pyphysio is a library of state of art algorithms for the analysis of physiological signals.
It contains the implementations of the most important algorithms for the analysis of physiological data like ECG, BVP, EDA, inertial, and fNIRS.


### Signals
To allow optimization, two wrapper classes for signals are provided:
- **EvenlySignal**: wraps a signal sampled with a constant frequency, specifying its _values_ (samples) and its _sampling_freq_
- **UnevenlySignal**: wraps a signal where the distance between samples is not constant, specifying the _values_, the original _sampling_freq_ and the _x_values_ that can be (_x_type_) _'instants'_ or _'indices'_ wrt the original signal.

### Classes of Algorithms
Every algorithm is available under the main module name e.g.

    import pyphysio as ph
    ph.IIRFilter(...)

however they are divided into the following groups:

- **estimators**: from a signal produce a signal of a different type
- **filters**: from a signal produce a filtered signal of the same type
- **indicators**: from a signal produce a value
- **segmentation**: from a signal produce a series of segments
- **tools**: from a signal produce arbitrary data

### Examples
Examples on how to use pyphysio can be found at:

<https://github.com/MPBA/pyphysio/tree/master/tutorials>
     
### Reference
If you use `pyphysio` for research, please cite us:

"Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', *SoftwareX*"
<https://www.sciencedirect.com/science/article/pii/S2352711019301839>
