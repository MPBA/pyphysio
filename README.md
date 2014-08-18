PyHRV
=====
*Requires pandas 0.13.1 or greater*

PyHRV is a Python based library for the on-line and off-line HRV analysis designed to be easy to use by coding in Python and also through Galaxy. It allows to compute the classical time and frequency domain indexes and also a set of non-linear indexes like Poincaré plot and entropy analysis indexes. It has an integrated windowing system that allows to easily work with temporal segments of data and to compute the indexes on them automatically. The algorithms are efficient and up-to-date, the library provides several optimizations both in the on-line mode and in the off-line one.

Off-Line mode
-------------
In the off-line mode indexes are calculated on the collected data after the experiment, every index is available in this mode.

Files
-----
The data can be loaded manually with an arbitrary library or through the utility package Files.py that contains functions for the file-system interaction. It contains also basic methods to load and convert data from ECG or BVP to the RR IBI format and methods to filter this kind of data.

Data
----
The main data structure for this mode is provided by the DataSeries class. DataSeries inherits from pandas.Series, a class of the practical Python library Pandas. Its version must be 0.13 or higher due to a strange inheritance problem of the previous version.

Windowing
---------
The library allows to compute the indexes on slices of the entire dataset. The library provides some classes to analyze the data and extract windows from there, for example if there are labels on the IBIs the NamedWinGen creates an iterable object that yields the windows (begin-end indexes pairs) as the labels change, and the WindowsMapper class that takes indexes and windows information and puts the results in a pandas.DataFrame.

Optimizations
-------------
As sometimes the elaboration of some indexes requires a former computation of other things on the same data, like the differences between consecutives values or for example the spectrum estimation for every frequency domain index, the DataSeries class provides an internal cache system to avoid the duplication of congruent computations on the same data.

The Cache.py package provides some classes for the generalized computation of these cached values. The average speed up on an about 8000 IBIs DataSeries between the cached and the non cached mode is about of the 60%.
