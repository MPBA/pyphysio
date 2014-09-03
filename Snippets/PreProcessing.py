__author__ = 'AleB'
from pyHRV.Files import *
from pyHRV.Filters import *
from pyHRV import DataSeries

data = DataSeries()

## Normalization
# 5 types of normalization are available
# Normalizes the series removing the mean (RR-mean)
data = IBIFilters.normalize_mean(data)
# Normalizes the series removing the mean and dividing by the standard deviation (RR-mean)/sd
data = IBIFilters.normalize_mean_sd(data)
# Normalizes the series removing the minimum value (RR-min)
data = IBIFilters.normalize_min(data)
# Normalizes the series removing the mean and dividing by the range width (RR-mean)/(max-min)
data = IBIFilters.normalize_max_min(data)
# Normalizes the series scaling by two factors ((PAR*RR)/meanCALM)
# Complete usage
data = IBIFilters.normalize_custom(series=data, par1=410, par2=132)

## Outliers filtering
# Removes outliers from RR series.
# Complete usage
data = IBIFilters.filter_outliers(data, last=14, min_bpm=25, max_bpm=200, win_length=50)

## Integrated way
# These filters can be applied after the loading phase specifying them in the filters= parameter as
# a list of names e.g.
data1 = load_ibi_from_bvp("my_bvp_data.csv", filters=[IBIFilters.normalize_mean_sd, IBIFilters.filter_outliers])
