__author__ = 'AleB'
from pyHRV.Files import *
from pyHRV.Filters import *
from pyHRV import Cache

data = Cache()

## Normalization
# 5 types of normalization are available
# Normalizes the series removing the mean (RR-mean)
data = Filters.normalize_mean(data)
# Normalizes the series removing the mean and dividing by the standard deviation (RR-mean)/sd
data = Filters.normalize_mean_sd(data)
# Normalizes the series removing the minimum value (RR-min)
data = Filters.normalize_min(data)
# Normalizes the series removing the mean and dividing by the range width (RR-mean)/(max-min)
data = Filters.normalize_max_min(data)
# Normalizes the series scaling by two factors ((PAR*RR)/meanCALM)
# Complete usage
data = Filters.normalize_custom(series=data, par1=410, par2=132)

## Outliers filtering
# Removes outliers from RR series.
# Complete usage
data = Filters.filter_outliers(data, last=14, min_bpm=25, max_bpm=200, win_length=50)

## Integrated way
# These filters can be applied after the loading phase specifying them in the filters= parameter as
# a list of names e.g.
data1 = load_ibi_from_bvp("my_bvp_data.csv", filters=[Filters.normalize_mean_sd, Filters.filter_outliers])
