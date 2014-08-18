__author__ = 'AleB'

import pandas

import pyHRV

# We load the data series from a csv file with tabulations as separators
dat = pandas.read_csv("../../z_data/D01.txt", sep="\t")["IBI"]
lab = pandas.read_csv("../../z_data/D01.txt", sep="\t")["sit"]
data_series = pyHRV.DataSeries(data=dat, labels=lab)
# and the windows collection with the linear time windows generator
windows = pyHRV.NamedWinGen(data_series)
# The windows mapper will do all the rest of the work, we just put
# there every Time (TD) and Frequency (FD) Domain Index
mapper = pyHRV.WindowsMapper(
    data_series, windows, pyHRV.indexes.TDIndexes.__all__ +
                          pyHRV.indexes.FDIndexes.__all__)  # +
#pyHRV.indexes.NonLinearIndexes.__all__)
mapper.compute_all()
# We convert the results to a data frame
data_frame = pandas.DataFrame(mapper.results)
# to give it an header
data_frame.columns = mapper.labels
# and to save it in a csv file
data_frame.to_csv("results.csv", sep="\t", index=False)
