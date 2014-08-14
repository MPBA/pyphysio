__author__ = 'AleB'

import pandas

import pyHRV


# We load the data series from a csv file with tabulations as separators
data_series = pyHRV.load_ds_from_csv_column("../../z_data/A05.txt")
# and the windows collection with the linear time windows generator
windows = pyHRV.LinearTimeWinGen(10000, 10000, data_series)
# The windows mapper will do all the rest of the work, we just put
# there every Time (TD) and Frequency (FD) Domain Index
mapper = pyHRV.WindowsMapper(
    data_series, windows, pyHRV.indexes.TDIndexes.__all__ +
                          pyHRV.indexes.FDIndexes.__all__ +
                     pyHRV.indexes.NonLinearIndexes.__all__)
mapper.compute_all()
# We convert the results to a data frame
data_frame = pandas.DataFrame(mapper.results)
# to give it an header
data_frame.columns = mapper.labels
# and to save it in a csv file
data_frame.to_csv("results.csv", sep="\t", index=False)
