__author__ = 'AleB'

import pandas

import pyHRV

# We load the data series from a csv file with tabulations as separators
data_series = pyHRV.load_ds_from_csv_column("../../z_data/A05.txt")
# and the windows collection with the linear time windows generator
ws = pyHRV.LinearTimeWinGen(10000, 10000, data_series)
# The windows mapper will do all the rest of the work, we just put
# there every Time (TD) and Frequency (FD) Domain Index
mm = pyHRV.WindowsMapper(
    data_series, ws, pyHRV.indexes.TDIndexes.__all__ +
                     pyHRV.indexes.FDIndexes.__all__ +
                     pyHRV.indexes.NonLinearIndexes.__all__)
mm.compute_all()
# We convert the results to a data frame
df = pandas.DataFrame(mm.results)
# to give it an header
df.columns = mm.labels
# and to save it in a csv file
df.to_csv("results.csv", sep="\t", index=False)
