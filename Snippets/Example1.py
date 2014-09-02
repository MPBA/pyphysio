__author__ = 'AleB'

import pyHRV
import pandas

# We use the included example data
from pyHRV.example_data import Test1
# and the windows collection with the linear time windows generator with windows of 40s every 20s.
# This windows generator uses only IBI data from the data series.
windows = pyHRV.LinearTimeWinGen(step=20000, width=40000, data=Test1.data_series)
# The windows mapper will do all the rest of the work, we just need to put
# there every Time (TD) and Frequency (FD) Domain and every Non Linear Index
mapper = pyHRV.WindowsMapper(
    Test1.data_series,
    windows,
    pyHRV.indexes.TDIndexes.__all__ +
    pyHRV.indexes.FDIndexes.__all__ +
    pyHRV.indexes.NonLinearIndexes.__all__)
mapper.compute_all()
# We convert the results to a example_data frame
data_frame = pandas.DataFrame(mapper.results)
# to give it an header
data_frame.columns = mapper.labels
# and to save it in a csv file, without the line number (index)
data_frame.to_csv("example1_results.csv", sep="\t", index=False)
# This results can be compared to the ones in Test1.results
