__author__ = 'AleB'
from pyHRV.Files import *
from pyHRV import DataSeries, data_series_from_bvp, data_series_from_ecg

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ecg_val = [1, 2, 3, 4, 5, 4, 3, 2, 1]
ecg_time = [0, 1, 2, 3, 4, 5, 6, 7, 8]
bvp_val = [1, 2, 3, 4, 5, 4, 3, 2, 1]
bvp_time = [0, 1, 2, 3, 4, 5, 6, 7, 8]

## Generating a DataSeries: Constructor
# Passing an iterable object (list, array, ndarray etc)
data_series = DataSeries(data)
## Generating a DataSeries: Aux functions
# From bvp or ecg value-time example_data through the internal peak detection algorithm
data_series_ecg = data_series_from_ecg(ecg_val, ecg_time)
data_series_bvp = data_series_from_bvp(bvp_val, bvp_time)

## Loading IBI from a CSV file
# This is the simplest way to load example_data.
# Complete usage
# The column name and the CSV separator can be specified if they are different from the default ones.
# Minimal usage
data1 = load_ds_from_csv_column("my_ibi_data.csv")
data2 = load_ds_from_csv_column(path="my_ibi_data.csv", column="Intervals", sep="\t")


## Loading IBI from an ECG CSV file
# This function will automatically convert the ECG example_data to IBI example_data,
# the delta parameter can be specified or left to the default value.
# Minimal usage
data4 = load_ibi_from_ecg("my_ecg_data.csv")
# Complete usage
data3 = load_ibi_from_ecg(path="my_ecg_data.csv", delta=2, ecg_col="Value", ecg_time_col="Time", sep=";")


## Loading IBI from a BVP CSV file
# This function will automatically convert the BVP example_data to IBI example_data,
# the delta parameter can be specified or left to the default value.
# Minimal usage
data6 = load_ibi_from_bvp("my_bvp_data.csv")
# Complete usage
data5 = load_ibi_from_bvp(path="my_bvp_data.csv", delta_ratio=47, bvp_col="Value", bvp_time_col="Time", sep=";")


## Loading a Windows Set from a CSV file
# This function will return an instance of a CollectionWinGen that is a WindowsGenerator, an iterable object that yields
# Windows.
# Minimal usage
data8 = load_windows_gen_from_csv("my_saved_windows.csv")
# Complete usage
data7 = load_windows_gen_from_csv(path="my_saved_windows.csv", column_begin="BeginSample", column_end="LastSample",
                                  sep=",")
