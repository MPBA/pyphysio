__author__ = 'AleB'
from pyHRV.Files import *

## Loading IBI from a CSV file
# This is the simplest way to load data.
# Complete usage
# The column name and the CSV separator can be specified if they are different from the default ones.
data2 = load_ds_from_csv_column(path="my_ibi_data.csv", column="Intervals", sep="\t")
# Minimal usage
data1 = load_ds_from_csv_column("my_ibi_data.csv")


## Loading IBI from an ECG CSV file
# This function will automatically convert the ECG data to IBI data,
# the delta parameter can be specified or left to the default value.
# Complete usage
data3 = load_rr_from_ecg(path="my_ecg_data.csv", delta=2, ecg_col="Value", ecg_time_col="Time", sep=";")
# Minimal usage
data4 = load_rr_from_ecg("my_ecg_data.csv")


## Loading IBI from a BVP CSV file
# This function will automatically convert the BVP data to IBI data,
# the delta parameter can be specified or left to the default value.
# Complete usage
data5 = load_rr_from_bvp(path="my_bvp_data.csv", delta_ratio=47, bvp_col="Value", bvp_time_col="Time", sep=";")
# Minimal usage
data6 = load_rr_from_bvp("my_bvp_data.csv")


## Loading a Windows Set from a CSV file
# This function will return an instance of a CollectionWinGen that is a WindowsGenerator, an iterable object that yields
# Windows.
# Complete usage
data7 = load_windows_gen_from_csv(path="my_saved_windows.csv", column_begin="BeginSample", column_end="LastSample",
                                  sep=",")
# Minimal usage
data8 = load_windows_gen_from_csv("my_saved_windows.csv")
