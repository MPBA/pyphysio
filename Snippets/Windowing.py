__author__ = 'AleB'
from pyHRV.windowing import *
from pyHRV import Cache

data = Cache()

## Linear Windows Generator
# Generates a linear-index set of windows (b+i*s, b+i*s+w).
wins1 = LinearWinGen(begin=0, step=20, width=40, data=data, end=100)
wins2 = LinearWinGen(0, 20, 40)

## Linear Time Windows Generator
# Generates a linearly-timed set of Time windows (b+i*s, b+i*s+w).
# The arguments are time values
wins3 = LinearTimeWinGen(begin=0, step=20, width=40, data=data, end=100)
wins4 = LinearTimeWinGen(0, 20, 40, data)

## Collection Windows Generator
# Wraps a windows collections, creates a windows generator with that windows
# Here the example_data is needed due to the time-offsets determination.
w = [Window(1, 2), Window(2, 5)]
wins5 = CollectionWinGen(win_list=w, data=data)
wins6 = CollectionWinGen(w, data)
# or faster (see Snippets/LoadingData.py)
from pyHRV.Files import load_windows_gen_from_csv

wins7 = load_windows_gen_from_csv("my_saved_windows.csv")

## Named Windows Generator
# Generates a set of windows reading a labels array.
# Each window begins and ends where the label changes.
# If the include_baseline_names is set to a value, the change from a name to that value is not considered
# e.g.
labels = ["Red", "Red", "Red", "Idle", "Idle", "Idle", "Blue", "Blue", "Blue", "Idle", "Idle"]
# with include_baseline_names="Idle" is computed as two windows:  [0:5:"Red", 6:10:"Blue"]
# with include_baseline_names=None   is computed as four windows: [0:2:"Red", 3:5:"Idle", 6:10:"Blue", 9:10:"Idle"]
wins8 = NamedWinGen(data, labels)
