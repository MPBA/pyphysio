import numpy as np
import pandas as pd
from PyHRV import DataSeries

from PyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


def load_rr_data_series(path, column=Sett.load_rr_column_names, sep=Sett.csv_separator):
    """
    For galaxy use loads an rrs. column from a csv file
    @param path:
    @param column:
    @param sep:
    @return:
    """

    d = pd.read_csv(path, sep)
    if column in d.columns:
        inst = DataSeries(np.array(d[column]))
        assert isinstance(inst, DataSeries)
        return inst
    else:
        raise KeyError("Column %s not found in file %s".format(column, path))


def save_rr_data_series(data_series, path, sep=Sett.csv_separator):
    """
    For galaxy use saves the DataSeries (rr) to a csv file.
    @param data_series:
    @param path:
    @param sep:
    @return:
    """
    assert isinstance(data_series, pd.Series)
    data_series.to_csv(path, sep=sep, header=True)

