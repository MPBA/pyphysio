__author__ = 'AleB'

from PyHRVSettings import PyHRVDefaultSettings as Sett
import pandas as pd
from DataSeries import DataSeries


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
        return DataSeries(d[column])
    else:
        raise KeyError("Colonna %s non presente nel file %s".format(column, path))


def save_rr_data_series(data_series, path, sep=Sett.csv_separator):
    """
    For gaqlaxy use saves the DataSeries (rr) to a csv file.
    @param data_series:
    @param path:
    @param sep:
    @return:
    """
    assert isinstance(data_series, pd.Series)
    data_series.to_csv(path, sep=sep, header=True)

