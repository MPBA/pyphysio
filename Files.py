__author__ = 'AleB'


from PyHRVSettings import PyHRVDefaultSettings as Sett
import pandas as pd
from DataSeries import DataSeries


def load_rr_data_series_from_csv(path, column=Sett.rr_column_names, sep=Sett.csv_separator, *args):
    d = pd.read_csv(path, sep, *args)
    if column in d.columns:
        return DataSeries(d[column])
    else:
        raise KeyError("Colonna %s non presente nel file %s".format(column, path))


def save_data_series(data_series, path, *args):
    assert isinstance(data_series, DataSeries)
    data_series.to_csv(path, args)