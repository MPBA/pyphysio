__author__ = 'AleB'


from PyHRVSettings import PyHRVDefaultSettings as Sett
import pandas as pd
from DataSeries import DataSeries


def load_rr_data_series(path, column=Sett.load_rr_column_names, sep=Sett.csv_separator, *args):
    d = pd.read_csv(path, sep, *args)
    if column in d.columns:
        return DataSeries(d[column])
    else:
        raise KeyError("Colonna %s non presente nel file %s".format(column, path))


def save_rr_data_series(data_series, path, *args):
    assert isinstance(data_series, DataSeries)
    # TODO: check "args"
    data_series.to_csv(path, args)

