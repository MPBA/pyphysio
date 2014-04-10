import numpy as np

import pandas as pd

from pyHRV.DataSeries import DataSeries
from pyHRV.utility import peak_detection
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett


__all__ = ['load_excel_column', 'load_data_series', 'load_rr', 'save_data_series', 'load_rr_from_bvp',
           'load_rr_from_ecg']


def load_excel_column(path, column, columnb, sheetname=0):
    return pd.read_excel(path, sheetname)[column], pd.read_excel(path, sheetname)[columnb]


def load_data_series(path, column, sep=Sett.Files.csv_separator):
    """For galaxy use loads a column from a csv file
    """

    d = pd.read_csv(path, sep)
    if column in d.columns:
        inst = DataSeries(np.array(d[column]))
        inst.name = column
        assert isinstance(inst, pd.Series)
        return inst
    else:
        raise KeyError("COLUMN_NAME: Column %s not found in file %s".format(column, path))


def save_data_series(data_series, path, sep=Sett.Files.csv_separator, header=True):
    """
    For galaxy use saves the DataSeries (rr) to a csv file.
    """
    assert isinstance(data_series, pd.Series)
    data_series.to_csv(path, sep=sep, header=header)


def load_rr(path, column=Sett.Files.load_rr_column_name, sep=Sett.Files.csv_separator):
    """For galaxy use loads an rrs column from a csv file
    """
    return load_data_series(path, column, sep)


def load_rr_from_ecg(path, delta=Sett.DataImports.ecg_delta, sep=Sett.Files.csv_separator, *args):
    df = pd.read_csv(path, sep=sep, *args)
    max_tab, min_tab = peak_detection(df[Sett.Files.load_ecg_column_name], delta,
                                      df[Sett.Files.load_ecg_time_column_name])
    s = DataSeries(np.diff(max_tab))
    for f in Sett.DataImports.import_ecg_filters:
        s = f(s)
    s.meta_tag['from_type'] = "csv_ecg"
    s.meta_tag['from_delta'] = delta
    s.meta_tag['from_filters'] = list(Sett.DataImports.import_ecg_filters)
    return s


def load_rr_from_bvp(path, delta=Sett.DataImports.bvp_delta, sep=Sett.Files.csv_separator, *args):
    df = pd.read_csv(path, sep=sep, *args)
    max_tab, min_tab = peak_detection(df[Sett.Files.load_bvp_column_name], delta,
                                      df[Sett.Files.load_bvp_time_column_name])
    s = DataSeries(np.diff(max_tab))
    for f in Sett.DataImports.import_bvp_filters:
        s = f(s)
    s.meta_tag['from_type'] = "csv_bvp"
    s.meta_tag['from_delta'] = delta
    s.meta_tag['from_filters'] = list(Sett.DataImports.import_bvp_filters)
    return s
