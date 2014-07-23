##ck2
__all__ = ['load_pd_from_excel_column', 'load_ds_from_csv_column', 'load_ds_from_csv_column', 'save_ds_to_csv',
           'load_rr_from_bvp',
           'load_rr_from_ecg']

import numpy as np

import pandas as pd

from pyHRV.DataSeries import DataSeries
from pyHRV.Utility import peak_detection
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Sett
from pyHRV.windowing.WindowsBase import Window
from pyHRV.windowing.WindowsGenerators import CollectionWinGen


def load_pd_from_excel_column(path, column, column_b=None, sheet_name=0):
    """
    Loads one or two columns as pandas.Series from an excel format file.
    @param path: path of the file to read or file-like object
    @param column: first column to load
    @type column: int
    @param column_b: second optional column to load
    @type column_b: int
    @param sheet_name: excel sheet ordinal position or name
    @type sheet_name: int or str or unicode
    @return: One or two loaded columns
    @rtype: (pandas.Series) or (pandas.Series, pandas.Series)
    """
    if column_b is None:
        a = pd.read_excel(path, sheet_name)
        return a[column] if isinstance(column, basestring) else a[a.columns[column]]
    else:
        a = pd.read_excel(path, sheet_name)
        b = pd.read_excel(path, sheet_name)
        a = a[column] if isinstance(column, basestring) else a[a.columns[column]]
        b = b[column_b] if isinstance(column_b, basestring) else b[b.columns[column_b]]
        return a, b


def load_ds_from_csv_column(path, column=Sett.load_rr_column_name, sep=Sett.load_csv_separator):
    """
    Loads a column from a csv file.
    @param path: path of the file to read
    @type path: str or unicode
    @param column: name of the column to load
    @type column: str or unicode
    @return: DataSeries read
    @rtype: DataSeries
    """
    d = pd.read_csv(path, sep)
    if not column in d.columns:
        column = d.columns[0]
    inst = DataSeries(d[column])
    inst.name = column
    return inst


def load_windows_gen_from_csv(path, column_begin=Sett.load_windows_col_begin, column_end=Sett.load_windows_col_end,
                              sep=Sett.load_csv_separator):
    """
    Loads a collection win generator from a csv column
    @param path: path of the file to read
    @type path: str or unicode
    @param column_begin: column of the begin values to load
    @type column_begin: str or unicode
    @param column_end: column of the end values to load
    @type column_end: str or unicode
    @param sep: separator char
    @type sep: str or unicode or char
    """
    d = pd.read_csv(path, sep=sep)
    return CollectionWinGen(map((lambda x, y: Window(x, y)), d[column_begin], d[column_end]))


def save_ds_to_csv(data_series, path, name=Sett.load_rr_column_name, sep=Sett.load_csv_separator, header=True):
    """
    Saves the DataSeries to a csv file.
    @param path: path of the file to write
    @type path: str or unicode
    @param name: name of the column to save
    @type name: str or unicode
    @param sep: separator char
    @type sep: str or unicode or char
    """
    data_series.name = name
    data_series.to_csv(path, sep=sep, header=header)


def load_rr_from_ecg(path, delta=Sett.import_ecg_delta, ecg_col=Sett.load_ecg_column_name,
                     ecg_time_col=Sett.load_ecg_time_column_name, sep=Sett.load_csv_separator, *args):
    """
    Loads an IBI (RR) data series from an ECG data set and filters it with the specified filters list.
    @param path: path of the file to read
    @type path: str or unicode
    @param delta: delta parameter for the peak detection
    @type delta: float
    @param ecg_col: ecg values column
    @type ecg_col: str or unicode
    @param ecg_time_col: ecg timestamps column
    @type ecg_time_col: str or unicode
    @param sep: separator char
    @type sep: str or unicode or char
    @param args: sequence of filters to be applied to the data (e.g. from RRFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    # TODO: explain delta
    df = pd.read_csv(path, sep=sep, *args)
    max_tab, min_tab = peak_detection(df[ecg_col], delta,
                                      df[ecg_time_col])
    s = DataSeries(np.diff(max_tab))
    for f in Sett.import_ecg_filters:
        s = f(s)
    s.meta_tag['from_type'] = "csv_ecg"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = np.mean(np.diff(df[ecg_time_col]))
    s.meta_tag['from_filters'] = list(Sett.import_ecg_filters)
    return s


def load_rr_from_bvp(path, delta_ratio=Sett.import_bvp_delta_max_min_numerator, bvp_col=Sett.load_bvp_column_name,
                     bvp_time_col=Sett.load_bvp_time_column_name, sep=Sett.load_csv_separator,
                     filters=Sett.import_bvp_filters, *args):
    """
    Loads an IBI (RR) data series from a BVP data set and filters it with the specified filters list.
    @param path: path of the file to read
    @type path: str or unicode
    @param delta: delta parameter for the peak detection
    @type delta: float
    @param bvp_col: ecg values column
    @type bvp_col: str or unicode
    @param bvp_time_col: ecg timestamps column
    @type bvp_time_col: str or unicode
    @param sep: separator char
    @type sep: str or unicode or char
    @param args: sequence of filters to be applied to the data (e.g. from RRFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    df = pd.read_csv(path, sep=sep, *args)
    delta = (np.max(df[Sett.load_bvp_column_name]) - np.min(df[Sett.load_bvp_column_name])) / delta_ratio
    max_i, ii, iii, iv = peak_detection(df[bvp_col], delta,
                                        df[bvp_time_col])
    s = DataSeries(np.diff(max_i) * 1000)
    for f in filters:
        s = f(s)
    s.meta_tag['from_type'] = "csv_bvp"
    s.meta_tag['from_peak_delta'] = delta
    s.meta_tag['from_freq'] = np.mean(np.diff(df[bvp_time_col]))
    s.meta_tag['from_filters'] = list(Sett.import_bvp_filters)
    return s
