# coding=utf-8
__all__ = ['load_pd_from_excel_column', 'load_ds_from_csv_column', 'load_windows_gen_from_csv', 'save_ds_to_csv',
           'load_ibi_from_bvp', 'load_ibi_from_ecg']

import pandas as pd

from pyPhysio.Utility import data_series_from_bvp, data_series_from_ecg
from pyPhysio.PyHRVSettings import MainSettings as Sett
from pyPhysio.segmentation.WindowsBase import Segment
from pyPhysio.segmentation.WindowsGenerators import ExistingSegments


def load_pd_from_excel_column(path, column, column_b=None, sheet_name=0):
    """
    Loads one or two columns as pandas.Series from an excel format file.
    @param path: path of the file to read or file-like object
    @param column: first column to load
    @type column: int
    @param column_b: second optional column to load
    @type column_b: int or None
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
    if column not in d.columns:
        column = d.columns[0]
    inst = pd.TimeSeries(d[column])
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
    return ExistingSegments(map((lambda x, y: Segment(x, y, None)), d[column_begin], d[column_end]))


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
    data_series.label = name
    data_series.to_csv(path, sep=sep, header=header)


def load_ibi_from_ecg(path, delta=Sett.import_ecg_delta, ecg_col=Sett.load_ecg_column_name,
                      ecg_time_col=Sett.load_ecg_time_column_name, filters=Sett.import_bvp_filters,
                      sep=Sett.load_csv_separator, *args):
    """
    Loads an IBI (RR) data series from an ECG data set csv file and filters it with the specified filters list.
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
    @param filters: list of filters to be applied to the data (e.g. from IBIFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    # TODO: explain delta
    df = pd.read_csv(path, sep=sep, *args)
    return data_series_from_ecg(df[ecg_col], df[ecg_time_col], delta, filters)


def load_ibi_from_bvp(path, delta_ratio=Sett.import_bvp_delta_max_min_numerator, bvp_col=Sett.load_bvp_column_name,
                      bvp_time_col=Sett.load_bvp_time_column_name, filters=Sett.import_bvp_filters,
                      sep=Sett.load_csv_separator, *args):
    """
    Loads an IBI (RR) data series from a BVP data set csv file and filters it with the specified filters list.
    @param path: path of the file to read
    @type path: str or unicode
    @param delta_ratio: delta parameter for the peak detection
    @type delta_ratio: float
    @param bvp_col: ecg values column
    @type bvp_col: str or unicode
    @param bvp_time_col: ecg timestamps column
    @type bvp_time_col: str or unicode
    @param sep: separator char
    @type sep: str or unicode or char
    @param filters: sequence of filters to be applied to the data (e.g. from IBIFilters)
    @return: Filtered signal DataSeries
    @rtype: DataSeries
    """
    # TODO: explain delta
    df = pd.read_csv(path, sep=sep, *args)
    return data_series_from_bvp(df[bvp_col], df[bvp_time_col], delta_ratio, filters)
