from pyPhysio.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

from pandas import Series

from pyPhysio.segmentation import LabeledSegments, ExistingSegments, Segment
from pyPhysio.Files import *


class GalaxyLoadWindows(ParamExecClass):
    """
    T_FILE_LABELS -> T_WIN_COLLECTION_CSV
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['column'] ---> column of the windows to load
    kwargs['windows_type'] ---> windows type to load in
                                [ 'labels_sequences_linear', 'labels_sequences', 'begin_values' ]
    """

    def execute(self):
        output_file = self._kwargs['output']
        if self._kwargs['excel']:
            c = load_pd_from_excel_column(self._kwargs['input'], self._kwargs['column'], None, self._kwargs['sheet'])
        else:
            c = load_ds_from_csv_column(self._kwargs['input'], self._kwargs['column'])

        if self._kwargs['windows_type'] == 'labels_sequences':
            w = LabeledSegments(None, c)
        else:
            if self._kwargs['windows_type'] == 'begin_values':
                w = map(self.__class__.map_end_window, c)[1:]
                w = ExistingSegments(None, w)
            else:
                raise NotImplemented("Not implemented segmentation mode: %s" % self._kwargs['windows_type'])
        Series(w).save(output_file)

    def map_end_window(self, end):
        if self.__s is None:
            self.__s = end
        else:
            return Segment(self.__s, end)
