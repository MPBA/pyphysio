import pandas as pd

from ParamExecClass import ParamExecClass
import pyHRV
from pyHRV.windowing import CollectionWinGen, WindowsMapper
from pyHRV.Files import load_data_series


class GalaxyHRVAnalysis(ParamExecClass):
    """
    T_RR_CSV -> T_TAB_INDEXES
    T_RR_CSV + T_WIN_COLLECTION_CSV -> T_WINDOWS_TAB_INDEXES
    kwargs['input_w'] --> None or windows input file
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['indexes'] --> indexes list (names)
    """

    @staticmethod
    def calculate_indexes(data, indexes=None, wins=None):
        values = dict()
        errors = list()
        if indexes is None:
            indexes = pyHRV.get_available_indexes()
        for index in indexes:
            if not hasattr(pyHRV, index):
                errors.append(index)
        if len(errors) > 0:
            raise NameError(pyHRV.PyHRVDefaultSettings.Local.names(
                pyHRV.PyHRVDefaultSettings.Local.indexes_not_found, errors))
        else:
            if wins is None:
                for index in indexes:
                    values[index] = getattr(pyHRV, index)(data).value
            else:
                values = WindowsMapper(data, CollectionWinGen(data, wins), indexes)
        return values

    def execute(self):
        data = pyHRV.Files.load_rr(self._kwargs['input'])

        indexes = self._kwargs['indexes']

        wins = load_data_series(self._kwargs['input_w'])
        values = self.calculate_indexes(data, indexes, wins)

        pyHRV.Files.save_data_series(pd.Series(values), self._kwargs['output'])
        return values
