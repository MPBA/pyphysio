import pandas as pd

import pyHRV
from pyHRV.Files import load_windows
from pyHRV.galaxy.ParamExecClass import ParamExecClass
from pyHRV.windowing import CollectionWinGen


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
                values = pd.DataFrame(columns=indexes)
                for index in indexes:
                    values[index] = getattr(pyHRV, index)(data).value
            else:
                m = pyHRV.WindowsMapper(data, CollectionWinGen(data, wins), indexes)
                m.compute_all()
                values = pd.DataFrame(columns=m.labels, data=m.results)
        return values

    def execute(self):
        data = pyHRV.Files.load_rr(self._kwargs['input'])
        wins = load_windows(self._kwargs['input_w']) if 'input_w' in self._kwargs else None

        indexes = self._kwargs['indexes']

        values = self.calculate_indexes(data, indexes, wins)
        values.to_csv(self._kwargs['output'], sep=pyHRV.PyHRVDefaultSettings.load_csv_separator)
        return values
