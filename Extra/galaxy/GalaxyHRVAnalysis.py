import pandas as pd

import pyPhysio
from pyPhysio.Files import load_windows_gen_from_csv
from pyPhysio.galaxy.ParamExecClass import ParamExecClass


class GalaxyHRVAnalysis(ParamExecClass):
    """
    T_RR_CSV -> T_TAB_INDEXES
    T_RR_CSV + T_WIN_COLLECTION_CSV -> T_WINDOWS_TAB_INDEXES
    kwargs['input_w'] --> None or windows input file
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['features'] --> features list (names)
    """

    @staticmethod
    def calculate_indexes(data, indexes=None, wing=None):
        errors = list()
        if indexes is None:
            indexes = pyPhysio.get_available_indexes()
        for index in indexes:
            if not hasattr(pyPhysio, index):
                errors.append(index)
        if len(errors) > 0:
            raise NameError(pyPhysio.MainSettings.Local.names(
                pyPhysio.MainSettings.Local.indexes_not_found, errors))
        else:
            if wing is None:
                values = pd.DataFrame(columns=indexes)
                for index in indexes:
                    values[index] = getattr(pyPhysio, index)(data).value
            else:
                m = pyPhysio.WindowsIterator(data, wing, indexes)
                m.compute_all()
                values = pd.DataFrame(columns=m.labels, data=m.results)
        return values

    def execute(self):
        data = pyPhysio.Files.load_ds_from_csv_column(self._kwargs['input'])
        wins = load_windows_gen_from_csv(self._kwargs['input_w']) if 'input_w' in self._kwargs else None

        indexes = self._kwargs['features']

        values = self.calculate_indexes(data, indexes, wins)
        values.to_csv(self._kwargs['output'], sep=pyPhysio.MainSettings.load_csv_separator)
        return values
