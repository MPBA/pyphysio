import pandas as pd

from ParamExecClass import ParamExecClass
import pyHRV

# TODO: add windowing


class GalaxyHRVAnalysis(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['indexes'] --> indexes list (names)
    """

    @staticmethod
    def calculate_indexes(data, indexes=None):
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
            for index in indexes:
                values[index] = getattr(pyHRV, index)(data).value
        return values

    def execute(self):
        data = pyHRV.Files.load_rr(self._kwargs['input'])
        indexes = self._kwargs['indexes']

        values = self.calculate_indexes(data, indexes)

        pyHRV.Files.save_data_series(pd.Series(values), self._kwargs['output'])
        return values
