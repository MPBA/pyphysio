import pandas as pd

from ParamExecClass import ParamExecClass
import pyHRV


## TODO: add windowing


class GalaxyHRVAnalysis(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['indexes'] --> indexes list [1,0, ... 1,0]
    """

    def execute(self):
        data = pyHRV.Files.load_rr(self._kwargs['input'])
        indexes = self._kwargs['indexes']
        values = dict()
        errors = list()

        # Pre-parse the list to save time
        for index in indexes:
            if not hasattr(pyHRV, index):
                errors.append(index)

        if len(errors) > 0:
            raise NameError(pyHRV.PyHRVDefaultSettings.Local.names(
                pyHRV.PyHRVDefaultSettings.Local.indexes_not_found, errors))
        else:
            for index in indexes:
                values[index] = getattr(pyHRV, index)(data).value

        pyHRV.Files.save_data_series(pd.Series(values), self._kwargs['output'])
        return values
