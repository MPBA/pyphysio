from ParamExecClass import ParamExecClass
import pandas as pd
import pyHRV

## DONE: load a .RR file (as functions do in Files)
## DONE: (by the wrapper) if the input file is a .tar.gz: un-tar and load more files
## TODO: add windowing


class GalaxyHRVAnalysis(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['indexes'] --> indexes list [1,0, ... 1,0]
    """
    def execute(self):
        data = pyHRV.Files.load_rr_data_series(self._kwargs['input'])
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

        pyHRV.Files.save_rr_data_series(pd.Series(values), self._kwargs['output'])
        return values
