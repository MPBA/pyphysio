__author__ = 'AleB'

from ParamExecClass import ParamExecClass
import pyHRV


class GalaxyOnLineHRVAnalysis(ParamExecClass):
    """
    kwargs['value'] ----> new value not None
    kwargs['state'] ----> last support values class not None
    return:
    last_value, updated_state
    """

    def execute(self):
        indexes = ['Mean']
        state = self._kwargs['state']
        value = self._kwargs['value']
        errors = list()

        # Pre-parse the list to save time
        for index in indexes:
            if not hasattr(pyHRV, index):
                errors.append(index)

        if len(errors) > 0:
            raise NameError(pyHRV.PyHRVDefaultSettings.Local.names(
                pyHRV.PyHRVDefaultSettings.Local.indexes_not_found, errors))
        else:
            state.update(value)

            for index in indexes:
                value = getattr(pyHRV, index).calculate_on(state)

        return value, state
