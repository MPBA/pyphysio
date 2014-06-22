from pyHRV.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

import pyHRV


class GalaxyOnLineHRVAnalysis(ParamExecClass):
    """
    kwargs['value'] ------> indexes to calculate
    kwargs['indexes'] ----> new value not None
    kwargs['state'] ------> last support values class not None
    return:
    last_value, updated_state
    """

    def execute(self):
        indexes = self._kwargs['indexes']
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

            values = map(lambda x: getattr(pyHRV, x).calculate_on(state), indexes)

        return values, state
