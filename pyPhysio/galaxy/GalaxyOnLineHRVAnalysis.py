from pyPhysio.galaxy.ParamExecClass import ParamExecClass

__author__ = 'AleB'

import pyPhysio


class GalaxyOnLineHRVAnalysis(ParamExecClass):
    """
    kwargs['value'] ------> features to calculate
    kwargs['features'] ----> new value not None
    kwargs['state'] ------> last support values class not None
    return:
    last_value, updated_state
    """

    def execute(self):
        indexes = self._kwargs['features']
        state = self._kwargs['state']
        value = self._kwargs['value']
        errors = list()

        # Pre-parse the list to save time
        for index in indexes:
            if not hasattr(pyPhysio, index):
                errors.append(index)

        if len(errors) > 0:
            raise NameError(pyPhysio.MainSettings.Local.names(
                pyPhysio.MainSettings.Local.indexes_not_found, errors))
        else:
            state.update(value)

            values = map(lambda x: getattr(pyPhysio, x).calculate_on(state), indexes)

        return values, state
