# coding=utf-8
import numpy as _np
from pyPhysio.Utility import PhUI as _PhUI


class Parameter(object):
    def __init__(self, requirement_level, pytype, description, default=None, constraint=None, activation=None):
        self._pytype = pytype
        self._requirement_level = requirement_level
        self._description = description
        self._default = default
        self._constraint = constraint
        self._activation = activation

    def check_type(self, value):
        return (_np.issubdtype(type(value), int) and self._pytype is float) or \
               (_np.issubdtype(type(value), float) and value.is_integer()) or \
            _np.issubdtype(type(value), self._pytype)

    def check_constraint(self, value):
        return self._constraint is None or self._constraint(value)

    def __call__(self, params, name):
        value = params[name]
        if self._activation is None or self._activation(value, params):
            if not self.check_type(value):
                raise ValueError("Wrong parameter type (" + name + "): " + str(type(value)) + " not sub-dtype of " + str(self._pytype))
            elif not self.check_constraint(value):
                raise ValueError("Parameter constraint (" + name + "): " + self._description)
        return True, None

    def not_present(self, params, name, algo):
        if self._activation is None or self._activation(None, params):
            if self._requirement_level == 0:
                return True, self._default
            elif self._requirement_level == 1:
                _PhUI.i("Default value in " + algo.__class__.__name__ + " for " + name + ": " + str(self._default))
                return True, self._default
            elif self._requirement_level == 2:
                _PhUI.e("Missing parameter: " + name + " in " + algo.__class__.__name__ + ", " + self._description)
                return False, None
        else:
            return True, None