import numpy as _np


class Parameter(object):
    def __init__(self, requirement_level, pytype, description, default=None, constraint=None, activation=None):
        self._pytype = pytype
        self._requirement_level = requirement_level
        self._description = description
        self._default = default
        self._constraint = constraint
        self._activation = activation

    # TODO: manage defaults

    def __call__(self, params, value):
        if self._activation is None or self._activation(params, value):
            for check in self.get_checks():
                if not getattr(self, check)(value):
                    return False, getattr(self, check)
        return True, None

    def get_checks(self):
        return filter(lambda x: x[:5] == "check",
                      [method for method in dir(self) if callable(getattr(self, method))])

    def check_type(self, value):
        return _np.issubdtype(_np.dtype(type(value)), self._pytype)

    def check_requirement(self, value):
        return self._requirement_level != 0 and value is not None

    def check_constraint(self, value):
        return self._constraint is None or self._constraint(value)
