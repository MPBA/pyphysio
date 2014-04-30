__author__ = 'AleB'

from BaseIndexes import SupportValue
from pyHRV.utility import template_interpolation
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Ps


class SumSV(SupportValue):
    """Support value: SUM
    """

    def __init__(self, sv_collection=None):
        self._s = 0
        self._c = sv_collection

    def enqueuing(self, new_value):
        self._s += new_value

    def dequeuing(self, old_value):
        self._s -= old_value


class DistributionSV(SupportValue):
    """Support value: FREQUENCIES DISTRIBUTION of the VALUES CONSIDERED
    """

    def __init__(self, sv_collection):
        self._c = sv_collection
        self._m = {}

    def enqueuing(self, new_value):
        if new_value in self._m:
            self._m[new_value] += 1  # can not be 0
        else:
            self._m[new_value] = 1  # as

    def dequeuing(self, old_value):
        assert not old_value is None, "Big Error: No values to dequeue"
        if self._m[old_value] == 1:  # can not be 0
            del self._m[old_value]  # as
        else:
            self._m[old_value] -= 1

    @property
    def items(self):
        """READ-ONLY!!!
        """
        return self._m


class MinSV(SupportValue):
    """Support value: MINOR VALUE
    """

    def __init__(self, sv_collection):
        self._c = sv_collection
        self._v = None

    def enqueuing(self, new_value):
        if self._v is None:
            self._v = new_value
        else:
            if new_value < self._v:
                self._v = new_value

    def dequeuing(self, old_value):
        if old_value == self._v:
            self._v = min(self._c[DistributionSV].items.keys())


class MaxSV(SupportValue):
    """Support value: MAXIMUM VALUE
    """

    def __init__(self, sv_collection):
        self._c = sv_collection
        self._v = None

    def enqueuing(self, new_value):
        if self._v is None:
            self._v = new_value
        else:
            if new_value > self._v:
                self._v = new_value

    def dequeuing(self, old_value):
        if old_value == self._v:
            self._v = max(self._c[DistributionSV].items.keys())


class LengthSV(SupportValue):
    """Support value: VALUES NUMBER
    """

    def __init__(self, sv_collection):
        self._c = sv_collection
        self._v = 0

    def enqueuing(self, new_value):
        self._v += 1

    def dequeuing(self, old_value):
        self._v -= 1


class VectorSV(SupportValue):
    """Support value: VECTOR of the VALUES
    """

    def __init__(self):
        self._v = []

    def enqueuing(self, new_value):
        self._v.append(new_value)

    def dequeuing(self, old_value=None):
        if len(self._v) > 0:
            del self._v[-1]

    @property
    def items(self):
        """READ-ONLY!!!
        """
        return self._v


class InterpolationSV(SupportValue):
    """Support value: INTERPOLATION of the VALUES in (seconds O bpm) [~ xOy]
    """

    def __init__(self):
        self._v = []
        self._lx = 0
        self._ly = None
        self._fx = 0
        self._fy = None

    def enqueuing(self, new_value):
        y = 60000.0 / new_value
        x = self._lx + new_value / 1000.0  # NO TIME-TOLERANCE ERRORS!
        if self._ly is None:  # with the first IBI I get the hr of the first (t=x=0) beat and of the second (t=x=IBI)
            self._fy = self._ly = y
            self._fx = self._lx = 0.0  # NO TIME-TOLERANCE ERRORS!
        hr, t = template_interpolation([self._ly, y],
                                       [self._lx, x],
                                       Ps.default_interpolation_freq)
        print ([self._ly, y], [self._lx, x]), hr
        self._v.extend(hr)
        self._lx = x  # NO TIME-TOLERANCE ERRORS!

    def dequeuing(self, old_value):
        mt = old_value / 1000.0  # NO TIME-TOLERANCE ERRORS!
        ct = self._fx
        while ct + self._v[0] <= mt:
            ct += self._v[0]
            del self._v[0]

    @property
    def items(self):
        """READ-ONLY!!!
        """
        return self._v
