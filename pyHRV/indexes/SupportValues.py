##ck3
__author__ = 'AleB'

from BaseIndexes import SupportValue
from pyHRV.Utility import template_interpolation
from pyHRV.PyHRVSettings import PyHRVDefaultSettings as Ps


class SumSV(SupportValue):
    """
    Support value: SUM of the samples' values
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._s = 0
        self._c = sv_collection

    def enqueuing(self, new_value):
        self._s += new_value

    def dequeuing(self, old_value):
        self._s -= old_value

    @property
    def value(self):
        return self._s


class DistributionSV(SupportValue):
    """
    Support value: FREQUENCIES DISTRIBUTION of the VALUES CONSIDERED
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._c = sv_collection
        self._m = {}

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
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
    def value(self):
        return self._m


class MinSV(SupportValue):
    """
    Support value: the MINOR VALUE in the window
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._c = sv_collection
        self._v = None

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        if self._v is None:
            self._v = new_value
        else:
            if new_value < self._v:
                self._v = new_value

    def dequeuing(self, old_value):
        if old_value == self._v:
            self._v = min(self._c[DistributionSV].value.keys())

    @property
    def value(self):
        return self._v


class MaxSV(SupportValue):
    """
    Support value: MAXIMUM VALUE of the window
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._c = sv_collection
        self._v = None

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        if self._v is None:
            self._v = new_value
        else:
            if new_value > self._v:
                self._v = new_value

    def dequeuing(self, old_value):
        if old_value == self._v:
            self._v = max(self._c[DistributionSV].value.keys())

    @property
    def value(self):
        return self._v


class LengthSV(SupportValue):
    """
    Support value: VALUES NUMBER
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._c = sv_collection
        self._v = 0

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        self._v += 1

    def dequeuing(self, old_value):
        self._v -= 1

    @property
    def value(self):
        return self._v


class VectorSV(SupportValue):
    """
    Support value: a simple VECTOR of the VALUES in the window
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self.sv_collection = sv_collection
        self._v = []

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        self._v.insert(0, new_value)

    def dequeuing(self, old_value=None):
        if len(self._v) > 0:
            del self._v[-1]

    @property
    def value(self):
        return self._v


class InterpolationSV(SupportValue):
    """
    Support value: INTERPOLATION of the VALUES in (x O y ~ seconds O bpm) [O != 0]
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self.sv_collection = sv_collection
        self._v = []
        self._lx = 0
        self._ly = None
        self._fx = 0
        self._fy = None

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
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
    def value(self):
        return self._v


class DiffsSV(SupportValue):
    """
    Support value: VECTOR of the DIFFERENCES between adjacent VALUES
    """

    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self.sv_collection = sv_collection
        self._v = []
        self._last = None

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        self._v.insert(0, (new_value - self._last) if not self._last is None else 0)
        self._last = new_value

    def dequeuing(self, old_value=None):
        if len(self._v) > 0:
            del self._v[-1]

    @property
    def value(self):
        return self._v


class MedianSV(SupportValue):
    """
    Support value: VECTOR of the DIFFERENCES between adjacent VALUES
    """

    # noinspection PyUnusedLocal
    def __init__(self, sv_collection):
        SupportValue.__init__(self)
        self._hist = {}
        self._bal = 0
        self._split = 0
        self._m = None

    def enqueuing(self, new_value):
        SupportValue.enqueuing(self, None)
        if new_value in self._hist:
            self._hist[new_value] += 1
        else:
            self._hist[new_value] = 1
        if self._m is None:
            self._m = new_value
        else:
            self._bal += -1 if new_value < self._m else 1
            if self._bal > 1:
                self._bal -= 2
                self._split += 1
                if (self._m in self._hist and self._hist[self._m] == self._split) \
                        or (not self._m in self._hist and self._split == 0):
                    self._split = 0
                    self._m += 1
                    while self._m in self._hist and self._hist[self._m] > 0:
                        self._m += 1
            elif self._bal < 0:
                self._bal += 2
                self._split -= 1
                if self._split < 0:
                    self._split = 0
                    self._m -= 1
                    while self._m in self._hist and self._hist[self._m] > 0:
                        self._m -= 1
                    if self._m in self._hist:
                        self._split = self._hist[self._m] - 1
                    else:
                        self._split = 0 - 1
            elif (self._m in self._hist and self._hist[self._m] == self._split) \
                    or (not self._m in self._hist and self._split == 0):
                self._split = 0
                self._m += 1
                while self._m in self._hist and self._hist[self._m] > 0:
                    self._m += 1

    def dequeuing(self, old_value=None):
        self._hist[old_value] -= 1
        self._bal += 1 if old_value < self._m else -1

    @property
    def value(self):
        return self._m
