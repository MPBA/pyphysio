##ck3
__author__ = 'AleB'
__all__ = ['SupportValuesCollection']

from BaseIndexes import SupportValue
from SupportValues import VectorSV
import pyHRV


class SupportValuesCollection(object):
    """
    This container-class helps the management of a set of support values required to calculate the needed indexes in the
    on-line mode.
    """

    def __init__(self, indexes, win_size=50):
        """
        Initializes the management system.
        @param indexes: List of indexes needed.
        @param win_size: Size in samples of the on-line window.
        """
        self._win_size = win_size
        self._supp = None
        self._supp = {VectorSV: VectorSV(self._supp)}
        for i in indexes:
            for r in getattr(pyHRV, i).required_sv():
                if not r in self._supp:
                    self._supp[r] = r(self._supp)

    def __len__(self):
        return len(self._supp)

    def __getitem__(self, item):
        """
        @rtype : SupportValue
        """
        if not item in self._supp:
            print("SupportValue not initialized: " + str(item))
            return None
        return self._supp[item]

    def __iter__(self):
        return self._supp.__iter__()

    @property
    def ready(self):
        """
        Indicates weather the collection has enough data to calculate the indexes in the window.
        @rtype: bool
        """
        return len(self._supp[VectorSV].value) >= self._win_size

    def update(self, new_value):
        """
        Adds a new sample to the support values.
        @param new_value: The new sample's value.
        @type new_value: float
        """
        for s in self._supp.values():
            s.enqueuing(new_value)
        if self.ready:
            old_value = self._supp[VectorSV].value[-1]
            for s in self._supp.values():
                s.dequeuing(old_value)
