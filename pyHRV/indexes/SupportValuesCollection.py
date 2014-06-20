__author__ = 'AleB'
__all__ = ['SupportValuesCollection']

from BaseIndexes import SupportValue
from SupportValues import VectorSV


class SupportValuesCollection(object):
    def __init__(self, indexes, win_size=50):
        self._win_size = win_size
        self._supp = {VectorSV: VectorSV()}
        for i in indexes:
            for r in i.required_sv():
                if not r in self._supp:
                    self._supp[r] = r()

    def __len__(self):
        return self._supp

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

    def ready(self):
        return len(self._supp[VectorSV].items) >= self._win_size

    def update(self, new_value):
        for s in self._supp.values():
            s.enqueuing(new_value)
        if self.ready():
            old_value = self._supp[VectorSV].items[-1]
            for s in self._supp.values():
                s.dequeuing(old_value)
