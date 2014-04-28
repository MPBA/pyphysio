__author__ = 'AleB'
__all__ = ['SupportValuesCollection']
from BaseIndexes import SupportValue


class SupportValuesCollection(object):
    def __init__(self, win_size=50):
        self._win_size = win_size
        self._supp = {}

    def __len__(self):
        return self._supp

    def __getitem__(self, item):
        """
        @rtype : SupportValue
        """
        if not item in self._supp:
            assert isinstance(item, type)
            self._supp[item] = item(self)
        return self._supp[item]

    def __delitem__(self, key):
        del self._supp[key]

    def __iter__(self):
        return self._supp.__iter__()
