__author__ = 'AleB'
__all__ = ['SupportValuesCollection']


class SupportValuesCollection(object):
    def __init__(self, window=50):
        self._last = []
        self._win_size = window
        self._p = {}
        self._len = 0
        self._old = None

    def get(self, index, default=0):
        if not index in self._p:
            self._p[index] = default
        return self._p[index]

    def set(self, index, value):
        self._p[index] = value

    @property
    def old(self):
        return self._last[0]

    @property
    def last(self):
        return self._last[0]

    @property
    def new(self):
        return self._last[-1]

    @property
    def vec(self):
        return self._last

    def update(self, values):
        for a in values:
            self._enqueue(a)
        if self._win_size >= 0:
            while self.len > self._win_size:
                self._dequeue()

    @property
    def len(self):
        return self._len

    @property
    def ready(self):
        return self._win_size < 0 < self.len or self.len == self._win_size

    def _enqueue(self, val):
        self._last.append(val)

    def _dequeue(self):
        val = self._last[0]
        del self._last[0]
        self._old = val
        self._len -= 1
