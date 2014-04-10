__author__ = 'AleB'
__all__ = ['WindowError', 'Window', 'WindowsGenerator']


class WindowError(Exception):
    """Generic Windowing error.
    """
    pass


class Window(object):
    """Base Window, a begin-end pair
    """

    def __init__(self, begin, end, name=None):
        self._begin = begin
        self._end = end
        self._name = name

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @property
    def len(self):
        return self._end - self._begin

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "Win(%d, %d, %s)" % (self.begin, self.end, self.name)


class WindowsIterator(object):
    """A generic iterator that is called from each WindowGenerator from the __iter__ method.
    Not for the end user.
    """

    def __init__(self, win):
        assert isinstance(win, WindowsGenerator)
        self._win = win

    def next(self):
        return self._win.step_windowing()


class WindowsGenerator(object):
    """Base and abstract class for the windows computation.
    """

    def __init__(self, data=None):
        self._data = None
        if data is None:
            pass
        else:
            self.init_windowing(data)

    def __iter__(self):
        return WindowsIterator(self)

    def init_windowing(self, data):
        # the class must have a reference to the used data
        self._data = data

    def step_windowing(self):
        raise StopIteration
