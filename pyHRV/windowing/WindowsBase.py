__author__ = 'AleB'
__all__ = ['WindowError', 'Window', 'WindowsIterator', 'WindowsGenerator']


class WindowError(Exception):
    pass


class Window(object):
    def __init__(self, begin, end):
        self._begin = begin
        self._end = end

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end


class WindowsIterator(object):
    def __init__(self, win):
        assert isinstance(win, WindowsGenerator)
        self._win = win

    def next(self):
        return self._win.step_windowing()


class WindowsGenerator(object):
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
        return Window(0, 0)
