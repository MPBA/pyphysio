__author__ = 'AleB'

import WindowsGenerators
from WindowsGenerators import *
from pyPhysio import WindowsIterator as Wm

__all__ = WindowsGenerators.__all__
__all__.extend(Wm.__all__)

del Wm
