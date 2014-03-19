import inspect
import os
from PyHRV.Galaxy.OldTools.a_wrap_tgz import wrap_tgz


def main():
    """
    General wrapping script
    """
    ff, fe = os.path.splitext(inspect.getfile(inspect.currentframe()))
    # !!! This file name dependent !!!
    wrap_tgz(ff[:-2] + fe)

if __name__ == "__main__":
    main()
