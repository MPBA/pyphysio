__author__ = 'AleB'


class PhUI(object):
    @staticmethod
    def a(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def i(mex):
        PhUI.p(mex, '', 35)

    @staticmethod
    def o(mex):
        PhUI.p(mex, '', 31)

    @staticmethod
    def w(mex):
        PhUI.p(mex, 'Warning: ', 33)

    @staticmethod
    def p(mex, lev, col):
        print(">%s\x1b[%dm%s\x1b[39m" % (lev, col, mex))