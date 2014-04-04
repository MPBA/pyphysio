import numpy as np

#TODOs: ???
# TODO: class RTDataSeries
# TODO: class RT_RR
# TODO: class RT_RRdiffs
# TODO: class RT_fft


class RRvect(object):
    def __init__(self, values=None):
        self.RR = np.array(values)
        self.length = len(values[:, 0])

    def __getitem__(self, item):
        return self.RR


    def nextRR(self, new, win=0):
        """ return new RRvect with deleted elements WITHOUT updating it
        """
        RRcurr = self.RR
        TS_NEW = new[0]
        TS_limit = TS_NEW - win
        if not win == 0:
            i = 0
            for ind in range(len(RRcurr)):
                print ind
                TS_current = RRcurr[ind, 0]
                if TS_current < TS_limit:
                    print TS_current
                    i += 1
                else:
                    break
        else:
            i = 1

        RRcurr = RRcurr[i:, :]
        RRcurr = np.vstack([RRcurr, new])
        RRdeleted = RRcurr[0:i, :]
        return RRvect(RRcurr), RRvect(RRdeleted)

    def update(self, RRcurr, RRdeleted):
        self.RR = RRcurr

    def getNewer(self):
        return self.RR[-1, :]

    def getOlder(self):
        return self.RR[0, :]

    def getOldestTS(self):
        return self.RR[0, 0]

    def getNewestTS(self):
        return self.RR[-1, 0]

    def getRRs(self):
        return self.RR[:, 1]

    def getTSs(self):
        return self.RR[:, 0]


class RRmean(object):
    def __init__(self, lastRR):
        self.sum = np.sum(lastRR.getRRs())

    def update(self, RRcurr, RRdeleted):
        RRnew = RRcurr.getRRs()
        self.sum = self.sum - np.sum(RRdeleted.getRRs()) + RRnew[-1]
        return self.sum / len(RRcurr.getRRs())


class pNNx(object):
    def __init__(self, lastRR, X):
        diffs = abs(np.diff(lastRR.getRRs()))
        self.NNx = len(diffs[diffs >= X])
        self.X = X

    def update(self, RRcurr, RRdeleted):
        RRnew = RRcurr.getRRs()
        RRdiffs_OLD = abs(np.diff(np.hstack([RRnew[-1],
                                             RRdeleted.getRRs()])))  # devo aggiungere anche il piu vecchio dei nuovi
        # per calcolare le differenze
        last_diffs = RRnew[-1] - RRnew[-2]
        if last_diffs >= self.X:
            A = 1
        else:
            A = 0
        self.NNx = self.NNx - len(RRdiffs_OLD[RRdiffs_OLD >= self.X]) + A

        return self.NNx


RRact = np.zeros([10, 2])
RRact[:, 0] = np.arange(10)
RRact[:, 1] = [1, 10, 2, 10, 3, 10, 4, 10, 5, 10]

RRnew = RRvect(RRact)

mean = RRmean(RRnew)

pnn0 = pNNx(RRnew, 0)

RRnuovo = [11, 1]

RRcurr, RRdeleted = RRnew.nextRR(RRnuovo)

mean.update(RRcurr, RRdeleted)
pnn0.update(RRcurr, RRdeleted)
