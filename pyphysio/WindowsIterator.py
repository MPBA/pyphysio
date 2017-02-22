# # coding=utf-8
# from BaseSegmentation import SegmentationIterator as _SegmentationIterator
# from Utility import PhUI as _PhUI
# from BaseAlgorithm import Algorithm
#
# __author__ = 'AleB'
#
#
# class WindowsIterator(object):
#     """
#     Takes some indicators and calculates them on the given set of windows.
#     Allows the iteration of the computation of a list of indicators over a WindowsGenerator.
#     Use compute_all to execute the computation.
#     """
#
#     verbose = True
#
#     def __init__(self, data, win_gen, indexes, params):
#         """
#         Initializes
#         @param data: data on which compute windowed indicators
#         @param win_gen: the windows generator
#         @param indexes: list of classes as CLASS(DATA).value() ==> index value
#         """
#         self._data = data
#         self._map = None
#         self._wing = win_gen
#         self._win_iter = win_gen.__iter__()
#         self._feats = indexes
#         self._winn = -1
#         self._params = params
#
#     def __iter__(self):
#         return _SegmentationIterator(self)
#
#     def _comp_one(self, win):
#         ret = []
#         win_ds = win(self._data)
#         for algorithm in self._feats:
#             if isinstance(algorithm, str) or isinstance(algorithm, unicode):
#                 p = vars()["algorithm"]
#                 ret.append(p(win_ds))
#             elif isinstance(algorithm, Algorithm):
#                 p = algorithm(self._params)
#                 ret.append(p(win_ds))
#             else:
#                 _PhUI.w("The specified algorithm '%s' is not an algorithm nor a PyPhysio algorithm name." % algorithm)
#         self._winn += 1
#         return [self._winn if win.get_label() is None else win.get_label(), win.get_begin(), win.get_end()] + ret
#
#     def step_windowing(self):
#         return self._comp_one(self._win_iter.next())
#
#     def compute_all(self):
#         """
#         Executes the indicators computation (mapping with the windows).
#         """
#         self._map = []
#         for w in self._wing:
#             if WindowsIterator.verbose:
#                 _PhUI.i("Processing " + str(w))
#             self._map.append(self._comp_one(w))
#         df = self._map
#         df.columns = self.labels()  # TODO: improve-me
#         return df
#
#     def labels(self):
#         """
#         Gets the labels of the table returned from the results property after the compute_all call.
#         @rtype : list
#         """
#         ret = ['w_name', 'w_begin', 'w_end']
#         for index in self._feats:
#             if isinstance(index, str) | isinstance(index, unicode):
#                 assert False, "The string addressing is temporarily not supported"
#                 # index = getattr(pyPhysio, index)
#             if isinstance(index, type):
#                 ret.append(index.__name__)
#             else:
#                 ret.append(index.__repr__())
#         return ret
#
#     def results(self):
#         """
#         Returns the results table calculated in the compute_all call.
#         @return: dict
#         """
#         return self._map
