# coding=utf-8
# TODO import names
__author__ = 'AleB'

__all_indexes__ = []
# __all_indexes__.extend(filter(lambda x: x[0] != '_', dir(TDFeatures)))
# __all_indexes__.extend(filter(lambda x: x[0] != '_', dir(FDFeatures)))
# __all_indexes__.extend(filter(lambda x: x[0] != '_', dir(NonLinearFeatures)))


def get_available_indexes():
    return __all_indexes__


def get_available_online_indexes():
    return filter(lambda x: hasattr(x in vars(), "required_sv"), get_available_indexes())
