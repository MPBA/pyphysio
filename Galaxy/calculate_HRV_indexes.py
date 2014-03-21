from PyHRV.Files import *
from PyHRV.Indexes import FDIndexes as FDI
from PyHRV.Indexes import TDIndexes as TDI
from ParamExecClass import ParamExecClass

## DONE: load a .RR file (as functions do in Files)
## DONE: (by the wrapper) if the input file is a .tar.gz: un-tar and load more files
## TODO: add windowing


class GalaxyHRVAnalysis(ParamExecClass):
    """
    kwargs['input'] ----> input file
    kwargs['output'] ---> output file
    kwargs['indexes'] --> indexes list [1,0, ... 1,0]
    """
    def execute(self):
        data = load_rr_data_series(self._kwargs['input'])
        indexes_list = self._kwargs['indexes']
        hrv_values = dict()

        i = 0

        #------------
        # TIME DOMAIN
        #------------
        if indexes_list[i] == 1:
            hrv_values["RRMean"] = TDI.RRMean(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRMean"] = TDI.HRMean(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRMedian"] = TDI.RRMedian(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRMedian"] = TDI.HRMedian(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRSTD"] = TDI.RRSTD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRSTD"] = TDI.HRSTD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN10"] = TDI.PNNx(10, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN25"] = TDI.PNNx(25, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN50"] = TDI.PNNx(50, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN10"] = TDI.NNx(10, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN25"] = TDI.NNx(25, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN50"] = TDI.NNx(50, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRSSD"] = TDI.RMSSD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["SDSD"] = TDI.SDSD(data).value
        i += 1

        #------------
        # FREQ DOMAIN
        #------------
        if indexes_list[i] == 1:
            hrv_values["VLF"] = FDI.VLF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LF"] = FDI.LF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HF"] = FDI.HF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["VLFpeak"] = FDI.VLFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LFpeak"] = FDI.LFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HFpeak"] = FDI.HFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["VLF_N"] = FDI.VLFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LF_N"] = FDI.LFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HF_N"] = FDI.HFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["Total"] = FDI.Total(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LFHF"] = FDI.LFHF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["nLF"] = FDI.NormalLF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["nHF"] = FDI.NormalHF(data).value

        #-----------
        # NON LIN
        #-----------

        ## TODO: nonlin indexes

        save_rr_data_series(pd.Series(hrv_values), self._kwargs['output'])
        return hrv_values
