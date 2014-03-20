from PyHRV.Files import *
from PyHRV.Indexes import FDIndexes as fdi
from PyHRV.Indexes import TDIndexes as tdi


## DONE: load a .RR file (as functions do in Files)
## DONE: (by the wrapper) if inputfile is a .tar.gz: untar and load more files
## TODO: add windowing

class GalaxyHRVAnalysis(object):
    @staticmethod
    def execute(input_file, indexes_list):
        data = load_rr_data_series(input_file)
        hrv_values = dict()

        i = 0

        #------------
        # TIME DOMAIN
        #------------
        if indexes_list[i] == 1:
            hrv_values["RRMean"] = tdi.RRMean(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRMean"] = tdi.HRMean(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRMedian"] = tdi.RRMedian(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRMedian"] = tdi.HRMedian(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRSTD"] = tdi.RRSTD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HRSTD"] = tdi.HRSTD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN10"] = tdi.PNNx(10, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN25"] = tdi.PNNx(25, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["PNN50"] = tdi.PNNx(50, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN10"] = tdi.NNx(10, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN25"] = tdi.NNx(25, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["NN50"] = tdi.NNx(50, data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["RRSSD"] = tdi.RMSSD(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["SDSD"] = tdi.SDSD(data).value
        i += 1

        #------------
        # FREQ DOMAIN
        #------------
        if indexes_list[i] == 1:
            hrv_values["VLF"] = fdi.VLF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LF"] = fdi.LF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HF"] = fdi.HF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["VLFpeak"] = fdi.VLFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LFpeak"] = fdi.LFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HFpeak"] = fdi.HFPeak(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["VLF_N"] = fdi.VLFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LF_N"] = fdi.LFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["HF_N"] = fdi.HFNormal(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["Total"] = fdi.Total(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["LFHF"] = fdi.LFHF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["nLF"] = fdi.NormalLF(data).value
        i += 1

        if indexes_list[i] == 1:
            hrv_values["nHF"] = fdi.NormalHF(data).value

        #-----------
        # NON LIN
        #-----------

        ## TODO: nonlin indexes
        return hrv_values
