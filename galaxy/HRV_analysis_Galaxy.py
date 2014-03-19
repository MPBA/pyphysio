from DataSeries import DataSeries
from Indexes import *
from Files import *
import optparse

## DONE: load a .RR file (as functions do in Files)
## TODO: add windowing
## TODO: if inputfile is a .tar.gz: untar and load more files


if __name__ == '__main__':
    usage = "Usage: load RR file"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-i", "--input",
                      action="store", type="string",
                      dest="input_file", help="Input File")

    parser.add_option("-o", "--output",
                      action="store", type="string",
                      dest="output_dir", help="Output Dir")

    parser.add_option("-l", "--list",
                      action="store", type="string", default='',
                      dest="hrvlist", help="List of HRV indexes to be computed")

    parser.add_option("-w", "--windowfile",
                      action="store", type="string", default='',
                      dest="window_file", help="Window File")
    # parser.add_option("-n", "--name",
    #                   action="store", type="string", default='RR',
    #                   dest="colname", help="Name of RR column")

    (options, args) = parser.parse_args()
    INPUTFILE = options.input_file
    OUTDIR = options.output_dir
    WINFILE = options.window_file
    HRVLIST = options.hrvlist.split(',')


    # HRVlist = [True,True,True,True]
    # inputfile is a csv, but it will be a .RR file, created with RRData.save(...)

    RRdata = load_rr_data_series(INPUTFILE)

    INDEXES = list()
    if HRVLIST[0]:
        INDEXES.append(RRMean(RRdata).value)
    if HRVLIST[1]:
        INDEXES.append(RRSTD(RRdata).value)
    if HRVLIST[2]:
        INDEXES.append(PNNx(50, RRdata).value)
    if HRVLIST[3]:
        INDEXES.append(PNNx(25, RRdata).value)

    print(INDEXES)


class GalaxyHRVAnalysis(object):
    def execute(self, **kwargs):
        INPUTFILE = kwargs['input']
        OUTDIR = kwargs['output']
        HRVLIST = kwargs['list']
        indexes = HRVLIST.split(',')

        RRdata = load_rr_data_series(INPUTFILE)

        hrv_values = list()

        i = 0

        #------------
        # TIME DOMAIN
        #------------
        if indexes[i] == 1:
            hrv_values["RRMean"] = RRMean(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HRMean"] = HRMean(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["RRMedian"] = RRMedian(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HRMedian"] = HRMedian(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["RRSTD"] = RRSTD(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HRSTD"] = HRSTD(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["PNN10"] = PNNx(10, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["PNN25"] = PNNx(25, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["PNN50"] = PNNx(50, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["NN10"] = NNx(10, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["NN25"] = NNx(25, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["NN50"] = NNx(50, RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["RRSSD"] = RMSSD(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["SDSD"] = SDSD(RRdata).value
        i += 1

        #------------
        # FREQ DOMAIN
        #------------

        if indexes[i] == 1:
            hrv_values["VLF"] = VLF(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["LF"] = LF(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HF"] = HF(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["VLFpeak"] = VLFPeak(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["LFpeak"] = LFPeak(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HFpeak"] = HFPeak(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["VLF_N"] = VLFNormal(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["LF_N"] = LFNormal(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["HF_N"] = HFNormal(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["Total"] = Total(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["LFHF"] = LFHF(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["nLF"] = NormalLF(RRdata).value
        i += 1

        if indexes[i] == 1:
            hrv_values["nHF"] = NormalHF(RRdata).value

        #-----------
        # NON LIN
        #-----------

        ## TODO: nonlin indexes

        #pd.DataFrame(results).to_csv(OUTDIR, header=True)
        return hrv_values