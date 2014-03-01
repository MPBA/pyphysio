import pyhrv as ph
import optparse

## TODO: load a .RR file
## TODO: add windowing
## TODO: if inputfile is a .tar.gz: untar and load more files



usage = "Usage: load RR file"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputfile",
                  action="store", type="string",
                  dest="input_file", help="Input File")

parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")

# parser.add_option("-l", "--list",
#                   action="store", type="string", default='',
#                   dest="hrvlist", help="List of HRV indexes to be computed")

parser.add_option("-w", "--windowfile",
                  action="store", type="string", default='',
                  dest="window_file", help="Window File")
# parser.add_option("-n", "--name",
#                   action="store", type="string", default='RR',
#                   dest="colname", help="Name of RR column")

(options, args) = parser.parse_args()
INPUTFILE=options.input_file
OUTDIR=options.output_dir
WINFILE=options.window_file


HRVlist=[True,True,True,True]
RRdata=ph.DataSeries()
# inputfile is a csv, but it will be a .RR file, created with RRData.save(...)
RRdata.load_from_csv(INPUTFILE, sep='\t', colname='IBI')

INDEXES=list()
if HRVlist[0]:
    rrmean=ph.RRmean(RRdata)
    RRMEAN=rrmean.calculate()
    INDEXES.append(RRMEAN)
if HRVlist[1]:
    rrstd=ph.RRSTD(RRdata)
    RRSTD=rrstd.calculate()
    INDEXES.append(RRSTD)
if HRVlist[2]:
    pnn50=ph.pNNX(RRdata, X=50)
    PNN50=pnn50.calculate()
    INDEXES.append(PNN50)
if HRVlist[3]:
    pnn25=ph.pNNX(RRdata, X=25)
    PNN25=pnn25.calculate()
    INDEXES.append(PNN25)

print(INDEXES)
