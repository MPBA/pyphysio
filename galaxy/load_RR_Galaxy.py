import pyhrv as ph
import optparse


## TODO: if inputfile is a .tar.gz: untar and load more files

usage = "Usage: load RR file"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputfile",
                  action="store", type="string",
                  dest="input_file", help="Input File")

parser.add_option("-o", "--outputdir",
                  action="store", type="string",
                  dest="output_dir", help="Output Dir")

parser.add_option("-s", "--sep",
                  action="store", type="string", default='\t',
                  dest="separator", help="Column separator")

parser.add_option("-n", "--name",
                  action="store", type="string", default='RR',
                  dest="colname", help="Name of RR column")

(options, args) = parser.parse_args()
INPUTFILE=options.input_file
OUTDIR=options.output_dir
SEP=options.separator
RR_NAME=options.colname

RRdata=ph.DataSeries()
RRdata.load_from_csv(INPUTFILE, sep=SEP, colname=RR_NAME)
