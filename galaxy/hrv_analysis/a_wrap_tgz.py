import tarfile
import os
import shutil
import sys
import optparse

DIR_SUFFIX = "_dir"

def reset_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def enter_wrapping_tgz():
    """
    Converts -i input and -o ouptput tgz files into directories using the same names
    """
    parser_w = optparse.OptionParser(usage="Wrapper: see the wrapped script (-'_w')")

    parser_w.add_option("-i", "--inputfile",
                        action="store", type="string",
                        dest="input_file", help="Input File")

    parser_w.add_option("-o", "--outputfile",
                        action="store", type="string",

                        dest="output_file", help="Output File")

    (o_w, args) = parser_w.parse_args()
    # mod 
    sys.argv[sys.argv.index(o_w.input_file)] += DIR_SUFFIX
    sys.argv[sys.argv.index(o_w.output_file)] += DIR_SUFFIX
    # chk dir (orig names + mod)
    reset_dir(o_w.input_file+DIR_SUFFIX)
    reset_dir(o_w.output_file+DIR_SUFFIX)
    # tar
    ff = tarfile.open(name=o_w.input_file)
    ff.extractall(o_w.input_file+DIR_SUFFIX)
    ff.close()


def exit_wrapping_tgz():
    """
    Converts -i input and -o ouptput directories into tgz files using the same names
    """
    parser_w = optparse.OptionParser(usage="Wrapper: see the wrapped script (-'_w')")

    parser_w.add_option("-i", "--inputfile",
                        action="store", type="string",
                        dest="input_file", help="Input File")

    parser_w.add_option("-o", "--outputfile",
                        action="store", type="string",
                        dest="output_file", help="Output File")

    (o_w, args) = parser_w.parse_args()

    # clean
    if os.path.exists(o_w.input_file):
        shutil.rmtree(o_w.input_file)
    # tar
    if os.path.exists(o_w.output_file):
        ff = tarfile.open(o_w.output_file[:-len(DIR_SUFFIX)], "w:gz")
        ff.add(o_w.output_file)
        ff.close()


def wrap_tgz(script):
    """
    This wrapper only converts -i input and -o ouptput tgz files into and from normal directories using the same names

    @param script: Name of the file containing the scrpt to be called.
    """
    if os.path.exists(script) and os.path.isfile(script):
        try:
            enter_wrapping_tgz()
            execfile(script)
        finally:
            exit_wrapping_tgz()
    else:
        sys.stderr.write('Wrapper error: wrapped script file {0} does not exist or is not a file.'.format(script))

