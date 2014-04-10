__author__ = 'AleB'

from pyHRV.Files import *
from pyHRV.windowing import NamedWinGen, WindowsMapper
from pyHRV.indexes import Mean
from pyHRV import DataSeries

lab, ibi = load_excel_column('../z_data/ECG_nonfathers/suzuki.xlsx', 'Stimuli', 'IBI')
ds = DataSeries(ibi)
wg = NamedWinGen(ds, lab)
mm = WindowsMapper(ds, wg, Mean)
mm.compute_all()
print mm.results
print wg

