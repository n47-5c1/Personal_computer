# -*- coding: utf-8 -*-
"""
Purpose: A example to read and plot netcdf data 

Created on Tue June 25 2018
@author: Shan He
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

nc_file = Dataset("./WPSH_IO_WNP_ENSO_NAO_indexes.nc")

WPSHI = nc_file.variables["WPSHI"][:]
T = WPSHI.shape[0]
data = np.empty((T,5))
data[:,0] = WPSHI
data[:,1] = nc_file.variables["IOI"][:]
data[:,2] = nc_file.variables["WNPI"][:]
data[:,3] = nc_file.variables["ENSOI"][:]
a = nc_file.variables["NAOIN"][:]
b = nc_file.variables["NAOIS"][:]
data[:,4] = a - b

data_mask = np.zeros(data.shape)
for t in range(1, T+1):
    if (t % 73) >= 12 and (t % 73) <= 30:
        data_mask[t-1,:] = True

# Initialize dataframe object, specify time axis and variable names
var_names = ['WPSH', 'IO', 'WNP', 'ENSO','NAO']
dataframe = pp.DataFrame(data, mask=data_mask)

parcorr = ParCorr(significance='analytic', mask_type='xyz')
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)
results = pcmci.run_pcmci(tau_max=12, pc_alpha=0.03)

# Correct p-values
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

# Plotting
link_matrix = pcmci.return_significant_parents(pq_matrix=q_matrix,
                                               val_matrix=results['val_matrix'],
                                               alpha_level=0.03)['link_matrix']

tp.plot_graph(val_matrix=results['val_matrix'],
              link_matrix=link_matrix, var_names=var_names)
"""
tp.plot_time_series_graph(val_matrix=results['val_matrix'],
                          link_matrix=link_matrix, var_names=var_names)
"""
plt.savefig('./NWPAC.svg')