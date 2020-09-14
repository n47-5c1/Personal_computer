# -*- coding: utf-8 -*-
"""
Purpose: A example to read and plot netcdf data 

Created on Tue July 27 2018
@author: Shan He
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from netCDF4 import Dataset

nc_file1 = Dataset("./WY_WNP_SA_RM_AUS_indexes_CTRL.nc")
nc_file2 = Dataset("./WY_WNP_SA_RM_AUS_indexes_NOTP.nc")
fig, ax = plt.subplots(1, 1)

WYI = nc_file1.variables["WYI"][:]
T = WYI.shape[0]
data = np.empty((T,6,2))
data[:,0,0] = WYI
data[:,1,0] = nc_file1.variables["WNPI"][:]
data[:,2,0] = nc_file1.variables["SAI"][:]
data[:,3,0] = nc_file1.variables["RM1I"][:]
data[:,4,0] = nc_file1.variables["RM2I"][:]
data[:,5,0] = nc_file1.variables["AUSI"][:]
data[:,0,1] = nc_file2.variables["WYI"][:]
data[:,1,1] = nc_file2.variables["WNPI"][:]
data[:,2,1] = nc_file2.variables["SAI"][:]
data[:,3,1] = nc_file2.variables["RM1I"][:]
data[:,4,1] = nc_file2.variables["RM2I"][:]
data[:,5,1] = nc_file2.variables["AUSI"][:]

parm1 = norm.fit(data[:,3,0])
rv1 = norm(parm1[0], parm1[1])
x1 = np.linspace(rv1.ppf(0.01), rv1.ppf(0.99), 100)
#ax.plot(x1, rv1.pdf(x1), 'k-', lw=2, label='CTRL')
#ax.hist(data[:,3,0], density=True, histtype='stepfilled', alpha=0.2)

parm0 = norm.fit(data[:,3,1])
rv0 = norm(parm0[0], parm0[1])
x0 = np.linspace(rv0.ppf(0.01), rv0.ppf(0.99), 100)
#ax.plot(x0, rv0.pdf(x0), 'r-', lw=2, label='NOTP')
"""
ax.plot(x1, 1.0 - rv0.sf(x1) / rv1.sf(x1), 'b-', lw=2, label='PN')
ax.plot(x1, 1.0 - (1 - rv1.sf(x1)) / (1 - rv0.sf(x1)),
        'r-', lw=2, label='PS')
"""
x = np.arange(1, 401)
c = 6.0
ax.plot(x, 1.0 - (1.0 - (1 - rv0.sf(c))**x) / (1 - (1 - rv1.sf(c))**x),
        'b-', lw=2, label='PN')
ax.plot(x, 1.0 - (1 - rv1.sf(c))**x / (1 - rv0.sf(c))**x,
        'r-', lw=2, label='PS')

ax.legend(loc='best', frameon=False)
plt.show()