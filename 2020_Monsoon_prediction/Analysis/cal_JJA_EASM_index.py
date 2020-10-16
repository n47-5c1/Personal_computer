"""
Purpose: Caculate the JJA East Asian summer monsoon index and output netcdf data

Created on September 09 2020
@author: Shan He
"""

import numpy
from scipy import signal
import pandas
import xarray
import os
import sys

case = 0
if case == 0:
    in_filename = 'E:/Data/NCEP-DOE/uwnd.mon.mean.nc'
    out_filename = 'JJA_EASM_index_US.nc'
elif case == 1:
    in_filename = 'E:/Data/ERA5/u.850.mon.mean.nc'
    out_filename = 'JJA_EASM_index_RU.nc'
yS = 1979
yE = 2019

in_file = xarray.open_dataset(in_filename)
uwnd_datetime = pandas.to_datetime(in_file['time'].data)
where_year = [i.year >= yS and i.year <= yE for i in uwnd_datetime]
if case == 0:
    uwnd = in_file['uwnd'][where_year].sel(level=850)
elif case == 1:
    uwnd = in_file['u'][where_year].rename(dict(latitude='lat', longitude='lon'))
month_length = uwnd.time.dt.days_in_month
#print(uwnd)

#Calculate monthly anomalies and remove linear trend
clim = uwnd.groupby('time.month').mean('time')
anom = uwnd.groupby('time.month') - clim
signal.detrend(anom, axis=0, overwrite_data=True)
#print(anom)

#Calculate the JJA index (Wang and Fan 1999)
where_month = [i.month in [6,7,8] for i in uwnd_datetime[where_year]]
lon1_weights = numpy.cos(numpy.deg2rad(uwnd.lat.sel(lat=slice(15, 5))))
lon2_weights = numpy.cos(numpy.deg2rad(uwnd.lat.sel(lat=slice(32.5, 22.5))))
time_weights = month_length[where_month].groupby('time.year') / month_length[where_month].groupby('time.year').sum()
U850_S = anom[where_month].sel(lat=slice(15, 5), lon=slice(90, 130)).weighted(lon1_weights).mean(('lat', 'lon'))
U850_N = anom[where_month].sel(lat=slice(32.5, 22.5), lon=slice(110, 140)).weighted(lon2_weights).mean(('lat', 'lon'))
EASMI = ((U850_S - U850_N) * time_weights).groupby('time.year').sum()
EASMI.name = 'EASMI'
EASMI.attrs = uwnd.attrs
EASMI.to_pandas().plot()
#print(EASMI)

#Writing netCDF data
if os.path.exists(out_filename):
    os.remove(out_filename)
EASMI.to_netcdf(out_filename)