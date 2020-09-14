"""
Purpose: Caculate the East Asian monsoon index and output netcdf data

Created on September 09 2020
@author: Shan He
"""

from scipy import signal
import pandas
import xarray

nc_filename = 'E:/Data/NCEP-DOE/uwnd.mon.mean.nc'

nc_file = xarray.open_dataset(nc_filename)
uwnd_datetime = pandas.to_datetime(nc_file['time'].data)
where_year = [i.year >= 1979 and i.year <= 2019 for i in uwnd_datetime]
uwnd = nc_file['uwnd'][where_year,:,:,:]
print(uwnd)

#Calculate monthly anomalies and remove linear trend
clim = uwnd.groupby('time.month').mean('time')
anom = uwnd.groupby('time.month') - clim
signal.detrend(anom, axis=0, overwrite_data=True)
print(anom)

U850_S = anom.sel(lat=slice(15, 5), lon=slice(90, 130), level=850).mean(('lat', 'lon'))
U850_N = anom.sel(lat=slice(32.5, 22.5), lon=slice(110, 140), level=850).mean(('lat', 'lon'))
where_month = [i.month in [6,7,8] for i in uwnd_datetime[where_year]]
EASMI = U850_S[where_month].groupby('time.year').mean() - U850_N[where_month].groupby('time.year').mean()
print(EASMI)
EASMI.to_pandas().plot()