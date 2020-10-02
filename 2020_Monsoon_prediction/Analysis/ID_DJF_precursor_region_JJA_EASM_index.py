"""
Purpose: Identify DJF precursor regions of the JJA East Asian summer monsoon index
and output netCDF data

Created on September 15 2020
@author: Shan He
"""

import numpy
import scipy.signal
import scipy.stats
import scipy.ndimage
import pandas
import xarray
import global_land_mask.globe
from matplotlib import pyplot
import cartopy
import gc
import sys

def anom_dtrend(x):
    #Calculate monthly anomalies and remove linear trend
    clim = x.groupby('time.month').mean('time')
    anom = x.groupby('time.month') - clim
    if anom.notnull().all():
        scipy.signal.detrend(anom, axis=0, overwrite_data=True)
    else:
        xvals = numpy.linspace(0, anom.time.size, anom.time.size)
        for j in range(anom.lat.size):
            for k in range(anom.lon.size):
                slope, intercept, tmp, tmp, tmp = scipy.stats.linregress(xvals, anom[:,j,k])
                anom[:,j,k].values -= slope * xvals + intercept
    return anom

def corr_3D(a,b):
    # embedded loop going through each grid point and calculating the correlation
    corr = numpy.zeros((b.lat.size,b.lon.size))
    pval = numpy.zeros((b.lat.size,b.lon.size))
    for i in range(b.lat.size):
        for j in range(b.lon.size):
            if b[:,i,j].notnull().all():
                corr[i,j], pval[i,j] = scipy.stats.pearsonr(a, b[:,i,j])
            else:
                corr[i,j] = float('nan')
                pval[i,j] = float('nan')
    return corr, pval

in1_filename = 'JJA_EASM_index.nc'
in2_filename = 'E:/Data/ERSST.V5/sst.mnmean.nc'
in3_filename = 'E:/Data/NCEP-DOE/air.2m.mon.mean.nc'
in4_filename = 'E:/Data/NCEP-DOE/icec.sfc.mon.mean.nc'
in5_filename = 'E:/Data/NCEP-DOE/mslp.mon.mean.nc'
out_filename = 'DJF_precursor_region_JJA_EASM_index.nc'
yS = '1979-01-01'
yE = '2019-12-31'
map_crs = cartopy.crs.PlateCarree(central_longitude=180)
data_crs = cartopy.crs.PlateCarree()
s = [[0,1,0], [1,1,1], [0,1,0]]

in1_file = xarray.open_dataset(in1_filename)
in2_file = xarray.open_dataset(in2_filename)
#in3_file = xarray.open_dataset(in3_filename)
#in4_file = xarray.open_dataset(in4_filename)
#in5_file = xarray.open_dataset(in5_filename)
EASMI = in1_file['EASMI']
sst = in2_file['sst'].sel(time=slice(yS, yE))
sst_noice = sst.where(sst.data > -1.8)
'''
air = in3_file['air'].sel(time=slice(yS, yE)).isel(level=0)
icec = in4_file['icec'].sel(time=slice(yS, yE)).max(dim='time')
lon_grid, lat_grid = numpy.meshgrid(air.lon.where(air.lon <= 180, air.lon - 360), air.lat)
air_land = air.where(numpy.logical_or(
    global_land_mask.globe.is_land(lat_grid, lon_grid),
    icec >= 0.1
))
mslp = in5_file['mslp'].sel(time=slice(yS, yE))
'''
#print(EASMI)
#print(sst_noice)
#print(air_land)
#print(icec)
#print(mslp)

#Calculate monthly anomalies and remove linear trend
sst_anom = anom_dtrend(sst_noice)
#air_anom = anom_dtrend(air_land)
#mslp_anom = anom_dtrend(mslp)
#print(sst_anom)
#print(air_anom)
#print(mslp_anom)

#Compute correlation
del sst, sst_noice#, air, air_land, mslp
gc.collect()
sst_corr = xarray.DataArray(
    numpy.zeros((3,sst_anom.lat.size,sst_anom.lon.size)),
    coords=[('time', ['Dec','Jan','Feb']), ('lat', sst_anom.lat), ('lon', sst_anom.lon)]
)
sst_pval = sst_corr.copy()
'''
air_corr = xarray.DataArray(
    numpy.zeros((3,air_anom.lat.size,air_anom.lon.size)),
    coords=[('time', ['Dec','Jan','Feb']), ('lat', air_anom.lat), ('lon', air_anom.lon)]
)
air_pval = air_corr.copy()
mslp_corr = xarray.DataArray(
    numpy.zeros((3,mslp_anom.lat.size,mslp_anom.lon.size)),
    coords=[('time', ['Dec','Jan','Feb']), ('lat', mslp_anom.lat), ('lon', mslp_anom.lon)]
)
mslp_pval = mslp_corr.copy()
'''
datetime = pandas.to_datetime(sst_anom.time.data)
sst_corr[0], sst_pval[0] = corr_3D(
    EASMI[1:],
    sst_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
sst_corr[1], sst_pval[1] = corr_3D(EASMI, sst_anom[datetime.month == 1])
sst_corr[2], sst_pval[2] = corr_3D(EASMI, sst_anom[datetime.month == 2])
'''
air_corr[0], air_pval[0] = corr_3D(
    EASMI[1:],
    air_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
air_corr[1], air_pval[1] = corr_3D(EASMI, air_anom[datetime.month == 1])
air_corr[2], air_pval[2] = corr_3D(EASMI, air_anom[datetime.month == 2])
mslp_corr[0], mslp_pval[0] = corr_3D(
    EASMI[1:],
    mslp_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
mslp_corr[1], mslp_pval[1] = corr_3D(EASMI, mslp_anom[datetime.month == 1])
mslp_corr[2], mslp_pval[2] = corr_3D(EASMI, mslp_anom[datetime.month == 2])
'''
#print(sst_corr)
#print(sst_pval)
#print(air_corr)
#print(air_pval)
#print(mslp_corr)
#print(mslp_pval)
'''
fig_sst, axes_sst = pyplot.subplots(3, 1, subplot_kw=dict(projection=map_crs))
for i in range(3):
    sst_corr.isel(time=i).plot.contourf(ax=axes_sst[i], add_colorbar=False, transform=data_crs)
    sst_pval.isel(time=i).plot.contourf(ax=axes_sst[i], add_colorbar=False, levels=[0.0,0.05,1.0], hatches=['...',''], colors='None', transform=data_crs)
    axes_sst[i].coastlines()
    axes_sst[i].gridlines(draw_labels=True)
#pyplot.colorbar(axes_air[0])
'''
#Obtain precursor regions and create their time series
sst_PR = sst_pval.copy()
N_sst = []
weights_sst = numpy.cos(numpy.deg2rad(sst_anom.lat))
weights_sst.name = "weights"
#fig_sst, axes_sst = pyplot.subplots(3, 1, subplot_kw=dict(projection=map_crs))
#fig_sst.subplots_adjust(hspace=0.3)
for i in range(3):
    sst_PR[i], tmp = scipy.ndimage.label(numpy.where(sst_pval[i] <= 0.05, sst_pval[i], 0), s)
    N_sst.append(tmp)
    '''
    sst_PR.isel(time=i).plot.pcolormesh(ax=axes_sst[i], add_colorbar=False, levels=range(N_sst[i]), transform=data_crs)
    axes_sst[i].set_title('')
    axes_sst[i].coastlines()
    gl = axes_sst[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.xlines = False
    gl.ylines = False
    del gl
    '''
for i in range(1, N_sst[0]+1):
    if i == 1:
        sst_Dec_mean = sst_anom[datetime.month == 12].where(sst_PR[0] == i).weighted(weights_sst).mean(("lon", "lat"))
    else:
        if i == N_sst[0]:
            x = xarray.concat([sst_Dec_mean, sst_anom[datetime.month == 12].where(sst_PR[0] == i).weighted(weights_sst).mean(("lon", "lat"))], pandas.Index(range(1, N_sst[0]+1), name="var"))
        else:
            x = xarray.concat([sst_Dec_mean, sst_anom[datetime.month == 12].where(sst_PR[0] == i).weighted(weights_sst).mean(("lon", "lat"))], 'var')
        del sst_Dec_mean
        sst_Dec_mean = x
        del x
for i in range(1, N_sst[1]+1):
    if i == 1:
        sst_Jan_mean = sst_anom[datetime.month == 1].where(sst_PR[1] == i).weighted(weights_sst).mean(("lon", "lat"))
    else:
        if i == N_sst[1]:
            x = xarray.concat([sst_Jan_mean, sst_anom[datetime.month == 1].where(sst_PR[1] == i).weighted(weights_sst).mean(("lon", "lat"))], pandas.Index(range(1, N_sst[1]+1), name="var"))
        else:
            x = xarray.concat([sst_Jan_mean, sst_anom[datetime.month == 1].where(sst_PR[1] == i).weighted(weights_sst).mean(("lon", "lat"))], 'var')
        del sst_Jan_mean
        sst_Jan_mean = x
        del x
for i in range(1, N_sst[2]+1):
    if i == 1:
        sst_Feb_mean = sst_anom[datetime.month == 2].where(sst_PR[2] == i).weighted(weights_sst).mean(("lon", "lat"))
    else:
        if i == N_sst[2]:
            x = xarray.concat([sst_Feb_mean, sst_anom[datetime.month == 2].where(sst_PR[2] == i).weighted(weights_sst).mean(("lon", "lat"))], pandas.Index(range(1, N_sst[2]+1), name="var"))
        else:
            x = xarray.concat([sst_Feb_mean, sst_anom[datetime.month == 2].where(sst_PR[2] == i).weighted(weights_sst).mean(("lon", "lat"))], 'var')
        del sst_Feb_mean
        sst_Feb_mean = x
        del x
sst_Dec_mean.plot.line(x='time')
#print(sst_Dec_mean)
#print(sst_Jan_mean)
#print(sst_Feb_mean)

#Writing netCDF data
sst_ds = xarray.Dataset(
    {
         'sst_coor': sst_corr, 'sst_pval': sst_pval, 'sst_PR': sst_PR,
         'sst_Dec_mean': sst_Dec_mean, 'sst_Jan_mean': sst_Jan_mean, 'sst_Feb_mean': sst_Feb_mean
    },
    attrs=dict(compat='override')
)
print(sst_ds)
sst_ds.to_netcdf(out_filename)