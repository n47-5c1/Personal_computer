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
import collections
from matplotlib import pyplot
import cartopy
import gc
import os
import sys

in1_filename = 'JJA_EASM_index.nc'
in2_filename = 'E:/Data/ERSST.V5/sst.mnmean.nc'
in3_filename = 'E:/Data/NCEP-DOE/air.2m.mon.mean.nc'
in4_filename = 'E:/Data/NCEP-DOE/icec.sfc.mon.mean.nc'
in5_filename = 'E:/Data/NCEP-DOE/mslp.mon.mean.nc'
out1_filename = 'DJF_sst_precursor_region_JJA_EASM_index.nc'
out2_filename = 'DJF_air_precursor_region_JJA_EASM_index.nc'
out3_filename = 'DJF_mslp_precursor_region_JJA_EASM_index.nc'
yS = '1979-01-01'
yE = '2019-12-31'
a = 0.05
s = [[0,1,0], [1,1,1], [0,1,0]]
map_crs = cartopy.crs.PlateCarree(central_longitude=180)
data_crs = cartopy.crs.PlateCarree()

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

def corr_3D(x, y):
    #Embedded loop going through each grid point and calculating the correlation
    corr = numpy.zeros((y.lat.size,y.lon.size))
    pval = numpy.zeros((y.lat.size,y.lon.size))
    for i in range(y.lat.size):
        for j in range(y.lon.size):
            if y[:,i,j].notnull().all():
                corr[i,j], pval[i,j] = scipy.stats.pearsonr(x, y[:,i,j])
            else:
                corr[i,j] = float('nan')
                pval[i,j] = float('nan')
    return corr, pval

def roll_and_mask(labeled,N, ocean_mask=False, icec=None, lat_grid=None, lon_grid=None):
    #Mask oceanic and small (<=4) precursors
    global s
    if(ocean_mask):
        for i in range(3):
            land_array = labeled[i].where(numpy.logical_or(
                global_land_mask.globe.is_land(lat_grid, lon_grid),
                icec >= 0.1
            ), 0)
            ocean_array = labeled[i].where(numpy.logical_and(
                global_land_mask.globe.is_ocean(lat_grid, lon_grid),
                icec < 0.1
            ), 0)
            land_label = collections.Counter(numpy.array(land_array.stack(z=('lat','lon')).data))
            ocean_label = collections.Counter(numpy.array(ocean_array.stack(z=('lat','lon')).data))
            land_mask = []
            for j in range(1, int(labeled[i].max().data) + 1):
                if (land_label[j] > ocean_label[j] and land_label[j] > 4):
                    land_mask.append(j)
            labeled[i] = labeled[i].where(numpy.any(numpy.array(
                [labeled[i] == j for j in land_mask]
            ), axis=0), 0)
            #print(land_mask)
            del land_label, ocean_label, land_mask
            gc.collect()
        lon = labeled.lon[numpy.all(labeled.max(dim='lead') == 0, axis=0)]
        roll = numpy.argwhere(labeled.lon.data == lon[0].data)
        labeled_p = labeled.roll(lon=-int(roll), roll_coords=True)
        for i in range(3):
            labeled_p[i], N[i] = scipy.ndimage.label(labeled_p[i], s)
        labeled = labeled_p.roll(lon=int(roll), roll_coords=True)
    else:
        lon = labeled.lon[numpy.all(labeled.max(dim='lead') == 0, axis=0)]
        roll = numpy.argwhere(labeled.lon.data == lon[0].data)
        labeled_p = labeled.roll(lon=-int(roll), roll_coords=True)
        for i in range(3):
            label = collections.Counter(numpy.array(labeled[i].stack(z=('lat','lon')).data))
            mask = []
            for j in range(1, int(labeled[i].max().data) + 1):
                if label[j] > 4:
                    mask.append(j)
            labeled_p[i] = labeled_p[i].where(numpy.any(numpy.array(
                [labeled_p[i] == j for j in mask]
            ), axis=0), 0)
            labeled_p[i], N[i] = scipy.ndimage.label(labeled_p[i], s)
            #print(mask)
            del label, mask
            gc.collect()
        labeled = labeled_p.roll(lon=int(roll), roll_coords=True)
    return labeled, N

def cal_series(anom, labeled, mon,dim_name):
    #Create time series of the precursors
    N = int(labeled.max().data)
    monthtime = pandas.to_datetime(anom.time.data).month
    weights = numpy.cos(numpy.deg2rad(anom.lat))
    for i in range(1, N+1):
        if i == 1:
            y = anom[monthtime == mon].where(labeled == i).weighted(weights).mean(("lon", "lat"))
        else:
            if i == N:
                x = xarray.concat([y, anom[monthtime == mon].where(labeled == i).weighted(weights).mean(("lon", "lat"))], dim_name)
                del y
                y = xarray.DataArray(x.data, coords=[
                    (dim_name, range(1, N+1)),
                    ('year', pandas.to_datetime(x.time.data).year)
                ])
            else:
                x = xarray.concat([y, anom[monthtime == mon].where(labeled == i).weighted(weights).mean(("lon", "lat"))], dim_name)
                del y
                y = x
            del x
    return y

def plot_fig(corr, pval,labeled, N):
    global map_crs, data_crs
    #1
    fig1, axes1 = pyplot.subplots(3, 1, subplot_kw=dict(projection=map_crs))
    fig1.subplots_adjust(hspace=0.3)
    for i in range(3):
        cf = corr[i].plot.contourf(
            ax=axes1[i], add_colorbar=False, transform=data_crs,
            cmap='bwr', vmin=-0.5, vmax=0.5
        )
        pval[i].plot.contourf(
            ax=axes1[i], add_colorbar=False, levels=[0.0,0.05,1.0],
            hatches=['...',''], colors='None', transform=data_crs
        )
        axes1[i].set_title('')
        axes1[i].coastlines(linewidth=0.5)
        gl = axes1[i].gridlines(draw_labels=True)
        gl.top_labels = False
        #gl.xlines = False
        #gl.ylines = False
    cbar_ax = fig1.add_axes([0.7, 0.25, 0.02, 0.5])
    fig1.colorbar(cf, cax=cbar_ax)
    #2
    fig2, axes2 = pyplot.subplots(3, 1, subplot_kw=dict(projection=map_crs))
    fig2.subplots_adjust(hspace=0.3)
    for i in range(3):
        im = labeled[i].plot.pcolormesh(
            ax=axes2[i], add_colorbar=False, cmap='nipy_spectral_r',
            levels=range(N.max().data), transform=data_crs
        )
        axes2[i].set_title('')
        axes2[i].coastlines(linewidth=0.5)
        gl = axes2[i].gridlines(draw_labels=True)
        gl.top_labels = False
    cbar_ax = fig2.add_axes([0.7, 0.25, 0.02, 0.5])
    fig2.colorbar(im, cax=cbar_ax)
    #sst_Dec_mean.plot.line(x='time')

in1_file = xarray.open_dataset(in1_filename)
in2_file = xarray.open_dataset(in2_filename)
in3_file = xarray.open_dataset(in3_filename)
in4_file = xarray.open_dataset(in4_filename)
in5_file = xarray.open_dataset(in5_filename)
EASMI = in1_file['EASMI']
sst = in2_file['sst'].sel(time=slice(yS, yE))
sst_noice = sst.where(sst.data > -1.8)
air = in3_file['air'].sel(time=slice(yS, yE)).isel(level=0)
icec = in4_file['icec'].sel(time=slice(yS, yE)).max(dim='time')
lon_grid, lat_grid = numpy.meshgrid(air.lon.where(air.lon <= 180, air.lon - 360), air.lat)
mslp = in5_file['mslp'].sel(time=slice(yS, yE))
#print(EASMI)
#print(sst_noice)
#print(air)
#print(icec)
#print(mslp)

#Calculate monthly anomalies and remove linear trend
sst_anom = anom_dtrend(sst_noice)
air_anom = anom_dtrend(air)
mslp_anom = anom_dtrend(mslp)
#print(sst_anom)
#print(air_anom)
#print(mslp_anom)

#Compute correlation
del sst, sst_noice, air, mslp
gc.collect()
sst_corr = xarray.DataArray(
    numpy.zeros((3,sst_anom.lat.size,sst_anom.lon.size)),
    coords=[('lead', ['Dec','Jan','Feb']), ('lat', sst_anom.lat), ('lon', sst_anom.lon)]
)
air_corr = xarray.DataArray(
    numpy.zeros((3,air_anom.lat.size,air_anom.lon.size)),
    coords=[('lead', ['Dec','Jan','Feb']), ('lat', air_anom.lat), ('lon', air_anom.lon)]
)
mslp_corr = xarray.DataArray(
    numpy.zeros((3,mslp_anom.lat.size,mslp_anom.lon.size)),
    coords=[('lead', ['Dec','Jan','Feb']), ('lat', mslp_anom.lat), ('lon', mslp_anom.lon)]
)
sst_pval = sst_corr.copy()
air_pval = air_corr.copy()
mslp_pval = mslp_corr.copy()
datetime = pandas.to_datetime(sst_anom.time.data)
sst_corr[0], sst_pval[0] = corr_3D(
    EASMI[1:],
    sst_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
air_corr[0], air_pval[0] = corr_3D(
    EASMI[1:],
    air_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
mslp_corr[0], mslp_pval[0] = corr_3D(
    EASMI[1:],
    mslp_anom.isel(time=[i.year < datetime[-1].year and i.month == 12 for i in datetime])
)
for i in range(1,3):
    sst_corr[i], sst_pval[i] = corr_3D(EASMI, sst_anom[datetime.month == i])
    air_corr[i], air_pval[i] = corr_3D(EASMI, air_anom[datetime.month == i])
    mslp_corr[i], mslp_pval[i] = corr_3D(EASMI, mslp_anom[datetime.month == i])
#print(sst_corr)
#print(sst_pval)
#print(air_corr)
#print(air_pval)
#print(mslp_corr)
#print(mslp_pval)

#Obtain precursor regions and create their time series
labeled_sst = sst_pval.copy()
labeled_air = air_pval.copy()
labeled_mslp = mslp_pval.copy()
N_sst = xarray.DataArray(numpy.zeros(3, dtype=int), coords=[('lead', ['Dec','Jan','Feb'])])
N_air = N_sst.copy()
N_mslp = N_sst.copy()
for i in range(3):
    labeled_sst[i], N_sst[i] = scipy.ndimage.label(numpy.where(sst_pval[i] <= a, sst_pval[i], 0), s)
    labeled_air[i], N_air[i] = scipy.ndimage.label(numpy.where(air_pval[i] <= a, air_pval[i], 0), s)
    labeled_mslp[i], N_mslp[i] = scipy.ndimage.label(numpy.where(mslp_pval[i] <= a, mslp_pval[i], 0), s)
labeled_sst, N_sst = roll_and_mask(labeled_sst, N_sst)
labeled_air, N_air = roll_and_mask(labeled_air, N_air, True, icec, lat_grid, lon_grid)
labeled_mslp, N_mslp = roll_and_mask(labeled_mslp, N_mslp)
print(labeled_sst)
print(labeled_sst.max().data)
sst_Dec_mean = cal_series(sst_anom, labeled_sst[0], 12, 'var1')
sst_Jan_mean = cal_series(sst_anom, labeled_sst[1], 1, 'var2')
sst_Feb_mean = cal_series(sst_anom, labeled_sst[2], 2, 'var3')
air_Dec_mean = cal_series(air_anom, labeled_air[0], 12, 'var1')
air_Jan_mean = cal_series(air_anom, labeled_air[1], 1, 'var2')
air_Feb_mean = cal_series(air_anom, labeled_air[2], 2, 'var3')
mslp_Dec_mean = cal_series(mslp_anom, labeled_mslp[0], 12, 'var1')
mslp_Jan_mean = cal_series(mslp_anom, labeled_mslp[1], 1, 'var2')
mslp_Feb_mean = cal_series(mslp_anom, labeled_mslp[2], 2, 'var3')
#print(sst_Dec_mean)
#print(sst_Jan_mean)
#print(sst_Feb_mean)

#Writing netCDF data
sst_ds = xarray.Dataset(
    {
         'sst_corr': sst_corr, 'sst_pval': sst_pval, 'labeled_sst': labeled_sst, 'N_sst': N_sst,
         'sst_Dec_mean': sst_Dec_mean, 'sst_Jan_mean': sst_Jan_mean, 'sst_Feb_mean': sst_Feb_mean
    }    
)
air_ds = xarray.Dataset(
    {
         'air_corr': air_corr, 'air_pval': air_pval, 'labeled_air': labeled_air, 'N_air': N_air,
         'air_Dec_mean': air_Dec_mean, 'air_Jan_mean': air_Jan_mean, 'air_Feb_mean': air_Feb_mean
    }    
)
mslp_ds = xarray.Dataset(
    {
         'mslp_corr': mslp_corr, 'mslp_pval': mslp_pval, 'labeled_mslp': labeled_mslp, 'N_mslp': N_mslp,
         'mslp_Dec_mean': mslp_Dec_mean, 'mslp_Jan_mean': mslp_Jan_mean, 'mslp_Feb_mean': mslp_Feb_mean
    }    
)
#print(sst_ds)
#print(air_ds)
#print(mslp_ds)
if os.path.exists(out1_filename):
    os.remove(out1_filename)
if os.path.exists(out2_filename):
    os.remove(out2_filename)
if os.path.exists(out3_filename):
    os.remove(out3_filename)
sst_ds.to_netcdf(out1_filename)
air_ds.to_netcdf(out2_filename)
mslp_ds.to_netcdf(out3_filename)

'''
in1_file = xarray.open_dataset(out1_filename)
sst_corr = in1_file['sst_corr']
sst_pval = in1_file['sst_pval']
labeled_sst = in1_file['labeled_sst']
N_sst = in1_file['N_sst']
sst_Dec_mean = in1_file['sst_Dec_mean']
sst_Jan_mean = in1_file['sst_Jan_mean']
sst_Feb_mean = in1_file['sst_Feb_mean']
in2_file = xarray.open_dataset(out2_filename)
air_corr = in2_file['air_corr']
air_pval = in2_file['air_pval']
labeled_air = in2_file['labeled_air']
N_air = in2_file['N_air']
air_Dec_mean = in2_file['air_Dec_mean']
air_Jan_mean = in2_file['air_Jan_mean']
air_Feb_mean = in2_file['air_Feb_mean']
in3_file = xarray.open_dataset(out3_filename)
mslp_corr = in3_file['mslp_corr']
mslp_pval = in3_file['mslp_pval']
labeled_mslp = in3_file['labeled_mslp']
N_mslp = in3_file['N_mslp']
mslp_Dec_mean = in3_file['mslp_Dec_mean']
mslp_Jan_mean = in3_file['mslp_Jan_mean']
mslp_Feb_mean = in3_file['mslp_Feb_mean']

#Plot the results
#labeled_sst[0] = labeled_sst[0].where(numpy.any(numpy.array(
#    [labeled_sst[0].data == i for i in numpy.array([4])]#9 4
#), axis=0), 0)
#labeled_sst[1] = labeled_sst[1].where(numpy.any(numpy.array(
#    [labeled_sst[1].data == i for i in numpy.array([4])]#11 12 5 4
#), axis=0), 0)
#labeled_sst[2] = labeled_sst[2].where(labeled_sst[2] == 7, 0)
#labeled_mslp[0] = labeled_mslp[0].where(numpy.any(numpy.array(
#    [labeled_mslp[0].data == i for i in numpy.array([16])]# 18 3 16
#), axis=0), 0)
#labeled_mslp[1] = labeled_mslp[1].where(labeled_mslp[1] == 9, 0)
'''
plot_fig(sst_corr,sst_pval,labeled_sst,N_sst)
plot_fig(air_corr,air_pval,labeled_air,N_air)
plot_fig(mslp_corr,mslp_pval,labeled_mslp,N_mslp)