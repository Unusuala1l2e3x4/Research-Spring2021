import h5py
import numpy as np
from numpy import cos, sin, arctan2, arccos

import json
import geopandas as gpd
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import xarray
import bottleneck as bn

import os, pathlib, sys, re

import time
import datetime as dt
import copy

import rasterio, rasterio.features, rasterio.warp

import netCDF4

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import importlib
fc = importlib.import_module('functions')

WGS84_RADIUS = 6378137
WGS84_RADIUS_SQUARED = WGS84_RADIUS**2
d2r = np.pi/180

def greatCircleBearing(lon1, lat1, lon2, lat2):
    dLong = lon1 - lon2
    s = cos(d2r*lat2)*sin(d2r*dLong)
    c = cos(d2r*lat1)*sin(d2r*lat2) - sin(lat1*d2r)*cos(d2r*lat2)*cos(d2r*dLong)
    return arctan2(s, c)

def quad_area(lat, lon, deg):
  deg = deg / 2
  lons = [lon+deg,lon+deg,lon-deg,lon-deg]
  lats = [lat+deg,lat-deg,lat-deg,lat+deg]
  N = 4 # len(lons)
  angles = np.empty(N)
  for i in range(N):
      phiB1, phiA, phiB2 = np.roll(lats, i)[:3]
      lB1, lA, lB2 = np.roll(lons, i)[:3]
      # calculate angle with north (eastward)
      beta1 = greatCircleBearing(lA, phiA, lB1, phiB1)
      beta2 = greatCircleBearing(lA, phiA, lB2, phiB2)
      # calculate angle between the polygons and add to angle array
      angles[i] = arccos(cos(-beta1)*cos(-beta2) + sin(-beta1)*sin(-beta2))
  return (np.sum(angles) - (N-2)*np.pi)*WGS84_RADIUS_SQUARED


# def save_df(df, folderPath, name):
#   # df.to_hdf(os.path.join(folderPath, filename + '.hdf5'), key='data')
#   fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
#   fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
#   fd.close()
#   # fd = h5py.File(os.path.join(folderPath, name + '.hdf5'),'w')
#   # fd.create_dataset('data', data=df)
#   # fd.close()

def save_tif(mat, fd, folderPath, name):
  # mat = np.where( mat < 0, 0, mat)
  with rasterio.open(
    os.path.join(folderPath, name + '.tif'),
    'w',
    driver='GTiff',
    height=mat.shape[0],
    width=mat.shape[1],
    count=1,
    dtype=mat.dtype,
    crs=fd.crs,
    transform=fd.transform,
  ) as fd2:
    fd2.write(mat, 1)
    fd2.close()
  

def flatten_list(regular_list):
  return [item for sublist in regular_list for item in sublist]

def lim_length(lim):
  return lim[1] - lim[0]

def marker_size(plt, xlim1, ylim1, deg):
  a = 0.4865063 * (deg / 0.25)
  # 0.4378557
  x = lim_length(plt.xlim())
  y = lim_length(plt.ylim())
  x1 = lim_length(xlim1)
  y1 = lim_length(ylim1)
  return (x1*y1*a) / (x*y)

def nearest_index(array, value):
    array = np.asarray(array)
    i = (np.abs(array - value)).argmin()
    # return array[idx]
    return i

def get_bound_indices(bounds, transform):
  buf = .001
  rc = rasterio.transform.rowcol(transform, [bounds[0] - buf, bounds[2]], [bounds[1] - buf, bounds[3]], op=round, precision=3)
  # (minx, miny, maxx, maxy)
  # print(rc)
  # print(rc[1][0])
  minLon = max(rc[1][0], 0)
  maxLon = rc[1][1]
  # print(rc[0][1])
  minLat = max(rc[0][1], 0)
  maxLat = rc[0][0]
  return minLat, maxLat, minLon, maxLon


def is_mat_smaller(mat, bounds, transform):
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  return minLat == minLon == 0 and maxLat + 1 >= len(mat) and maxLon + 1 >= len(mat[0])



def bound_ravel(lats_1d, lons_1d, bounds, transform):
  # bounds = (-179.995 - buf,-54.845 - buf,179.995,69.845) # entire mat
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  lats_1d = np.flip(lats_1d[minLat:maxLat])
  lons_1d = lons_1d[minLon:maxLon]
  return np.ravel(np.rot90(np.matrix([lats_1d for i in lons_1d]))), np.ravel(np.matrix([lons_1d for i in lats_1d]))


if __name__ == "__main__":
  numArgs = len(sys.argv)
  startDate, endDate = sys.argv[1], sys.argv[2]
  cmap = sys.argv[3]
  geojsonDir = sys.argv[4]
  regionFile = sys.argv[5]
  mapFile = sys.argv[6]
  isYearly = sys.argv[7] == 'True'

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  outputDir = os.path.join(pPath, 'read_acag_pm2-5-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  

  points_in_region_filenames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  filenames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]
  # exit()


  # PARAMS
  # cmap = 'YlOrRd'
  # regionFile = 'TENA.geo'
  unit = 'μm_m^-3'
  res = 3 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  ext = 'hdf5'
  
  # locmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  locmap = gpd.read_file(os.path.join(shapefilesDir, mapFile))
  # locmap = locmap.boundary

  lats0, lons0, points_in_shape, df, basisregion = None, None, None, None, None
  minLat, maxLat, minLon, maxLon = None, None, None, None
  lats_1d, lons_1d = None, None
  bounded_mat = None
  
  levels1, levels2 = [], []

  t0 = fc.timer_start()
  t1 = t0

  # with open(os.path.join(basisregionsDir, regionFile + '.geo.json'), 'r') as f:
  with open(os.path.join(shapefilesDir, geojsonDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])

  # t0 = fc.timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '.hdf5' in points_in_region_filenames:
    # df = pd.read_hdf(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), key='points')
    df = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
    lats0 = np.array(df['lat'])
    lons0 = np.array(df['lon'])
    # t0 = fc.timer_restart(t0, 'load df')


  for filename in filenames:
    startend = re.split('_|-',filename)[3:5]
    
    if startend[0] < startDate or startend[1] > endDate or (startend[0] == startend[1] and isYearly) or (startend[0] != startend[1] and not isYearly):
      continue
    
    fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), filename, 'nc') # http://unidata.github.io/netcdf4-python/
    # print(fd.variables.keys())
    # print(fd.dimensions['LON'].size)
    # print(fd.variables['PM25'][:])

    mat = fd.variables['PM25'][:]

    deg = np.average(np.abs(fd.variables['LON'][:-1] - fd.variables['LON'][1:]))
    # print(deg)

    # transform = rasterio.transform.from_origin(np.round(basisregion.bounds[0], 2), np.round(basisregion.bounds[3], 2), deg,deg)
    transform = rasterio.transform.from_origin(np.round(np.min(fd.variables['LON'][:]), 2), np.round(np.max(fd.variables['LAT'][:]), 2), deg,deg)
    minLat, maxLat, minLon, maxLon = get_bound_indices(basisregion.bounds, transform)
    # print(minLat, maxLat, minLon, maxLon)
  
    # t0 = fc.timer_restart(t0, 'read tif')
    xy = rasterio.transform.xy(transform, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
    lats_1d = np.array(xy[1])
    xy = rasterio.transform.xy(transform, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
    lons_1d = np.array(xy[0])

    if df is None and not is_mat_smaller(mat, basisregion.bounds, transform):
      lats0, lons0 = bound_ravel(lats_1d, lons_1d, basisregion.bounds, transform)

      df = pd.DataFrame()
      df['lat'] = lats0
      df['lon'] = lons0

      fc.save_df(df, os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
      
      # t0 = fc.timer_restart(t0, 'save df')

    lats_1d = lats_1d[minLat:maxLat]
    lons_1d = lons_1d[minLon:maxLon]

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]          # for contour plotting
    bounded_mat = np.where(bounded_mat < 0, 0, bounded_mat) # for contour plotting

    # print(bounded_mat)
    
    if df is not None:
      df[unit] = np.ravel(bounded_mat) # dont remove zero values; not many zeros, will only cause problems
      # t0 = fc.timer_restart(t0, 'bounded_mat_raveled')


    minUnit = np.nanmin(bounded_mat) if df is None else np.nanmin(df[unit])
    maxUnit = np.nanmax(bounded_mat) if df is None else np.nanmax(df[unit])

    # print(maxUnit)

    norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
    # df['color'] = [mapper.to_rgba(v) for v in df[unit]]
    # # t0 = fc.timer_restart(t0, 'color mapping')

    with plt.style.context(("seaborn", "ggplot")):
      locmap.plot(figsize=(18*res,10*res),
                  color="white",
                  edgecolor = "black")

      plt.xlabel("Longitude", fontsize=7*res)
      plt.ylabel("Latitude", fontsize=7*res)
      plt.title(filename, fontsize=7*res)
      # plt.title(filename + ' (' + unit + ')', fontsize=7*res)

      xlim0 = plt.xlim()
      ylim0 = plt.ylim()

      plt.xlim((np.min(lons_1d) - deg, np.max(lons_1d) + deg))
      plt.ylim((np.min(lats_1d) - deg, np.max(lats_1d) + deg))

      ## contour lines
      # levels2 = np.arange(0,25,2)
      # levels2 = np.arange(0,maxUnit,2)
      # levels2 = np.arange(0,maxUnit,4)
      # contours = plt.contour(lons_1d, lats_1d, bounded_mat, levels=levels2, colors='white', linewidths=0, alpha=1)
      # labels = plt.clabel(contours, fontsize=3*res, colors='black')

      ## contour filler
      # levels1 = np.arange(minUnit,maxUnit,0.6)
      # levels1 = np.arange(0,25,0.2)
      # levels1 = np.arange(0,maxUnit,0.2)
      # levels1 = levels2
      levels1 = np.arange(0,maxUnit,4)
      plt.contourf(lons_1d, lats_1d, bounded_mat, levels=levels1, cmap=cmap, alpha=0.6)

      # ms = marker_size(plt, xlim0, ylim0, deg)
      # print(ms)
      # plt.scatter(df.lon, df.lat, s=ms, c='red', alpha=1, linewidths=0, marker='s')
      # plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')

      ticks = np.sort([minUnit] + list(levels1) + [maxUnit])
      # print(ticks)
      plt.colorbar(mapper, ticks=ticks)

      # t0 = fc.timer_restart(t0, 'create plot')
      
      fc.save_plt(plt, outputDir, regionFile + '_' + startend[0] + '_' + startend[1] + '_' + '{:.3f}'.format(maxUnit) + '_' + fc.utc_time_filename(), 'png')
      # fc.save_plt(plt, outputDir, name + '_' + regionFile + '_' + '-'.join([str(i) for i in levels2]))
      # t0 = fc.timer_restart(t0, 'save outfiles')


  t1 = fc.timer_restart(t1, 'total time')

      # plt.show()











