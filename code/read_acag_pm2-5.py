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
import matplotlib.path as mplp
import xarray
import bottleneck as bn

import os, pathlib, sys, re

import time
import datetime as dt
import copy

import rasterio, rasterio.features, rasterio.warp

import netCDF4

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon, LineString, MultiLineString
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
  lats_1d = lats_1d[minLat:maxLat]
  lons_1d = lons_1d[minLon:maxLon]
  X, Y = np.meshgrid(lons_1d, lats_1d)
  return np.ravel(Y), np.ravel(X)

def boundary_to_mask(boundary, x, y):
  mpath = mplp.Path(boundary)
  X, Y = np.meshgrid(x, y)
  points = np.array((X.flatten(), Y.flatten())).T
  mask = mpath.contains_points(points).reshape(X.shape)
  return 
  


def rasterize_geoids(bounds, transform, shapeData, lats_1d, lons_1d): # https://stackoverflow.com/a/38095929
  df = pd.DataFrame()
  df['lat'], df['lon'] = bound_ravel(lats_1d, lons_1d, bounds, transform)

  geoidMat = np.empty((len(lats_1d), len(lons_1d)), dtype='<U20')

  for row in shapeData.itertuples():
    # if row.GEOID != '53051': # test
    #   continue
    # print(row.GEOID,row.NAME)

    if row.geometry.boundary.geom_type == 'LineString':
      minLat, maxLat, minLon, maxLon = get_bound_indices(row.geometry.boundary.bounds, transform)
      if minLat == maxLat or minLon == maxLon:
        continue
      mask = boundary_to_mask(row.geometry.boundary, lons_1d[minLon:maxLon], lats_1d[minLat:maxLat])
      mask = np.where(mask, row.GEOID, '')
      geoidMat[minLat:maxLat,minLon:maxLon] = np.char.add(geoidMat[minLat:maxLat,minLon:maxLon], mask) # https://numpy.org/doc/stable/reference/routines.char.html#module-numpy.char
    else:
      for linestring in row.geometry.boundary:
        minLat, maxLat, minLon, maxLon = get_bound_indices(linestring.bounds, transform)
        if minLat == maxLat or minLon == maxLon:
          continue
        mask = boundary_to_mask(linestring, lons_1d[minLon:maxLon], lats_1d[minLat:maxLat])
        mask = np.where(mask, row.GEOID, '')
        geoidMat[minLat:maxLat,minLon:maxLon] = np.char.add(geoidMat[minLat:maxLat,minLon:maxLon], mask)  

    # break # test
  
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  # print(minLat, maxLat, minLon, maxLon)

  df['GEOID'] = np.ravel(geoidMat[minLat:maxLat,minLon:maxLon])
  print(df[df['GEOID'] != ''])
  
  return df



def overlaps_df(df):
  pairs = set()
  for row in df.itertuples():
    l = len(row.GEOID)
    if l <= 5:
      continue
    items = list( dict.fromkeys([row.GEOID[j:j+5] for j in range(0,l,5)]) )
    pairs.add(tuple(items))
  print(sorted(pairs), len(pairs))





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
  res = 3 # shapeData.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  ext = 'hdf5'
  testing = False


  # shapeData = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  shapeData = gpd.read_file(os.path.join(shapefilesDir, mapFile)).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData) # removes 08014
  # print(shapeData)

  # exit()


  points_in_shape, df, basisregion = None, None, None
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

  t0 = fc.timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '.hdf5' in points_in_region_filenames:
    # df = pd.read_hdf(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), key='points')
    df = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')

    if testing:
        
      overlaps = [i for i in df['GEOID'] if len(i)>5]
      print(set(overlaps))

      a1 = set([i[0:5] for i in overlaps])
      a2 = set([i[5:10] for i in overlaps])
      a3 = set([i[10:15] for i in overlaps])
      a4 = set([i[15:] for i in overlaps])

      print(a1)
      print(a2)
      print(a3)
      print(a4)
      
      u = set().union(a1, a2, a3, a4)
      
      s51 = [i[2:] for i in list(u) if i[:2] == '51']
      s08 = [i[2:] for i in list(u) if i[:2] == '08']
      s13 = [i[2:] for i in list(u) if i[:2] == '13']
      
      print(sorted(s08)) # ['005', '031'] - Arapahoe, City and County of Denver (all in pop data, all in deaths data)
      print(sorted(s13)) # ['249', '269'] - Schley, Taylor (all in pop data, all in deaths data)
      print(sorted(s51)) # many independent cities overlapping with counties (all in pop data, all in deaths data)
      # ['003', '005', '015', '059', '069', '081', '089', '153', '161', '163', '165', '195', '530', '540', '580', '595', '600', '660', '678', '683', '685', '690', '720', '770', '775', 
      # '790', '820', '840'] 

    overlaps_df(df)

    t0 = fc.timer_restart(t0, 'load df')
  exit()

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
  
    t0 = fc.timer_restart(t0, 'get transform')
    xy = rasterio.transform.xy(transform, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
    lats_1d = np.array(xy[1])
    xy = rasterio.transform.xy(transform, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
    lons_1d = np.array(xy[0])
    

    if df is None and not is_mat_smaller(mat, basisregion.bounds, transform):
      df = rasterize_geoids(basisregion.bounds, transform, shapeData, lats_1d, lons_1d)

      fc.save_df(df, os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
      fc.save_df(df, os.path.join(pmDir, 'points_in_region'), regionFile, 'csv') # helper
      
      t0 = fc.timer_restart(t0, 'save df')
    # exit()
    lats_1d = lats_1d[minLat:maxLat]
    lons_1d = lons_1d[minLon:maxLon]
    # print(lats_1d.shape, lons_1d.shape)
    # exit()

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]          # for contour plotting
    bounded_mat = np.where(bounded_mat < 0, 0, bounded_mat) # for contour plotting


    if df is not None:
      df[unit] = np.ravel(bounded_mat) # dont remove zero values; not many zeros, will only cause problems
      t0 = fc.timer_restart(t0, 'bounded_mat_raveled')


    minUnit = np.nanmin(bounded_mat) if df is None else np.nanmin(df[unit])
    maxUnit = np.nanmax(bounded_mat) if df is None else np.nanmax(df[unit])

    # print(maxUnit)

    norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
    # df['color'] = [mapper.to_rgba(v) for v in df[unit]]
    # t0 = fc.timer_restart(t0, 'color mapping')

    with plt.style.context(("seaborn", "ggplot")):
      shapeData.plot(figsize=(18*res,10*res),
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
      levels1 = np.arange(0,maxUnit,4)
      plt.contourf(lons_1d, lats_1d, bounded_mat, levels=levels1, cmap=cmap, alpha=0.6)

      # ms = marker_size(plt, xlim0, ylim0, deg)
      # print(ms)
      # plt.scatter(df.lon, df.lat, s=ms, c='red', alpha=1, linewidths=0, marker='s')
      # plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')

      ticks = np.sort([minUnit] + list(levels1) + [maxUnit])
      # print(ticks)
      plt.colorbar(mapper, ticks=ticks)

      t0 = fc.timer_restart(t0, 'create plot')
      
      fc.save_plt(plt, outputDir, regionFile + '_' + startend[0] + '_' + startend[1] + '_' + '{:.3f}'.format(maxUnit) + '_' + fc.utc_time_filename(), 'png')
      t0 = fc.timer_restart(t0, 'save outfiles')

      # plt.show()
  t1 = fc.timer_restart(t1, 'total time')

      











