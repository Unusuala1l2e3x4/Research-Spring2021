import h5py
import numpy as np

import json
import geopandas
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import xarray
import bottleneck as bn

import os
import pathlib
import sys

import time
import datetime as dt
import copy

import rasterio
import rasterio.features
import rasterio.warp

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon


def tif_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.tif' in filename:
      ret.append(filename)
  return ret

def save_plt(plt, folderPath, name):
  plt.savefig(os.path.join(folderPath, name + '.png'))

def save_df(df, folderPath, name):
  # df.to_hdf(os.path.join(folderPath, filename + '.hdf5'), key='data')

  fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
  fd.close()
  
  # fd = h5py.File(os.path.join(folderPath, name + '.hdf5'),'w')
  # fd.create_dataset('data', data=df)
  # fd.close()

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
  

def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()


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

def geojson_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.geo' in filename:
      ret.append(filename)
  return ret

# def are_points_inside(basisregion, df):
#   return [ basisregion.contains(Point(row.lon, row.lat)) for row in df.itertuples() ]


def get_bound_indices(bounds, transform):
  buf = .001
  rc = rasterio.transform.rowcol(transform, [bounds[0] - buf, bounds[2]], [bounds[1] - buf, bounds[3]], op=round, precision=3)
  minLon = rc[1][0]
  maxLon = rc[1][1]
  minLat = rc[0][1]
  maxLat = rc[0][0]
  return minLat, maxLat, minLon, maxLon


def bound_ravel(lats_1d, lons_1d, bounds, transform):
  # bounds = (-179.995 - buf,-54.845 - buf,179.995,69.845) # entire mat
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  lats_1d = np.flip(lats_1d[minLat:maxLat])
  lons_1d = lons_1d[minLon:maxLon]
  return np.ravel(np.rot90(np.matrix([lats_1d for i in lons_1d]))), np.ravel(np.matrix([lons_1d for i in lats_1d]))


world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

if __name__ == "__main__":

  numArgs = len(sys.argv)
  startYear, endYear = int(sys.argv[1]), int(sys.argv[2])
  cmap = sys.argv[3]
  regionFile = sys.argv[4].split('.')[0]

  # cmap = 'YlOrRd'
  # regionFile = 'TENA.geo'

  unit = 'μm_m^-3'
  
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'read_pm2-5-outfiles')
  basisregionsDir = os.path.join(pPath, 'shapefiles', 'basisregions')

  filenames = os.listdir(os.path.join(pmDir, 'data'))
  points_in_region_filenames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  filenames = sorted(tif_filenames(filenames)) # same as in gfedDir_timesArea

  lats0, lons0, points_in_shape, df, basisregion = None, None, None, None, None
  minLat, maxLat, minLon, maxLon = None, None, None, None
  lats_1d, lons_1d = None, None
  bounded_mat = None
  
  levels1, levels2 = [], []


  t0 = timer_start()
  t1 = t0

  with open(os.path.join(basisregionsDir, regionFile + '.geo.json'), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])
  t0 = timer_restart(t0, 'read basisregion')

  if regionFile + '.hdf5' in points_in_region_filenames:
    df = pd.read_hdf(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), key='points')
    lats0 = np.array(df['lat'])
    lons0 = np.array(df['lon'])
    t0 = timer_restart(t0, 'load df')


  for filename in filenames:
    name = filename.split('.')[0]
    year = int(name.split('_')[-1])
    if year < startYear or year > endYear:
      continue

    print(filename)
    fd = rasterio.open(os.path.join(pmDir, 'data', filename))
    # print(fd.bounds)
    # print(fd.crs)
    # (lon index, lat index) 
    # print(fd.transform*(0,0))
    # print(fd.transform)

    mat = fd.read(1)
    t0 = timer_restart(t0, 'read tif')
    xy = fd.xy(range(len(mat)), range(len(mat)))
    lats_1d = np.array(xy[1])
    xy = fd.xy(range(len(mat[0])), range(len(mat[0])))
    lons_1d = np.array(xy[0])

    if df is None:
      lats0, lons0 = bound_ravel(lats_1d, lons_1d, basisregion.bounds, fd.transform)

      df = pd.DataFrame()
      df['lat'] = lats0
      df['lon'] = lons0

      # shapely stuff; get indices (impossible time contraint for all TENA points - ~70 days)
      # df = df[are_points_inside(basisregion, df)]
      # print(df)

      t0 = timer_restart(t0, 'make df')

      with pd.HDFStore(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), mode='w') as f:
        f.put('points', df, format='table', data_columns=True)
        f.close()

      t0 = timer_restart(t0, 'save df')


    # print(df)

    minLat, maxLat, minLon, maxLon = get_bound_indices(basisregion.bounds, fd.transform)
    
    lats_1d = lats_1d[minLat:maxLat]
    lons_1d = lons_1d[minLon:maxLon]

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]
    bounded_mat


    mat = np.ravel(bounded_mat)
    t0 = timer_restart(t0, 'mat bound ravel')


    df[unit] = mat
    df = df[df[unit] > 0]
    t0 = timer_restart(t0, 'make df')
    # print(df, np.shape(df))
    # print(np.max(df[unit]))
    # print(np.min(df[unit]))
    # print(np.average(df[unit]))

    # exit()

    color_min = np.min(df[unit])
    color_max = np.max(df[unit])
    norm = cl.Normalize(vmin=color_min, vmax=color_max, clip=False) # clip=False is default
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
    # df['color'] = [mapper.to_rgba(v) for v in df[unit]]
    # t0 = timer_restart(t0, 'color mapping')

    with plt.style.context(("seaborn", "ggplot")):
      world.plot(figsize=(18,10),
                  color="white",
                  edgecolor = "grey")

      plt.xlabel("Longitude")
      plt.ylabel("Latitude")
      plt.title(filename + ' (' + unit + ')')

      # plt.xlim((-180,180)) # default
      # plt.ylim((-90,90)) # default

      xlim0 = plt.xlim()
      ylim0 = plt.ylim()

      plt.xlim((min(df.lon) - .01, max(df.lon) + .01))
      plt.ylim((min(df.lat) - .01, max(df.lat) + .01))

      ms = marker_size(plt, xlim0, ylim0, 0.01)
      # print(ms)


      # contours = 

      levels1 = np.arange(np.min(df[unit]),np.max(df[unit]),0.15)
      plt.contour(lons_1d, lats_1d, bounded_mat, levels=levels1, cmap=cmap, linewidths=0.4, alpha=0.8)

      levels2 = [5,7.5,10,12.5,15,17.5,20]
      plt.contour(lons_1d, lats_1d, bounded_mat, levels=levels2, colors='black', linewidths=0.2, alpha=1)

      # labels = plt.clabel(contours, fontsize=10)
      # print(np.array(labels))

      # plt.scatter(df.lon, df.lat, s=ms, c='red', alpha=1, linewidths=0, marker='s')
      # plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')
      plt.colorbar(mapper)

      t0 = timer_restart(t0, 'create plot')

      # save_plt(plt, outputDir, name + '_' + str(len(levels1)) + '_' + str(len(levels2)) + '_' + utc_time_filename())
      save_plt(plt, outputDir, name + '_' + '-'.join([str(i) for i in levels2]))
      # save_df_plt(plt, df, hist_month, hist_year, cldf, stats, outputDir, title + '-' + utc_time_filename())
      t0 = timer_restart(t0, 'save outfiles')

      t1 = timer_restart(t1, 'total time')

      # plt.show()











