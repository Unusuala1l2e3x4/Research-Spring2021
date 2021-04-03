import numpy as np

import json
import geopandas as gpd
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os
import pathlib
import sys

import time
import datetime as dt

import rasterio
import rasterio.features
import rasterio.warp

from shapely.geometry import shape



def tif_filenames(filenames):
  ret = []
  for filename in filenames:
    if '.tif' in filename:
      ret.append(filename)
  return ret

def save_plt(plt, folderPath, name, ext):
  plt.savefig(os.path.join(folderPath, name + '.' + ext), format=ext)

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
  minLon = max(rc[1][0], 0)
  maxLon = rc[1][1]
  minLat = max(rc[0][1], 0)
  maxLat = rc[0][0]
  return minLat, maxLat, minLon, maxLon


def is_mat_smaller(mat, bounds, fd):
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, fd.transform)
  return minLat == minLon == 0 and maxLat + 1 >= len(mat) and maxLon + 1 >= len(mat[0])



def bound_ravel(lats_1d, lons_1d, bounds, transform):
  # bounds = (-179.995 - buf,-54.845 - buf,179.995,69.845) # entire mat
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  lats_1d = lats_1d[minLat:maxLat]
  lons_1d = lons_1d[minLon:maxLon]
  X, Y = np.meshgrid(lons_1d, np.flip(lats_1d))
  return np.ravel(Y), np.ravel(X)



# in main.py:
  # # 1998, 2016
  # years = np.arange(1998, 2016 + 1)

  # for y in years:
  #   # run(dir, 'read_sedac_pm2-5.py', [y, y, 'YlOrRd', 'basisregions', 'TENA.geo.json', USAstates])
  #   run(dir, 'read_sedac_pm2-5.py', [y, y, 'YlOrRd', 'geo-countries', 'geo-countries-union.json', world])
  # run(dir, 'read_sedac_pm2-5.py', [2000, 2000, 'YlOrRd', 'basisregions', 'TENA.geo.json', USAstates])
  # run(dir, 'read_sedac_pm2-5.py', [2000, 2000, 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), '06-CA-California.geojson', USAcounties])
  # t0 = timer_restart(t0, '')
  # exit()


if __name__ == "__main__":
  numArgs = len(sys.argv)
  startYear, endYear = int(sys.argv[1]), int(sys.argv[2])
  cmap = sys.argv[3]
  geojsonDir = sys.argv[4]
  regionFile = sys.argv[5]
  mapFile = sys.argv[6]

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'read_pm2-5-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')

  filenames = os.listdir(os.path.join(pmDir, 'data_0.01'))
  points_in_region_filenames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  filenames = sorted(tif_filenames(filenames)) # same as in gfedDir_timesArea


  # PARAMS
  # cmap = 'YlOrRd'
  # regionFile = 'TENA.geo'
  unit = 'μm_m^-3'
  res = 3 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  
  # locmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  locmap = gpd.read_file(os.path.join(shapefilesDir, mapFile))

  # locmap = locmap.boundary


  lats0, lons0, points_in_shape, df, basisregion = None, None, None, None, None
  minLat, maxLat, minLon, maxLon = None, None, None, None
  lats_1d, lons_1d = None, None
  bounded_mat = None
  
  levels1, levels2 = [], []


  t0 = timer_start()
  t1 = t0

  # with open(os.path.join(basisregionsDir, regionFile + '.geo.json'), 'r') as f:
  with open(os.path.join(shapefilesDir, geojsonDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])

  # t0 = timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '.hdf5' in points_in_region_filenames:
    df = pd.read_hdf(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), key='points')
    lats0 = np.array(df['lat'])
    lons0 = np.array(df['lon'])
    # t0 = timer_restart(t0, 'load df')

  for filename in filenames:
    name = filename.split('.')[0]
    year = int(name.split('_')[-1])
    if year < startYear or year > endYear:
      continue

    print(filename)
    fd = rasterio.open(os.path.join(pmDir, 'data_0.01', filename))
    # print(fd.bounds)
    # print(fd.crs)
    # (lon index, lat index) 
    # print(fd.transform*(0,0))
    # print(fd.transform)

    minLat, maxLat, minLon, maxLon = get_bound_indices(basisregion.bounds, fd.transform)

    mat = fd.read(1)
    # t0 = timer_restart(t0, 'read tif')
    xy = fd.xy(range(len(mat)), range(len(mat)))
    lats_1d = np.array(xy[1])
    xy = fd.xy(range(len(mat[0])), range(len(mat[0])))
    lons_1d = np.array(xy[0])

    deg = np.abs(lats_1d[0] - lats_1d[1])

    if df is None and not is_mat_smaller(mat, basisregion.bounds, fd):
      lats0, lons0 = bound_ravel(lats_1d, lons_1d, basisregion.bounds, fd)

      df = pd.DataFrame()
      df['lat'] = lats0
      df['lon'] = lons0

      # print(df)

      # shapely stuff; get indices (impossible time contraint for all .01 TENA points - ~70 days)
      # df = df[are_points_inside(basisregion, df)]
      # print(df)
      # t0 = timer_restart(t0, 'make df')

      with pd.HDFStore(os.path.join(pmDir, 'points_in_region', regionFile + '.hdf5'), mode='w') as f:
        f.put('points', df, format='table', data_columns=True)
        f.close()

      # t0 = timer_restart(t0, 'save df')
    # print(df)

    lats_1d = lats_1d[minLat:maxLat]
    lons_1d = lons_1d[minLon:maxLon]

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]          # for contour plotting
    bounded_mat = np.where(bounded_mat < 0, 0, bounded_mat) # for contour plotting
    
    if df is not None:
      mat = np.ravel(bounded_mat)
      # t0 = timer_restart(t0, 'mat bound ravel')
      df[unit] = mat
      df = df[df[unit] > 0]
      # t0 = timer_restart(t0, 'df remove <= 0')

    minUnit = np.min(bounded_mat) if df is None else np.min(df[unit])
    maxUnit = np.max(bounded_mat) if df is None else np.max(df[unit])

    norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
    # df['color'] = [mapper.to_rgba(v) for v in df[unit]]
    # # t0 = timer_restart(t0, 'color mapping')

    with plt.style.context(("seaborn", "ggplot")):
      locmap.plot(figsize=(18*res,10*res),
                  color="white",
                  edgecolor = "black")

      plt.xlabel("Longitude", fontsize=7*res)
      plt.ylabel("Latitude", fontsize=7*res)
      plt.title(name + ' (' + unit + ')', fontsize=7*res)

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

      # t0 = timer_restart(t0, 'create plot')
      
      save_plt(plt, outputDir, regionFile + '_' + str(year) + '_' + '{:.3f}'.format(maxUnit) + '_' + utc_time_filename(), 'png')
      # t0 = timer_restart(t0, 'save outfiles')

      t1 = timer_restart(t1, 'total time')

      # plt.show()











