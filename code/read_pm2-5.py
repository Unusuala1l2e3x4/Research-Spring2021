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
  mat = np.where( mat < 0, 0, mat)
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


def marker_size(plt, xlim1, ylim1):
  a = 0.4865063 * 0.04
  # 0.4378557
  x = lim_length(plt.xlim())
  y = lim_length(plt.ylim())
  x1 = lim_length(xlim1)
  y1 = lim_length(ylim1)
  return (x1*y1*a) / (x*y)


world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'read_pm2-5-outfiles')

  filenames = os.listdir(pmDir)
  filenames = sorted(tif_filenames(filenames)) # same as in gfedDir_timesArea

  unit = 'μm_m^-3'
  cmap = 'YlOrRd'

  t0 = timer_start()
  t1 = t0

  lats0, lons0 = None, None


  for filename in filenames[18:]:
    print(filename)
    name = filename.split('.')[0]
    print(name)
    fd = rasterio.open(os.path.join(pmDir, filename))
    # print(fd.bounds)
    # print(fd.crs)
    # (lon index, lat index) 
    # print(fd.transform*(0,0))
    # print(fd.transform)

    mat = fd.read(1)
    # print(np.shape(mat))
    t0 = timer_restart(t0, 'read file')

    if lats0 is None or lons0 is None:
      xy = fd.xy(range(len(mat)), range(len(mat)))
      la = np.flip(xy[1])
      xy = fd.xy(range(len(mat[0])), range(len(mat[0])))
      lo = np.array(xy[0])

      lats0 = np.ravel(np.rot90(np.matrix([la for i in range(len(lo))])))
      lons0 = np.ravel([lo for i in range(len(la))])

      t0 = timer_restart(t0, 'make lats/lons')

    mat = np.ravel(mat)
    where_nonnero = np.where(mat <= 0)

    # remove <= 0 for plotting
    lats = np.delete(lats0, where_nonnero)
    lons = np.delete(lons0, where_nonnero)
    mat = np.delete(mat, where_nonnero)

    t0 = timer_restart(t0, 'remove <= 0')

    df = pd.DataFrame()
    df[unit] = mat
    df['lon'] = lons
    df['lat'] = lats
    
    t0 = timer_restart(t0, 'make df')

    # print(df)
    # exit()

    # color_min = min(df[unit])
    # color_max = max(df[unit])
    # norm = cl.Normalize(vmin=color_min, vmax=color_max, clip=False) # clip=False is default
    # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
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

      ms = marker_size(plt, xlim0, ylim0)
      # print(ms)

      plt.scatter(df.lon, df.lat, s=ms, c='red', alpha=1, linewidths=0, marker='s')
      # plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')
      # plt.colorbar(mapper)

      t0 = timer_restart(t0, 'create plot')

      save_plt(plt, outputDir, name)
      # save_df_plt(plt, df, hist_month, hist_year, cldf, stats, outputDir, title + '-' + utc_time_filename())
      t0 = timer_restart(t0, 'save outfiles')

      t1 = timer_restart(t1, 'total time')

      # plt.show()











