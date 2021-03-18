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

import time
import datetime as dt

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import rasterio
import rasterio.features
import rasterio.warp



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



def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()



# https://jcutrer.com/python/learn-geopandas-plotting-usmaps
# https://gis.stackexchange.com/questions/336437/colorizing-polygons-based-on-color-values-in-dataframe-column/

# https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal


if __name__ == "__main__":
  

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  # outputDir = os.path.join(pPath, 'read_pm2-5-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')


  t0 = timer_start()
  t1 = t0


  allStatesFile = 'cb_2019_us_state_500k'
  allStatesData = gpd.read_file(os.path.join(usaDir, allStatesFile, allStatesFile + '.shp'))
  # print(allStatesData)
  t0 = timer_restart(t0, 'read allStatesData')

  allCountiesFile = 'cb_2019_us_county_500k'
  allCountiesData = gpd.read_file(os.path.join(usaDir, allCountiesFile, allCountiesFile + '.shp'))
  # print(allCountiesData)
  t0 = timer_restart(t0, 'read allCountiesData')


  basisregionFile = os.path.join(shapefilesDir, 'basisregions', 'TENA.geo.json')
  # basisregionFile = os.path.join(usaDir, allStatesFile + '.geojson')
  # basisregionFile = os.path.join(usaDir, 'us_states', '48-TX-Texas' + '.geojson')
  with open(basisregionFile, 'r') as f:
    contents = json.load(f)
    basisregion = GeometryCollection( [shape(feature["geometry"]).buffer(0) for feature in contents['features'] ] )


  t0 = timer_restart(t0, 'read basisregion')

  # locmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  # locmap = gpd.read_file(os.path.join(shapefilesDir, mapFile))
  # locmap = locmap.boundary

  res = 1 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  # unit = 'GEOID'
  cmap = 'YlOrRd'
  deg = 0.1

  # unit = 'STATEFP'
  # df = allStatesData

  unit = 'COUNTYFP'
  df = allCountiesData


  unit = 'LANDPERCENTAGE'
  
  df[unit] = np.divide(list(map(float, df['ALAND'])), np.array(list(map(float, df['ALAND']))) + np.array(list(map(float, df['AWATER']))))

  print(df[unit])

  minUnit = np.min(df[unit])
  maxUnit = np.max(df[unit])

  norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  df['color'] = [mapper.to_rgba(v) for v in df[unit]]


  t0 = timer_restart(t0, 'color mapping')

  with plt.style.context(("seaborn", "ggplot")):
    df.plot(figsize=(18*res,10*res),
                # color="white",
                color=df['color'],
                edgecolor = "black")

    plt.xlabel("Longitude", fontsize=7*res)
    plt.ylabel("Latitude", fontsize=7*res)
    # plt.title(name + ' (' + unit + ')', fontsize=7*res)

    xlim0 = plt.xlim()
    ylim0 = plt.ylim()


    bounds = basisregion.bounds
    # (minx, miny, maxx, maxy)


    plt.xlim((bounds[0] - deg, bounds[2] + deg))
    plt.ylim((bounds[1] - deg, bounds[3] + deg))

    levels1 = np.arange(0,maxUnit,(maxUnit/10))

    # ticks = np.sort([minUnit] + list(levels1) + [maxUnit])
    ticks = np.sort(list(levels1))
    # print(ticks)
    plt.colorbar(mapper, ticks=ticks)

    t0 = timer_restart(t0, 'create plot')
    

    t1 = timer_restart(t1, 'total time')

    plt.show()











