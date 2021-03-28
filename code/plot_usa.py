
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os, pathlib, re, json, sys

import time
import datetime as dt

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import rasterio, rasterio.features, rasterio.warp


def save_df(df, folderPath, name, ext):
  # print('save', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  elif ext == 'hdf5':
    df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format='fixed')
  # df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  # df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format=None)
  # fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  # fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
  # fd.close()
  # fd = h5py.File(os.path.join(folderPath, name + '.hdf5'),'w')
  # fd.create_dataset('data', data=df)
  # fd.close()

def read_df(folderPath, name, ext):
  # print('read', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    return pd.read_csv(os.path.join(folderPath, name + '.csv'))
  elif ext == 'hdf5':
    return pd.read_hdf(os.path.join(folderPath, name + '.hdf5'))

def is_in_dir(folderPath, name, ext):
  return name + '.' + ext in os.listdir(folderPath)

def save_plt(plt, folderPath, name, ext):
  plt.savefig(os.path.join(folderPath, name + '.' + ext), format=ext)



def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()


def countyGEOIDstring(geo):
  geo = str(geo)
  while len(geo) != 5:
    geo = '0' + geo
  return geo

def stateGEOIDstring(geo):
  geo = str(geo)
  while len(geo) != 2:
    geo = '0' + geo
  return geo

def makeCountyFileGEOIDs(df):
  df.GEOID = [countyGEOIDstring(item) for item in df.GEOID]
  return df

def makeStateFileGEOIDs(df):
  df.GEOID = [stateGEOIDstring(item) for item in df.GEOID]
  return df

def makeCountyFileGEOIDs_STATEFPs(df):
  df = makeCountyFileGEOIDs(df)
  df['STATEFP'] = [item[0:2] for item in df.GEOID]
  return df


def clean_states_reset_index(df):
  statefpExists = 'STATEFP' in df.keys()
  if not statefpExists:
    df['STATEFP'] = [item[0:2] for item in df.GEOID]
  df = df[df['STATEFP'] <= '56']
  df = df[df['STATEFP'] != '02']
  df = df[df['STATEFP'] != '15']
  if not statefpExists:
    del df['STATEFP']
  return df.reset_index(drop=True)

def county_changes_deaths_reset_index(df): # see code/write_county_month_pop_testing.txt
  dates = [i for i in df.keys() if i != 'GEOID' and i != 'STATEFP']

  # 51515 put into   51019 (2013)
  temp1 = df[df.GEOID == '51515']
  if len(temp1.GEOID) != 0:
    temp2 = df[df.GEOID == '51019']
    df.loc[temp2.index[0], dates] = df.loc[[temp1.index[0], temp2.index[0]], dates].sum()
    df = df.drop([temp1.index[0]])

  # 51560 put into   51005 (2013)
  temp1 = df[df.GEOID == '51560']
  if len(temp1.GEOID) != 0:
    temp2 = df[df.GEOID == '51005']
    df.loc[temp2.index[0], dates] = df.loc[[temp1.index[0], temp2.index[0]], dates].sum()
    df = df.drop([temp1.index[0]])

  # 46113 renamed to 46102 (2014)
  temp1 = df[df.GEOID == '46113']
  if len(temp1.GEOID) != 0:
    df.loc[temp1.index[0], 'GEOID'] = '46102'

  # 08014 created from parts of 08001, 08013, 08059, 08123 (2001)
  # data unavailable for it (no unsuppressed data).
    # Thus: divide pop by 4, add result to original 4
  temp1 = df[df.GEOID == '08014']
  if len(temp1.GEOID) != 0:
    df = df.drop([temp1.index[0]])

  return df.sort_values(by='GEOID').reset_index(drop=True)


# https://jcutrer.com/python/learn-geopandas-plotting-usmaps
# https://gis.stackexchange.com/questions/336437/colorizing-polygons-based-on-color-values-in-dataframe-column/

# https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal


if __name__ == "__main__":
  # numArgs = len(sys.argv)

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')

  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')

  t0 = timer_start()
  t1 = t0

  stateMapFile = 'cb_2019_us_state_500k'
  stateMapData = gpd.read_file(os.path.join(usaDir, stateMapFile, stateMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  stateMapData = stateMapData[stateMapData.STATEFP <= '56']
  # print(stateMapData)
  # t0 = timer_restart(t0, 'read stateMapData')

  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  countyMapData = countyMapData[countyMapData.STATEFP <= '56']
  # print(countyMapData)
  # t0 = timer_restart(t0, 'read countyMapData')


  basisregionFile = os.path.join(shapefilesDir, 'basisregions', 'TENA.geo.json')
  # basisregionFile = os.path.join(usaDir, stateMapFile + '.geojson')
  # basisregionFile = os.path.join(usaDir, 'us_states', '48-TX-Texas' + '.geojson')
  with open(basisregionFile, 'r') as f:
    contents = json.load(f)
    basisregion = GeometryCollection( [shape(feature["geometry"]).buffer(0) for feature in contents['features'] ] )
  # t0 = timer_restart(t0, 'read basisregion')


  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title
  countySupEstTitle = countyTitle + ', suppressed estimates'

  deg = 0.1
  suppValString = '-1'
  unit = 'total_deaths'

  ext = 'hdf5'

  deathsData = makeCountyFileGEOIDs(read_df(cdcWonderDir, countySupEstTitle, ext))
  # deathsData = makeCountyFileGEOIDs(read_df(cdcWonderDir, countyTitle, ext))
  # deathsData = makeCountyFileGEOIDs(read_df(cdcWonderDir, stateTitle, ext))
  # t0 = timer_restart(t0, 'read deaths data')

  startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]
  # startYYYYMM, endYYYYMM = '199901', '201907'

  shapeData = countyMapData

  pltTitle = sys.argv[3] + ' (' + startYYYYMM + '-' + endYYYYMM + ')'
  # pltTitle = countySupEstTitle + ' (' + startYYYYMM + '-' + endYYYYMM + ')'

  res = 2 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  cmap = 'YlOrRd'

  # END PARAMS


  shapeData = clean_states_reset_index(shapeData)
  shapeData = county_changes_deaths_reset_index(shapeData) # removes 08014
  # print(list(shapeData.GEOID) == list(deathsData.GEOID)) # True

  dates = sorted(i for i in deathsData if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
  # print(dates)

  # print(np.sum(deathsData.loc[:, dates], axis=0)) # totals for each YYYYM
  # print(np.sum(deathsData.loc[:, dates], axis=1)) # totals for each GEOID

  shapeData[unit] = np.sum(deathsData.loc[:, dates], axis=1)

  # shapeData['LANDPERCENTAGE'] = np.divide(list(map(float, shapeData['ALAND'])), np.array(list(map(float, shapeData['ALAND']))) + np.array(list(map(float, shapeData['AWATER']))))
  # minUnit = np.min(shapeData['LANDPERCENTAGE'])
  # maxUnit = np.max(shapeData['LANDPERCENTAGE'])
  # norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # shapeData['color'] = [mapper.to_rgba(v) for v in shapeData['LANDPERCENTAGE']]


  minUnit = np.min(shapeData[unit])
  maxUnit = np.max(shapeData[unit])
  # maxUnit = 15
  norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  shapeData['color'] = [mapper.to_rgba(v) for v in shapeData[unit]]
  


  # t0 = timer_restart(t0, 'color mapping')

  with plt.style.context(("seaborn", "ggplot")):
    shapeData.plot(figsize=(18*res,10*res),
                # color="white",
                color=shapeData['color'],
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

    levels1 = np.arange(0,maxUnit+1,(maxUnit/10))

    # ticks = np.sort([minUnit] + list(levels1) + [maxUnit])
    ticks = np.sort(list(levels1))
    # print(ticks)
    plt.colorbar(mapper, ticks=ticks)

    # t0 = timer_restart(t0, 'create plot')

    # save_plt(plt,outputDir,'TENA_land_to_total_area_ratio', 'png')
    save_plt(plt,outputDir, pltTitle, 'png')

    # t1 = timer_restart(t1, 'total time')

    # plt.show()











