from ast import parse
import numpy as np

import geopandas as gpd
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os, pathlib, re, json

import time
import datetime as dt
from pandas._libs.missing import NAType
from pandas.core.algorithms import unique

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import rasterio, rasterio.features, rasterio.warp


def save_plt(plt, folderPath, name, ext):
  plt.savefig(os.path.join(folderPath, name + '.' + ext), format=ext)

def save_df(df, folderPath, name):
  df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  # df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format=None)
  # fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  # fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
  # fd.close()
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


def parse_lines_deaths(path):
  lines = open(path).readlines()
  lines = [list(filter(None, re.split('\t|\n|"|/',l))) for l in lines]
  for l in lines:
    if 'Suppressed'in l:
      l[l.index('Suppressed')] = '-1'
  lines = [[item for item in line if (item == '-1' or str.isnumeric(item))] for line in lines[1:lines.index(['---'])]]
  return [[l[0], l[1] + l[2], l[3]] for l in lines]

def deaths_by_date_geoid(title):
  linesAll = []
  if os.path.isdir(title):
    filenames = os.listdir(title)
    for filename in filenames:
      linesAll += parse_lines_deaths(os.path.join(title, filename))
  else:
    title += '.txt'
    linesAll = parse_lines_deaths(title)
  ret = pd.DataFrame(linesAll, columns=['GEOID', 'YYYYMM', 'deaths']).sort_values(by=['YYYYMM','GEOID']).reset_index(drop=True)
  ret['deaths'] = pd.to_numeric(ret['deaths'])
  return ret




# https://jcutrer.com/python/learn-geopandas-plotting-usmaps
# https://gis.stackexchange.com/questions/336437/colorizing-polygons-based-on-color-values-in-dataframe-column/

# https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal


if __name__ == "__main__":
  

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
  # print(stateMapData)
  t0 = timer_restart(t0, 'read stateMapData')

  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  # print(countyMapData)
  t0 = timer_restart(t0, 'read countyMapData')


  stateMapData['total_deaths'] = [0.0 for i in range(len(stateMapData))]
  stateMapData['county_unsuppressed_deaths'] = [0.0 for i in range(len(stateMapData))]
  # stateMapData['county_unsuppressed_ALAND'] = [0.0 for i in range(len(stateMapData))]


  countyMapData['total_deaths'] = [0.0 for i in range(len(countyMapData))]


  basisregionFile = os.path.join(shapefilesDir, 'basisregions', 'TENA.geo.json')
  # basisregionFile = os.path.join(usaDir, stateMapFile + '.geojson')
  # basisregionFile = os.path.join(usaDir, 'us_states', '48-TX-Texas' + '.geojson')
  with open(basisregionFile, 'r') as f:
    contents = json.load(f)
    basisregion = GeometryCollection( [shape(feature["geometry"]).buffer(0) for feature in contents['features'] ] )
  t0 = timer_restart(t0, 'read basisregion')






  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'

  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title
  deg = 0.1


  # shapeData = stateMapData
  # shapeData = countyMapData

  unit = 'total_deaths'




  # print(shapeData)
  # exit()

  
  countyData = deaths_by_date_geoid(os.path.join(cdcWonderDir, countyTitle)) # has missing rows
  # print(countyData)
  stateData = deaths_by_date_geoid(os.path.join(cdcWonderDir, stateTitle)) # no missing rows
  # print(stateData)
  t0 = timer_restart(t0, 'read deaths data')

  it_stateData, it_countyData = 0, 0
  it_stateMapData, it_countyMapData = 0, 0

  prevYYYYMM = next(stateData.itertuples()).YYYYMM

  stateCountyStartIndices = dict()
  stateIndices = dict()

  prevSTATEFP = '00'
  # stateMapData : add column for starting index of counties in each state (iterate through countyMapData)
  for county in countyMapData.itertuples():
    if prevSTATEFP != county.STATEFP:
      stateCountyStartIndices[county.STATEFP] = county.Index
      prevSTATEFP = county.STATEFP
  stateCountyStartIndices['length'] = len(countyMapData)

  prevSTATEFP = '00'
  for state in stateMapData.itertuples():
    if prevSTATEFP != state.STATEFP:
      stateIndices[state.STATEFP] = state.Index
      prevSTATEFP = state.STATEFP
  stateIndices['length'] = len(stateMapData)
  
  t0 = timer_restart(t0, 'stateCountyStartIndices, stateIndices')
  # exit()

  # given: sorted by 'YYYYMM','GEOID'
  
  for deaths_state in stateData.itertuples():
    if prevYYYYMM != deaths_state.YYYYMM: # new month
      prevYYYYMM = deaths_state.YYYYMM
      it_stateMapData = 0
    if deaths_state.GEOID == stateMapData.at[it_stateMapData, 'GEOID'] and deaths_state.deaths != -1:
      stateMapData.at[it_stateMapData, 'total_deaths'] += deaths_state.deaths
    it_stateMapData += 1

  # print(stateMapData)



  i = 0
  while i != len(countyData):
    if prevYYYYMM != countyData.YYYYMM[i]: # new month
      prevYYYYMM = countyData.YYYYMM[i]
      it_countyMapData = 0

    if countyData.GEOID[i] == countyMapData.at[it_countyMapData, 'GEOID'] and countyData.deaths[i] != -1:
      countyMapData.at[it_countyMapData, 'total_deaths'] += countyData.deaths[i]

      stateFP = stateIndices[countyData.GEOID[i][0:2]]
      
      stateMapData.at[stateFP, 'county_unsuppressed_deaths'] += countyData.deaths[i]
      # stateMapData.at[stateFP, 'county_unsuppressed_ALAND'] += countyData.deaths[i]

    elif countyData.GEOID[i] != countyMapData.at[it_countyMapData, 'GEOID']:
      it_countyMapData += 1
      continue
    it_countyMapData += 1
    i += 1


  t0 = timer_restart(t0, 'get total_deaths')


  # save_df(pd.DataFrame(stateData), outputDir, stateTitle)
  # save_df(pd.DataFrame(countyData), outputDir, countyTitle)




 
  
  print(countyMapData)
  # print(countyMapData.iloc[[i for i in range(len(countyMapData) -  100)]])
  print(stateMapData)
  exit()

  


  # while it_stateData != len(stateData):
  #   it_stateData += 1
  

    



  # uniqueStates = pd.unique(stateData['YYYYMM'].values.ravel('K'))
  # print(uniqueStates)
  # print( len(uniqueStates)*21*12 )
  # print( len(stateData) )



  # exit()


  
  # shapeData['LANDPERCENTAGE'] = np.divide(list(map(float, shapeData['ALAND'])), np.array(list(map(float, shapeData['ALAND']))) + np.array(list(map(float, shapeData['AWATER']))))
  # minUnit = np.min(shapeData['LANDPERCENTAGE'])
  # maxUnit = np.max(shapeData['LANDPERCENTAGE'])
  # norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # shapeData['color'] = [mapper.to_rgba(v) for v in shapeData['LANDPERCENTAGE']]

  shapeData = countyMapData
  pltTitle = countyTitle
  res = 2 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  cmap = 'YlOrRd'

  minUnit = np.min(shapeData[unit])
  maxUnit = np.max(shapeData[unit])
  norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  shapeData['color'] = [mapper.to_rgba(v) for v in shapeData[unit]]
  


  t0 = timer_restart(t0, 'color mapping')

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

    t0 = timer_restart(t0, 'create plot')

    # save_plt(plt,outputDir,'TENA_land_to_total_area_ratio', 'png')
    save_plt(plt,outputDir, pltTitle, 'png')
    

    t1 = timer_restart(t1, 'total time')

    # plt.show()











