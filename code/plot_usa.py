
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.font_manager as fm

import os, pathlib, re, json, sys, copy

from shapely.geometry import shape, GeometryCollection

import rasterio, rasterio.features, rasterio.warp

import importlib
fc = importlib.import_module('functions')



# https://jcutrer.com/python/learn-geopandas-plotting-usmaps
# https://gis.stackexchange.com/questions/336437/colorizing-polygons-based-on-color-values-in-dataframe-column/

# https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal

tickSpacings = [1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1, 1,2,5,10,50,100,200,500,1000,2000,5000,10000,20000,50000,100000]
if __name__ == "__main__":
  numArgs = len(sys.argv)
  maxMappedValue = None
  if numArgs == 5:
    maxMappedValue = float(sys.argv[4])
  isMaxMappedValueGiven = maxMappedValue is not None


  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')

  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')

  t0 = fc.timer_start()
  t1 = t0

  stateMapFile = 'cb_2019_us_state_500k'
  stateMapData = gpd.read_file(os.path.join(usaDir, stateMapFile, stateMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  # stateMapData = stateMapData[stateMapData.STATEFP <= '56']
  # print(stateMapData)
  # t0 = fc.timer_restart(t0, 'read stateMapData')

  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  # countyMapData = countyMapData[countyMapData.STATEFP <= '56']
  # print(countyMapData)
  # t0 = fc.timer_restart(t0, 'read countyMapData')


  basisregionFile = os.path.join(shapefilesDir, 'basisregions', 'TENA.geo.json')
  # basisregionFile = os.path.join(usaDir, stateMapFile + '.geojson')
  # basisregionFile = os.path.join(usaDir, 'us_states', '48-TX-Texas' + '.geojson')
  with open(basisregionFile, 'r') as f:
    contents = json.load(f)
    basisregion = GeometryCollection( [shape(feature["geometry"]).buffer(0) for feature in contents['features'] ] )
  # t0 = fc.timer_restart(t0, 'read basisregion')


  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title
  countySupEstTitle = countyTitle + ', suppressed estimates'

  deg = 0.1
  suppValString = '-1'
  unit = 'total_deaths'

  ext = 'hdf5'

  deathsData = fc.makeCountyFileGEOIDs(fc.read_df(cdcWonderDir, countySupEstTitle, ext))
  # deathsData = fc.makeCountyFileGEOIDs(fc.read_df(cdcWonderDir, countyTitle, ext))
  shapeData = countyMapData

  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData)
  # print(list(shapeData.GEOID) == list(deathsData.GEOID)) # True
  

  # deathsData = fc.makeCountyFileGEOIDs(fc.read_df(cdcWonderDir, stateTitle, ext))
  # shapeData = stateMapData

  # t0 = fc.timer_restart(t0, 'read deaths data')






  startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]
  # startYYYYMM, endYYYYMM = '199901', '201907'
  res = 2 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  cmap = 'YlOrRd'
  
  # END PARAMS
  

  cmap = copy.copy(cm.get_cmap(cmap))


  dates = sorted(i for i in deathsData if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
  

  # shapeData[unit] = np.sum(deathsData.loc[:, dates], axis=1)

  rateper = 1000000


  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'population')
  popData = fc.read_df(usCensusDir, 'TENA_county_pop_1999_2019', ext)
  popData = fc.clean_states_reset_index(popData) # does nothing
  popData = fc.county_changes_deaths_reset_index(popData) # does nothing
  deathsSum = np.sum(deathsData.loc[:, dates], axis=1)
  popSum = np.sum(popData.loc[:, dates], axis=1)
  unit = 'monthly_death_rate'
  shapeData[unit] = (deathsSum / popSum) * rateper

  maxMappedValue = maxMappedValue * rateper



  # print(deathsSum)
  # print(popSum)
  # print(shapeData[unit])

  # exit()

  # shapeData['LANDPERCENTAGE'] = np.divide(list(map(float, shapeData['ALAND'])), np.array(list(map(float, shapeData['ALAND']))) + np.array(list(map(float, shapeData['AWATER']))))
  # minUnit = np.min(shapeData['LANDPERCENTAGE'])
  # maxUnit = np.max(shapeData['LANDPERCENTAGE'])
  # norm = cl.Normalize(vmin=minUnit, vmax=maxUnit, clip=False) # clip=False is default
  # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # shapeData['color'] = [mapper.to_rgba(v) for v in shapeData['LANDPERCENTAGE']]


  minUnit = np.min(shapeData[unit])
  maxUnit = np.max(shapeData[unit])
  if maxMappedValue is None or not isMaxMappedValueGiven:
    maxMappedValue = maxUnit

  tickSpacing = fc.closest(tickSpacings, maxMappedValue/15)

  # maxUnit = 15
  norm = cl.Normalize(vmin=0, vmax=maxMappedValue, clip=False) # clip=False is default
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # shapeData['color'] = [mapper.to_rgba(v) for v in shapeData[unit]]

  # print(shapeData)
  

  pltTitle = sys.argv[3] + ' (' + startYYYYMM + '-' + endYYYYMM + ')' + '_' + "{:.3f}".format(maxUnit) + '_' + "{:.3f}".format(maxMappedValue)
  # pltTitle = countySupEstTitle + ' (' + startYYYYMM + '-' + endYYYYMM + ')'
  
  # t0 = fc.timer_restart(t0, 'color mapping')

  with plt.style.context(("seaborn", "ggplot")):
    shapeData.plot(column = unit, figsize=(18*res,10*res), edgecolor='black', linewidth=0.3*res, cmap = cmap)

    plt.xlabel("Longitude", fontsize=12*res)
    plt.ylabel("Latitude", fontsize=12*res)
    plt.xticks(fontsize=12*res)
    plt.yticks(fontsize=12*res)
    plt.title('Average Monthly Death Rate (per 1 million people) (' + startYYYYMM + '-' + endYYYYMM + ')', fontsize=12*res)

    xlim0 = plt.xlim()
    ylim0 = plt.ylim()

    bounds = basisregion.bounds
    # (minx, miny, maxx, maxy)

    plt.xlim((bounds[0] - deg, bounds[2] + deg))
    plt.ylim((bounds[1] - deg, bounds[3] + deg))

    levels1 = np.arange(0,maxMappedValue + tickSpacing,tickSpacing)

    # ticks = np.sort([minUnit] + list(levels1) + [maxUnit])
    ticks = np.sort(list(levels1))
    # print(ticks)
    ticks = sorted(list(set([maxMappedValue] + list(levels1) ))) # [minUnit, maxUnit] + 
    cb = plt.colorbar(mapper, ticks=ticks, drawedges=True, label='Average Monthly Death Rate (per 1 million people)', pad=0.001)
    cb.ax.yaxis.label.set_font_properties(fm.FontProperties(size=12*res))
    cb.ax.tick_params(labelsize=9*res)
    # plt.grid(False)

    # t0 = fc.timer_restart(t0, 'create plot')

    # # fc.save_plt(plt,outputDir,'TENA_land_to_total_area_ratio', 'png')
    fc.save_plt(plt,outputDir, pltTitle, 'png')

    # t1 = fc.timer_restart(t1, 'total time')

    # plt.show()











