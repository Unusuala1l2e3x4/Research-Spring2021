
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os, pathlib, re, json, sys

from shapely.geometry import shape, GeometryCollection

import rasterio, rasterio.features, rasterio.warp

import importlib
fc = importlib.import_module('functions')



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
  # deathsData = fc.makeCountyFileGEOIDs(fc.read_df(cdcWonderDir, stateTitle, ext))
  # t0 = fc.timer_restart(t0, 'read deaths data')

  startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]
  # startYYYYMM, endYYYYMM = '199901', '201907'

  shapeData = countyMapData

  pltTitle = sys.argv[3] + ' (' + startYYYYMM + '-' + endYYYYMM + ')'
  # pltTitle = countySupEstTitle + ' (' + startYYYYMM + '-' + endYYYYMM + ')'

  res = 2 # locmap.plot figsize=(18*res,10*res); plt.clabel fontsize=3*res
  cmap = 'YlOrRd'

  # END PARAMS


  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData) # removes 08014
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
  


  # t0 = fc.timer_restart(t0, 'color mapping')

  with plt.style.context(("seaborn", "ggplot")):
    shapeData.plot(figsize=(18*res,10*res),
                # color="white",
                color=shapeData['color'],
                edgecolor = "black")

    plt.xlabel("Longitude", fontsize=7*res)
    plt.ylabel("Latitude", fontsize=7*res)
    plt.title(pltTitle)

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

    # t0 = fc.timer_restart(t0, 'create plot')

    # # fc.save_plt(plt,outputDir,'TENA_land_to_total_area_ratio', 'png')
    fc.save_plt(plt,outputDir, pltTitle, 'png')

    t1 = fc.timer_restart(t1, 'total time')

    # plt.show()











