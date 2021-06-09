
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.font_manager as fm

import os, pathlib, json, sys, copy

from shapely.geometry import shape, GeometryCollection

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
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')

  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')



  t0 = fc.timer_start()
  t1 = t0



  climdivMapData = gpd.read_file(os.path.join(nClimDivDir, 'CONUS_CLIMATE_DIVISIONS', 'GIS.OFFICIAL_CLIM_DIVISIONS' + '.shp')).sort_values(by=['CLIMDIV']).reset_index(drop=True)

  print(climdivMapData)
  # climdivMapData['indices'] = list(climdivMapData.index)
  # exit()

  basisregionFile = os.path.join(shapefilesDir, 'basisregions', 'TENA.geo.json')
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



  shapeData = climdivMapData
  unit = 'CD_NEW'
  



  res = 2
  cmap = 'YlOrRd'
  
  # END PARAMS
  

  cmap = copy.copy(cm.get_cmap(cmap))


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
  

  # pltTitle = sys.argv[3] + ' (' + startYYYYMM + '-' + endYYYYMM + ')' + '_' + "{:.3f}".format(maxUnit) + '_' + "{:.3f}".format(maxMappedValue)
  # pltTitle = countySupEstTitle + ' (' + startYYYYMM + '-' + endYYYYMM + ')'
  
  # t0 = fc.timer_restart(t0, 'color mapping')

  with plt.style.context(("seaborn", "ggplot")):
    shapeData.plot(column = unit, figsize=(18*res,10*res), edgecolor='black', linewidth=0.3*res, cmap = cmap)

    plt.xlabel("Longitude", fontsize=7*res)
    plt.ylabel("Latitude", fontsize=7*res)
    plt.xticks(fontsize=7*res)
    plt.yticks(fontsize=7*res)
    # plt.title('Average Monthly Death Rate (per 1 million people) (' + startYYYYMM + '-' + endYYYYMM + ')', fontsize=7*res)

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
    cb = plt.colorbar(mapper, ticks=ticks, drawedges=True, label=unit, pad=0.001)
    cb.ax.yaxis.label.set_font_properties(fm.FontProperties(size=7*res))
    cb.ax.tick_params(labelsize=5*res)
    # plt.grid(False)

    # t0 = fc.timer_restart(t0, 'create plot')

    # # fc.save_plt(plt,outputDir,'TENA_land_to_total_area_ratio', 'png')
    # fc.save_plt(plt,outputDir, pltTitle, 'png')

    # t1 = fc.timer_restart(t1, 'total time')

    plt.show()











