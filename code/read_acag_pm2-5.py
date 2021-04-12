import numpy as np

import json
import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.font_manager as fm

import os, pathlib, sys, re
import copy
import rasterio, rasterio.features, rasterio.warp

from shapely.geometry import shape, MultiPoint

import importlib
fc = importlib.import_module('functions')





def overlaps_df(df):
  pairs = set()
  for row in df.itertuples():
    l = len(row.GEOID)
    if l <= 5:
      continue
    items = list( dict.fromkeys([row.GEOID[j:j+5] for j in range(0,l,5)]) ) # preserves order
    items += [round(row.lat, 3), round(row.lon, 3)]
    pairs.add(tuple(items))
  print(sorted(pairs), len(pairs))


  
contourSpacings = [0.01,0.02,0.05,0.1,0.2,0.5,1] # 2, 5, 10, 20    # may need to be adjusted in consideration
tickSpacings = [0.1,0.2,0.5,1,2,5,10]
if __name__ == "__main__":
  numArgs = len(sys.argv)
  startDate, endDate = sys.argv[1], sys.argv[2]
  cmap = sys.argv[3]
  regionDir = sys.argv[4]
  regionFile = sys.argv[5]
  mapFile = sys.argv[6]
  isYearly = sys.argv[7] == 'True'
  maxMappedValue = None
  if numArgs == 9:
    maxMappedValue = float(sys.argv[8])
  isMaxMappedValueGiven = maxMappedValue is not None

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  outputDir = os.path.join(pPath, 'read_acag_pm2-5-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  

  points_in_region_fnames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]
  # exit()


  # PARAMS
  unit = 'μm_m^-3'
  res = 3
  testing = False
  only_points_in_region = False
  save_maxVals = False
  # END PARAMS


  # shapeData = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  shapeData = gpd.read_file(os.path.join(shapefilesDir, mapFile)).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData) # removes 08014
  # print(shapeData)
  

  points_in_shape, df, basisregion, transform, deg = None, None, None, None, None
  minLat, maxLat, minLon, maxLon = None, None, None, None
  lats_1d, lons_1d = None, None
  bounded_mat = None
  
  levelsDiscrete, levelsContinuous = [], []

  t0 = fc.timer_start()
  t1 = t0

  # with open(os.path.join(basisregionsDir, regionFile + '.geo.json'), 'r') as f:
  with open(os.path.join(shapefilesDir, regionDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])
  contourSpacing = fc.closest(contourSpacings, fc.shortestAxisLength(basisregion.bounds)/30)
  # t0 = fc.timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '.hdf5' in points_in_region_fnames: # must use hdf5 - preserves indices
    df = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
    # t0 = fc.timer_restart(t0, 'load df')
    if testing:
      overlaps = [i for i in df['GEOID'] if len(i)>5]
      print(sorted(set(overlaps)))
      overlaps_df(df)
  # exit()


  maxVals = []
  dates = []

  for pm_fname in pm_fnames:
    startend = re.split('_|-',pm_fname)[3:5]
    if startend[0] < startDate or startend[1] > endDate or (startend[0] == startend[1] and isYearly) or (startend[0] != startend[1] and not isYearly):
      continue
    
    if save_maxVals:
      if isYearly:
        dates.append(startend[0][:4])
      else:
        dates.append(startend[0])

    fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), pm_fname, 'nc') # http://unidata.github.io/netcdf4-python/
    mat = fd.variables['PM25'][:]


    if transform is None or not isMaxMappedValueGiven: # minLat, maxLat, minLon, maxLon are also none
      deg = np.average(np.abs(fd.variables['LON'][:-1] - fd.variables['LON'][1:]))
      # print(deg)
      transform = rasterio.transform.from_origin(np.round(np.min(fd.variables['LON'][:]), 2), np.round(np.max(fd.variables['LAT'][:]), 2), deg,deg)
      minLat, maxLat, minLon, maxLon = fc.get_bound_indices(basisregion.bounds, transform)

      # t0 = fc.timer_restart(t0, 'get transform')
      xy = rasterio.transform.xy(transform, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
      lats_1d = np.array(xy[1])
      xy = rasterio.transform.xy(transform, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
      lons_1d = np.array(xy[0])
    
    if df is None and not fc.is_mat_smaller(mat, basisregion.bounds, transform):
      df = fc.rasterize_geoids_df(basisregion.bounds, transform, shapeData, lats_1d, lons_1d)
      # t0 = fc.timer_restart(t0, 'rasterize_geoids_df')
      fc.save_df(df, os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
      # t0 = fc.timer_restart(t0, 'save df hdf5')
      fc.save_df(df, os.path.join(pmDir, 'points_in_region'), regionFile, 'csv') # helper
      # t0 = fc.timer_restart(t0, 'save df csv')

    if regionFile + '.geojson' not in points_in_region_fnames and df is not None: # ~29 min for TENA.geojson
      geodf = copy.deepcopy(shapeData) 
      geodf.geometry = [ MultiPoint([(row.lon, row.lat) for row in df.loc[df.GEOID == geodf.GEOID[r], ['lon','lat']].itertuples()]) for r in range(len(geodf)) ]
      geodf.to_file(os.path.join(pmDir, 'points_in_region', regionFile + '.geojson'), driver='GeoJSON')
      # t0 = fc.timer_restart(t0, 'save df geojson')

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]          # for contour plotting
    # bounded_mat = np.where(bounded_mat < 0, 0, bounded_mat) # for contour plotting


    if only_points_in_region and df is not None: # and not is_mat_smaller
      df = df.reindex(pd.Index(np.arange(0,bounded_mat.shape[0]*bounded_mat.shape[1])))
      df['lat'], df['lon'] = fc.bound_ravel(lats_1d, lons_1d, basisregion.bounds, transform)
      df[unit] = np.ravel(bounded_mat)
      df['GEOID'] = df['GEOID'].replace(NaN,'')
      df = df[df.GEOID != '']
      # t0 = fc.timer_restart(t0, 'bounded_mat_raveled')
    # exit()


    maxUnit = np.nanmax(bounded_mat) if not only_points_in_region or df is None else np.nanmax(df[unit])
    # print(dates[-1], maxUnit)
    if save_maxVals:
      maxVals.append(maxUnit)
    # continue
    minUnit = np.nanmin(bounded_mat) if not only_points_in_region or df is None else np.nanmin(df[unit])
    if maxMappedValue is None or not isMaxMappedValueGiven:
      maxMappedValue = maxUnit

    # print(maxMappedValue)

    tickSpacing = fc.closest(tickSpacings, maxMappedValue/15)
    # print('tickSpacing =',tickSpacing)

    # print(maxUnit)
    cmap = copy.copy(cm.get_cmap(cmap))

    norm = cl.Normalize(vmin=0, vmax=maxMappedValue, clip=False) # clip=False is default; see https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html#matplotlib.colors.Normalize
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
    # df['color'] = [mapper.to_rgba(v) for v in df[unit]]
    # # t0 = fc.timer_restart(t0, 'color mapping')


    # shapeData = shapeData.boundary
    with plt.style.context(("seaborn", "ggplot")):
      shapeData.boundary.plot(figsize=(18*res,10*res), edgecolor='black', color="white", linewidth=0.3*res)

      plt.xlabel("Longitude", fontsize=12*res)
      plt.ylabel("Latitude", fontsize=12*res)
      plt.xticks(fontsize=12*res)
      plt.yticks(fontsize=12*res)
      plt.title('Average Monthly PM2.5 concentration (' + startend[0] + '-' + startend[1] + ')', fontsize=12*res)
      # plt.title(pm_fname + ' (' + unit + ')', fontsize=10*res)

      xlim0 = plt.xlim()
      ylim0 = plt.ylim()

      plt.xlim((lons_1d[minLon] - deg, lons_1d[maxLon] + deg))
      plt.ylim((lats_1d[maxLat] - deg, lats_1d[minLat] + deg))

      ## contour lines
      levelsDiscrete = np.arange(0,maxMappedValue + tickSpacing,tickSpacing)
      # levelsContinuous = np.arange(0,maxMappedValue + tickSpacing,contourSpacing)
      # print(levelsDiscrete)
      # img = plt.contourf(lons_1d, lats_1d, bounded_mat, levels=levelsContinuous, cmap=cmap, alpha=0.6, extend='both')
      img = plt.imshow(bounded_mat, vmin=minUnit, vmax=maxMappedValue, cmap=cmap, origin='lower', alpha=0.5, extent=[lons_1d[minLon], lons_1d[maxLon], lats_1d[minLat], lats_1d[maxLat]])
      img.cmap.set_over('black')
      img.cmap.set_under('white')
      img.changed()

      ticks = sorted(list(set([maxMappedValue] + list(levelsDiscrete) ))) # [minUnit, maxUnit] + 
      cb = plt.colorbar(mapper, ticks=ticks, drawedges=True, label='μm/m^3', pad=0.001)
      cb.ax.yaxis.label.set_font_properties(fm.FontProperties(size=12*res))
      cb.ax.tick_params(labelsize=9*res)
      plt.grid(False)

      # t0 = fc.timer_restart(t0, 'create plot')
      
      fc.save_plt(plt, outputDir, regionFile + '_' + startend[0] + '_' + startend[1] + '_' + "{:.3f}".format(maxUnit) + '_' + "{:.3f}".format(maxMappedValue) + '_' + fc.utc_time_filename(), 'png')
      # t0 = fc.timer_restart(t0, 'save outfiles')

      # plt.show()
      plt.close()


  if save_maxVals:
    maxValsdf = pd.DataFrame()
    if isYearly:
      maxValsdf['YYYY'] = dates
    else:
      maxValsdf['YYYYMM'] = dates
    maxValsdf[unit] = maxVals

    print(maxValsdf)

    fc.save_df(maxValsdf, outputDir, regionFile + '_maxVals_' + startDate + '_' + endDate + '_' + fc.utc_time_filename(), 'hdf5')
    fc.save_df(maxValsdf, outputDir, regionFile + '_maxVals_' + startDate + '_' + endDate + '_' + fc.utc_time_filename(), 'csv')


  
  t1 = fc.timer_restart(t1, 'total time')

      











