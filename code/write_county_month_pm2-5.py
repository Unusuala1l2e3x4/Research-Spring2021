
import numpy as np

import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, re, json
from shapely.geometry import shape
import rasterio, rasterio.features, rasterio.warp

import importlib
fc = importlib.import_module('functions')


if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # outputDir = os.path.join(pPath, 'write_county_month_pm2-5-outfiles')
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  
  points_in_region_fnames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]

  # PARAMS
  regionDir, regionFile = 'basisregions', 'TENA.geo.json'
  startDate, endDate = '200001', '201812'
  # END PARAMS

  title = 'TENA_county_PM25_' + startDate + '_' + endDate
  unit = 'μm_m^-3'

  t0 = fc.timer_start()
  t1 = t0

  latlonGEOID, tf, geoidMat, areaMat, latAreas = None, None, None, None, None

  
  with open(os.path.join(shapefilesDir, regionDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])
  t0 = fc.timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '_rounded.hdf5' not in os.listdir(os.path.join(pmDir, 'points_in_region_rounded')):
    latlonGEOID = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
    latlonGEOID.lat = [round(i, 3) for i in latlonGEOID.lat]
    latlonGEOID.lon = [round(i, 3) for i in latlonGEOID.lon]
    fc.save_df(latlonGEOID, os.path.join(pmDir, 'points_in_region_rounded'), regionFile + '_rounded', 'hdf5')
  else:
    latlonGEOID = pd.DataFrame(fc.read_df(os.path.join(pmDir, 'points_in_region_rounded'), regionFile + '_rounded', 'hdf5'))

  t0 = fc.timer_restart(t0, 'get latlonGEOID')

  
  countyMapFile = 'cb_2019_us_county_500k'
  shapeData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp'))
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData)
  shapeData = shapeData.sort_values(by=['GEOID']).reset_index(drop=True)

  shapeData['ATOTAL'] = shapeData['ALAND'] + shapeData['AWATER']
  # print(shapeData)

  deg = 0.01
  templon = 0

  countyPM25 = pd.DataFrame()
  countyPM25['GEOID'] = shapeData['GEOID']

  t0 = fc.timer_restart(t0, 'setup')

  # print(countyPM25)

  for pm_fname in pm_fnames:
    startend = re.split('_|-',pm_fname)[3:5]
    if startend[0] < startDate or startend[1] > endDate or (startend[0] != startend[1]):
      continue
    print(startend[0] + '-' + startend[1])

    fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), pm_fname, 'nc') # http://unidata.github.io/netcdf4-python/
    mat = fd.variables['PM25'][:]

    if tf is None: # if geoidMat is None and areaMat is None -> same effect
      tf = rasterio.transform.from_origin(np.round(np.min(fd.variables['LON'][:]), 2), np.round(np.max(fd.variables['LAT'][:]), 2), deg,deg)
      minLat, maxLat, minLon, maxLon = fc.get_bound_indices(basisregion.bounds, tf)
      # print(minLat, maxLat, minLon, maxLon)
      xy = rasterio.transform.xy(tf, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
      lats_1d = np.array(xy[1])
      xy = rasterio.transform.xy(tf, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
      lons_1d = np.array(xy[0])

      bounded_mat = mat[minLat:maxLat,minLon:maxLon]
      # print(bounded_mat.shape)
      latlonGEOID = latlonGEOID.reindex(pd.Index(np.arange(0,bounded_mat.shape[0]*bounded_mat.shape[1])))
      latlonGEOID['lat'], latlonGEOID['lon'] = fc.bound_ravel(lats_1d, lons_1d, basisregion.bounds, tf)
      # latlonGEOID[unit] = np.ravel(bounded_mat)
      latlonGEOID['GEOID'] = latlonGEOID['GEOID'].replace(NaN,'')
      # latlonGEOID = latlonGEOID[latlonGEOID.GEOID != '']
      temp = np.reshape(list(latlonGEOID['GEOID']), bounded_mat.shape)
      # print(latlonGEOID[latlonGEOID.GEOID != ''])
      geoidMat = np.empty(mat.shape, dtype='<U5')
      geoidMat[minLat:maxLat,minLon:maxLon] = temp

      latAreas = pd.DataFrame(np.reshape(list(latlonGEOID['lat']), bounded_mat.shape)[:,0], columns=['lat'])
      latAreas['area'] = [fc.quad_area(templon, lat, deg) for lat in latAreas.lat]
      # print(latAreas)

      temp = np.matrix([np.repeat(a, bounded_mat.shape[1]) for a in latAreas.area])
      areaMat = np.empty(mat.shape)
      areaMat[minLat:maxLat,minLon:maxLon] = temp

      # print(areaMat.shape, mat.shape, geoidMat.shape)
      # print(areaMat[minLat:maxLat,minLon:maxLon], mat[minLat:maxLat,minLon:maxLon], geoidMat[minLat:maxLat,minLon:maxLon])

      t0 = fc.timer_restart(t0, 'initialize')

    countyPM25[startend[0]] = fc.aggregate_by_geoid(areaMat, mat, geoidMat, tf, shapeData)

    # t0 = fc.timer_restart(t0, 'aggregate_by_geoid '+startend[0] + '-' + startend[1])
    # print(countyPM25[startend[0]])

  
  t0 = fc.timer_restart(t0, 'run loop') # ~6.7 min

  # print(countyPM25)

  fc.save_df(countyPM25, pmDir, title, 'hdf5')
  t0 = fc.timer_restart(t0, 'save hdf5')
  fc.save_df(countyPM25, pmDir, title, 'csv')
  t0 = fc.timer_restart(t0, 'save csv')

  


  t1 = fc.timer_restart(t1, 'total time') # ~6.9 min


