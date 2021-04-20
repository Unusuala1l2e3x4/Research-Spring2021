
import numpy as np
from numpy import cos, sin, arctan2, arccos
import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, re, json, h5py
from shapely.geometry import shape
import rasterio, rasterio.features, rasterio.warp

import importlib
fc = importlib.import_module('functions')


def decrease_deg_lat_lon(deg0, deg1, lat, lon):
  assert deg0 > deg1
  ratio = int(round(deg0 / deg1))
  d = np.arange(0, deg1*(ratio/2), deg1) if ratio % 2 == 1 else np.arange(deg1/2, deg1*(ratio/2), deg1)
  diffs = sorted(set(list(d)+list(-d)))
  lat_ = np.ravel([ [i-j for j in diffs] for i in lat])
  lon_ = np.ravel([ [i+j for j in diffs] for i in lon])
  print(lat_, len(lat_), len(lat_)/ratio)
  print(lon_, len(lon_), len(lon_)/ratio)
  return lat_, lon_

def decrease_deg_mat(deg0, deg1, mat):
  assert deg0 > deg1
  ratio = int(round(deg0 / deg1))
  ret = []
  for row in mat:
    row_ = np.ravel([ np.repeat(item,ratio) for item in row])
    for r in range(ratio):
      ret.append(row_)
  return np.matrix(ret)





if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(ppPath, 'GFED4s_county')
  if not os.path.isdir(outputDir):
    os.makedirs(outputDir)
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  gfedDir = os.path.join(ppPath, 'GFED4s')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')
  
  points_in_region_fnames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]

  gfed_fnames = os.listdir(gfedDir)
  gfed_fnames = sorted(fc.gfed_filenames(gfed_fnames)) # same as in gfedDir_timesArea
  # print(gfed_fnames)
  # exit()

  # PARAMS
  regionDir, regionFile = 'basisregions', 'TENA.geo.json'
  startDate, endDate = '200001', '201812'
  # END PARAMS

  
  unit = 'μm_m^-3'

  t0 = fc.timer_start()
  t1 = t0

  latlonGEOID, mat, tf, geoidMat, areaMat, latAreas = None, None, None, None, None, None

  fdGFED = None

  
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

  deg, deg_ = 0.01, 0.25
  templon = 0
  datasets = dict()
  minLat_, maxLat_, minLon_, maxLon_, lats_1d_, lons_1d_, tf_ = None,None,None,None,None,None,None

  t0 = fc.timer_restart(t0, 'setup')


  fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), pm_fnames[0], 'nc') # for checking
  mat = fd.variables['PM25'][:] # for checking

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
  t0 = fc.timer_restart(t0, 'initialize (areaMat, geoidMat; pm25, lats/lons for checking)')



  # ~1.5 min per hfed file
  for fname in gfed_fnames:
    year = re.split('.hdf5|_',fname)[1]
    if year < startDate[:4] or year > endDate[:4]:
      continue
    print(year)

    fdGFED = h5py.File(os.path.join(gfedDir, fname), 'r')

    if lats_1d_ is None or lons_1d_ is None:
      lats_1d_ = sorted(np.unique(fdGFED['lat'][:]), reverse=True)
      lons_1d_ = sorted(np.unique(fdGFED['lon'][:]))
      assert tf_ is None
      tf_ = rasterio.transform.from_origin(np.round(np.min(lons_1d_), 0), np.round(np.max(lats_1d_), 0), deg_,deg_)
      # (minx, miny, maxx, maxy)
      bounds_ = (min(lons_1d), min(lats_1d), max(lons_1d), max(lats_1d))

      minLat_, maxLat_, minLon_, maxLon_ = fc.get_bound_indices(bounds_, tf_)
      # print(minLat_, maxLat_, minLon_, maxLon_)

      xy = rasterio.transform.xy(tf_, range(len(lats_1d_)), range(len(lats_1d_)))
      lats_1d_ = np.array(xy[1])
      xy = rasterio.transform.xy(tf_, range(len(lons_1d_)), range(len(lons_1d_)))
      lons_1d_ = np.array(xy[0])

      lats_1d_ = lats_1d_[minLat_:maxLat_]
      lons_1d_ = lons_1d_[minLon_:maxLon_]

      lats_1d_,lons_1d_ = decrease_deg_lat_lon(deg_,deg,lats_1d_,lons_1d_)
      assert set(np.round(lats_1d,3)) == set(np.round(lats_1d_,3))
      assert set(np.round(lons_1d,3)) == set(np.round(lons_1d_,3))
      t0 = fc.timer_restart(t0, 'check decrease_deg_lat_lon')
    
    for group in fdGFED.keys():
      if group in ['ancill','lat','lon']:
        continue
      # print(group)
      for month in fdGFED[group].keys():
        if year+month < startDate or year+month > endDate:
          continue
        # print('\t',year+month)
        for ds in fdGFED[group][month].keys():
          if ds in ['daily_fraction', 'diurnal_cycle', 'partitioning', 'source']:
            continue
          
          # print('\t\t',ds)

          if ds not in datasets.keys():
            datasets[ds] = pd.DataFrame()
            datasets[ds]['GEOID'] = shapeData['GEOID']

          mat_ = fdGFED[group][month][ds][:][minLat_:maxLat_,minLon_:maxLon_]
          # print(mat_.shape)
          mat_ = decrease_deg_mat(deg_, deg, mat_)
          # print(mat_)
          # print(mat_.shape)

          assert mat.shape == mat_.shape

          datasets[ds][year+month] = fc.aggregate_by_geoid(areaMat, mat_, geoidMat, tf, shapeData)



  t0 = fc.timer_restart(t0, 'run loop') 


  for key in datasets:
    title = 'TENA_' + key + '_' + startDate + '_' + endDate
    fc.save_df(datasets[key], outputDir, title, 'hdf5')
    t0 = fc.timer_restart(t0, 'save hdf5')
    fc.save_df(datasets[key], outputDir, title, 'csv')
    t0 = fc.timer_restart(t0, 'save csv')

  


  t1 = fc.timer_restart(t1, 'total time')


