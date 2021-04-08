
import numpy as np
from numpy import cos, sin, arctan2, arccos
import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, io, sys, copy, re, json
from shapely.geometry import shape, MultiPoint
import rasterio, rasterio.features, rasterio.warp

import matplotlib.path as mplp

import importlib

from shapely.ops import transform
fc = importlib.import_module('functions')


# WGS84_RADIUS = 6378137
WGS84_RADIUS = 6366113.579189922 # from area_test_for_earth_radius.py
WGS84_RADIUS_SQUARED = WGS84_RADIUS**2
d2r = np.pi/180

def greatCircleBearing(lon1, lat1, lon2, lat2):
    dLong = lon1 - lon2
    s = cos(d2r*lat2)*sin(d2r*dLong)
    c = cos(d2r*lat1)*sin(d2r*lat2) - sin(lat1*d2r)*cos(d2r*lat2)*cos(d2r*dLong)
    return arctan2(s, c)

def quad_area(lon, lat, deg):
  deg = deg / 2
  lons = [lon+deg,lon+deg,lon-deg,lon-deg]
  lats = [lat+deg,lat-deg,lat-deg,lat+deg]
  N = 4 # len(lons)
  angles = np.empty(N)
  for i in range(N):
      phiB1, phiA, phiB2 = np.roll(lats, i)[:3]
      lB1, lA, lB2 = np.roll(lons, i)[:3]
      # calculate angle with north (eastward)
      beta1 = greatCircleBearing(lA, phiA, lB1, phiB1)
      beta2 = greatCircleBearing(lA, phiA, lB2, phiB2)
      # calculate angle between the polygons and add to angle array
      angles[i] = arccos(cos(-beta1)*cos(-beta2) + sin(-beta1)*sin(-beta2))
  return (np.sum(angles) - (N-2)*np.pi)*WGS84_RADIUS_SQUARED


def get_bound_indices(bounds, transform):
  rc = rasterio.transform.rowcol(transform, [bounds[0], bounds[2]], [bounds[1], bounds[3]], op=round, precision=4)
  minLon = max(rc[1][0], 0)
  maxLon = rc[1][1]
  minLat = max(rc[0][1], 0)
  maxLat = rc[0][0]
  return minLat, maxLat, minLon, maxLon

def bound_ravel(lats_1d, lons_1d, bounds, transform):
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  lats_1d = lats_1d[minLat:maxLat]
  lons_1d = lons_1d[minLon:maxLon]
  X, Y = np.meshgrid(lons_1d, lats_1d)
  return np.ravel(Y), np.ravel(X)

def boundary_to_mask(boundary, x, y):  # https://stackoverflow.com/questions/34585582/how-to-mask-the-specific-array-data-based-on-the-shapefile/38095929#38095929
  mpath = mplp.Path(boundary)
  X, Y = np.meshgrid(x, y)
  points = np.array((X.flatten(), Y.flatten())).T
  mask = mpath.contains_points(points).reshape(X.shape)
  return mask
  
def get_geoidMat(bounds, transform, shapeData, lats_1d, lons_1d):
  geoidMat = np.empty((len(lats_1d), len(lons_1d)), dtype='<U20')
  minLat0, maxLat0, minLon0, maxLon0 = get_bound_indices(bounds, transform)
  for row in shapeData.itertuples():
    if row.geometry.boundary.geom_type == 'LineString': # assuming there is no
      minLat, maxLat, minLon, maxLon = get_bound_indices(row.geometry.boundary.bounds, transform)
      if minLat == maxLat or minLon == maxLon:
        continue
      mask = boundary_to_mask(row.geometry.boundary, lons_1d[minLon:maxLon], lats_1d[minLat:maxLat])
      mask = np.where(mask, row.GEOID, '')
      geoidMat[minLat:maxLat,minLon:maxLon] = np.char.add(geoidMat[minLat:maxLat,minLon:maxLon], mask) # https://numpy.org/doc/stable/reference/routines.char.html#module-numpy.char
    else:
      # sort line indices by nest depth
      lineIndexNestDepth = dict()
      for i in range(len(row.geometry.boundary)):
        lineIndexNestDepth[i] = [mplp.Path(outerline).contains_path(mplp.Path(row.geometry.boundary[i])) for outerline in row.geometry.boundary if row.geometry.boundary[i] != outerline].count(True)
      # sort indices by nest depth (sort by dict values)
      for l in sorted(lineIndexNestDepth, key=lineIndexNestDepth.get): 
        minLat, maxLat, minLon, maxLon = get_bound_indices(row.geometry.boundary[l].bounds, transform)
        if minLat == maxLat or minLon == maxLon:
          continue
        mask = boundary_to_mask(row.geometry.boundary[l], lons_1d[minLon:maxLon], lats_1d[minLat:maxLat])
        if lineIndexNestDepth[l] % 2 == 1: # nest depth = 1
          for r in range(geoidMat[minLat:maxLat,minLon:maxLon].shape[0]):
            for c in range(geoidMat[minLat:maxLat,minLon:maxLon].shape[1]):
              if mask[r,c] and geoidMat[minLat+r,minLon+c] != '':
                geoidMat[minLat+r,minLon+c] = geoidMat[minLat+r,minLon+c][:-5] # remove points from when nest depth = 0
        else:
          mask = np.where(mask, row.GEOID, '')
          geoidMat[minLat:maxLat,minLon:maxLon] = np.char.add(geoidMat[minLat:maxLat,minLon:maxLon], mask)
  return geoidMat[minLat0:maxLat0,minLon0:maxLon0]


def rasterize_geoids_df(bounds, transform, shapeData, lats_1d, lons_1d):
  df = pd.DataFrame()
  df['lat'], df['lon'] = bound_ravel(lats_1d, lons_1d, bounds, transform)
  df['GEOID'] = np.ravel(get_geoidMat(bounds, transform, shapeData, lats_1d, lons_1d))
  return df[df['GEOID'] != '']



dates = ['199901', '199902', '199903', '199904', '199905', '199906', '199907', '199908', '199909', '199910', '199911', '199912', '200001', '200002', '200003', '200004', '200005', '200006', '200007', '200008', '200009', '200010', '200011', '200012', '200101', '200102', '200103', '200104', '200105', '200106', '200107', '200108', '200109', '200110', '200111', '200112', '200201', '200202', '200203', '200204', '200205', '200206', '200207', '200208', '200209', '200210', '200211', '200212', '200301', '200302', '200303', '200304', '200305', '200306', '200307', '200308', '200309', '200310', '200311', '200312', '200401', '200402', '200403', '200404', '200405', '200406', '200407', '200408', '200409', '200410', '200411', '200412', '200501', '200502', '200503', '200504', '200505', '200506', '200507', '200508', '200509', '200510', '200511', '200512', '200601', '200602', '200603', '200604', '200605', '200606', '200607', '200608', '200609', '200610', '200611', '200612', '200701', '200702', '200703', '200704', '200705', '200706', '200707', '200708', '200709', '200710', '200711', '200712', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907']

if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'write_county_month_pm2-5-outfiles')
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')
  
  
  points_in_region_fnames = os.listdir(os.path.join(pmDir, 'points_in_region'))
  pm_fnames = [re.split('.nc',l)[0] for l in sorted(os.listdir(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25')))]

  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title

  unit = 'μm_m^-3'

  testing = False
  origDataMonth = '07'
  suppValString = '-1'

  regionDir, regionFile = 'basisregions', 'TENA.geo.json'
  startDate, endDate = '200001', '200001'
  isYearly = False

  # END PARAMS

  t0 = fc.timer_start()
  t1 = t0

  df, tf = None, None
  minLat, maxLat, minLon, maxLon = None, None, None, None
  lats_1d, lons_1d = None, None
  bounded_mat = None


  dates = [i for i in dates if i >= startDate and i <= endDate]

  
  with open(os.path.join(shapefilesDir, regionDir, regionFile), 'r') as f:
    contents = json.load(f)
    basisregion = shape(contents['features'][0]['geometry'])
  t0 = fc.timer_restart(t0, 'read basisregion')

  regionFile = regionFile.split('.')[0]

  if regionFile + '_rounded.hdf5' not in os.listdir(os.path.join(pmDir, 'points_in_region_rounded')):
    df = fc.read_df(os.path.join(pmDir, 'points_in_region'), regionFile, 'hdf5')
    df.lat = [round(i, 3) for i in df.lat]
    df.lon = [round(i, 3) for i in df.lon]
    fc.save_df(df, os.path.join(pmDir, 'points_in_region_rounded'), regionFile + '_rounded', 'hdf5')
  else:
    df = pd.DataFrame(fc.read_df(os.path.join(pmDir, 'points_in_region_rounded'), regionFile + '_rounded', 'hdf5'))

  t0 = fc.timer_restart(t0, 'get df')

  # df2 = df.sort_values(by='lat', ascending=False).reset_index(drop=True)
  # print(list(df.lat) == list(df2.lat)) # True

  
  
  countyMapFile = 'cb_2019_us_county_500k'
  countyMapData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp'))
  countyMapData = fc.clean_states_reset_index(countyMapData)
  countyMapData = fc.county_changes_deaths_reset_index(countyMapData)
  # countyMapData = countyMapData.sort_values(by=['GEOID']).reset_index(drop=True)

  countyMapData['ATOTAL'] = countyMapData['ALAND'] + countyMapData['AWATER']

  # print(countyMapData)
  # exit()
  deg = 0.01
  templon = 0

  latsArea = pd.DataFrame(sorted(set(df.lat), reverse=True), columns=['lat'])
  latsArea['area'] = [quad_area(templon, lat, deg) for lat in latsArea.lat]

  # print(latsArea)

  countyPM25 = pd.DataFrame()
  countyPM25['GEOID'] = countyMapData['GEOID']
  # if testing:
  #   countyPM25['ATOTAL'] = countyMapData['ATOTAL']
  #   countyPM25['area'] = NaN
  #   countyPM25['error'] = NaN

  t0 = fc.timer_restart(t0, 'get df')

  # print(countyPM25)

  for pm_fname in pm_fnames:
    startend = re.split('_|-',pm_fname)[3:5]
    if startend[0] < startDate or startend[1] > endDate or (startend[0] == startend[1] and isYearly) or (startend[0] != startend[1] and not isYearly):
      continue

    fd = fc.read_df(os.path.join(pmDir, 'V4NA03/NetCDF/NA/PM25'), pm_fname, 'nc') # http://unidata.github.io/netcdf4-python/
    mat = fd.variables['PM25'][:]

    if tf is None: # minLat, maxLat, minLon, maxLon are also none
      deg = np.average(np.abs(fd.variables['LON'][:-1] - fd.variables['LON'][1:]))
      # print(deg)
      transform = rasterio.transform.from_origin(np.round(np.min(fd.variables['LON'][:]), 2), np.round(np.max(fd.variables['LAT'][:]), 2), deg,deg)
      minLat, maxLat, minLon, maxLon = get_bound_indices(basisregion.bounds, transform)

      # t0 = fc.timer_restart(t0, 'get transform')
      xy = rasterio.transform.xy(transform, range(fd.dimensions['LAT'].size), range(fd.dimensions['LAT'].size))
      lats_1d = np.array(xy[1])
      xy = rasterio.transform.xy(transform, range(fd.dimensions['LON'].size), range(fd.dimensions['LON'].size))
      lons_1d = np.array(xy[0])

    bounded_mat = mat[minLat:maxLat,minLon:maxLon]


    df = df.reindex(pd.Index(np.arange(0,bounded_mat.shape[0]*bounded_mat.shape[1])))
    df['lat'], df['lon'] = bound_ravel(lats_1d, lons_1d, basisregion.bounds, tf)
    df[unit] = np.ravel(bounded_mat)
    df['GEOID'] = df['GEOID'].replace(NaN,'')
    df = df[df.GEOID != '']


    for geoid in countyPM25.GEOID:
      # print(geoid)
      temp = df.loc[df.GEOID == geoid,'lat'].value_counts().sort_index(ascending=False)
      # print(temp)
      temp2 = latsArea.loc[(latsArea.lat >= min(temp.index)) & (latsArea.lat <= max(temp.index)),:]
      # print(temp2.lat)
      temp = temp.reindex(temp2.lat, fill_value=0)
      # assert list(temp.index) == list(temp2.lat)

      # print(temp)
      # print(temp2.area)
      area = np.sum([x*y for x,y in zip(temp,temp2.area)])

      # if testing:
      #   area2 = countyPM25.loc[countyPM25.GEOID == geoid, 'ATOTAL']
      #   countyPM25.loc[countyPM25.GEOID == geoid, 'area'] = area
      #   countyPM25.loc[countyPM25.GEOID == geoid, 'error'] = np.abs(area - area2)/area2

      break
  
  t0 = fc.timer_restart(t0, 'run loop')

  # if testing:
  #   print(np.nanmean(countyPM25.error))
  print(countyPM25)


  fc.save_df(countyPM25, outputDir, 'CountyAreasErrors', 'csv')
# (df, folderPath, name, ext)




  # fc.save_df(dfOut, nClimDivDir, outFileName, 'hdf5')
  # t0 = fc.timer_restart(t0, 'save hdf5 '+outFileName)
  # fc.save_df(dfOut, nClimDivDir, outFileName, 'csv')
  # t0 = fc.timer_restart(t0, 'save csv '+outFileName)



  t1 = fc.timer_restart(t1, 'total time')


