import numpy as np
from numpy.core.numeric import NaN
from numpy import cos, sin, arctan2, arccos

import pandas as pd
import geopandas as gpd
import zipfile

import os, re, netCDF4, pathlib

import time
import datetime as dt

import matplotlib.path as mplp
import rasterio, rasterio.features, rasterio.warp

def save_df(df, folderPath, name, ext):
  print('save', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  elif ext == 'hdf5':
    df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format='fixed')
  elif ext == 'zip':
    df.to_csv(os.path.join(folderPath, name + '.zip'), compression=dict(method='zip',archive_name=name + '.csv'), index=False)
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
  elif ext == 'nc':
    return netCDF4.Dataset(os.path.join(folderPath, name + '.nc'))
  elif ext == 'zip':
    return pd.read_csv(zipfile.ZipFile(os.path.join(folderPath, name + '.zip')).open(name + '.csv', 'r'), index_col=False)

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


def rank_decreasing(arr):
  return [int(i) for i in pd.Series(arr).rank(method='min', ascending=False)]
  

def gfed_filenames(gfed_fnames):
  ret = []
  for filename in gfed_fnames:
    if '.hdf5' in filename:
      ret.append(filename)
  return ret


def GEOID_string_state_county(state, county):
  state = str(state)
  county = str(county)
  while len(state) != 2:
    state = '0' + state
  while len(county) != 3:
    county = '0' + county
  return state + county
  
def countyGEOIDstring(geo):
  geo = str(int(geo))
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
  statefps = set([item[0:2] for item in df.GEOID])

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
    # Edit: if not in data, add and set pop to 0 #####################
  temp1 = df[df.GEOID == '08014']
  # if len(temp1.GEOID) != 0:
  #   df = df.drop([temp1.index[0]])
  if len(temp1.GEOID) == 0 and '08' in statefps and statefps != set(df.GEOID):
    df = df.append(pd.DataFrame([['08014'] + [NaN]*len(dates)], columns=['GEOID']+dates))

  return df.sort_values(by='GEOID').reset_index(drop=True)

def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)


def parse_lines_deaths(path, suppValString):
  lines = open(path).readlines()
  lines = [list(filter(None, re.split('\t|\n|"|/',l))) for l in lines]
  for l in lines:
    if 'Suppressed'in l:
      l[l.index('Suppressed')] = suppValString
  lines = [[item for item in line if (item == suppValString or str.isnumeric(item))] for line in lines[1:lines.index(['---'])]]
  return [[l[0], l[1] + l[2], l[3]] for l in lines]

def deaths_by_date_geoid(title, suppValString):
  linesAll = []
  if os.path.isdir(title):
    filenames = os.listdir(title)
    for filename in filenames:
      linesAll += parse_lines_deaths(os.path.join(title, filename), suppValString)
  else:
    title += '.txt'
    linesAll = parse_lines_deaths(title, suppValString)
  ret = pd.DataFrame(linesAll, columns=['GEOID', 'YYYYMM', 'deaths']).sort_values(by=['YYYYMM','GEOID']).reset_index(drop=True)
  ret['deaths'] = pd.to_numeric(ret['deaths'])
  return ret

def closest(lst, K):
  return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def shortestAxisLength(bounds):
  return min(bounds[2]-bounds[0], bounds[3]-bounds[1])

def lim_length(lim):
  return lim[1] - lim[0]

def marker_size(plt, xlim1, ylim1, deg):
  a = 0.4865063 * (deg / 0.25)
  # 0.4378557
  x = lim_length(plt.xlim())
  y = lim_length(plt.ylim())
  x1 = lim_length(xlim1)
  y1 = lim_length(ylim1)
  return (x1*y1*a) / (x*y)


############################################### for aggregating grid data to counties
def boundary_to_mask(boundary, x, y):
  # https://stackoverflow.com/questions/34585582/how-to-mask-the-specific-array-data-based-on-the-shapefile/38095929#38095929
  mpath = mplp.Path(boundary)
  X, Y = np.meshgrid(x, y)
  points = np.array((X.flatten(), Y.flatten())).T
  mask = mpath.contains_points(points).reshape(X.shape)
  return mask

def get_bound_indices(bounds, transform):
  rc = rasterio.transform.rowcol(transform, [bounds[0], bounds[2]], [bounds[1], bounds[3]], op=round, precision=4)
  minLon = max(rc[1][0], 0)
  maxLon = rc[1][1]
  minLat = max(rc[0][1], 0)
  maxLat = rc[0][0]
  return minLat, maxLat, minLon, maxLon

def is_mat_smaller(mat, bounds, transform):
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  return minLat == minLon == 0 and maxLat + 1 >= len(mat) and maxLon + 1 >= len(mat[0])

def bound_ravel(lats_1d, lons_1d, bounds, transform):
  minLat, maxLat, minLon, maxLon = get_bound_indices(bounds, transform)
  lats_1d = lats_1d[minLat:maxLat]
  lons_1d = lons_1d[minLon:maxLon]
  X, Y = np.meshgrid(lons_1d, lats_1d)
  return np.ravel(Y), np.ravel(X)

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

  
def aggregate_by_geoid(areaMat, mat, geoidMat, transform, shapeData):
  ret = []
  for row in shapeData.itertuples():
    minLat, maxLat, minLon, maxLon = get_bound_indices(row.geometry.boundary.bounds, transform)
    mask = np.where(geoidMat[minLat:maxLat,minLon:maxLon] == row.GEOID, 1, 0)
    # print(mask.shape)
    # print(mat[minLat:maxLat,minLon:maxLon].shape)
    dataMask = np.multiply(mask, mat[minLat:maxLat,minLon:maxLon])
    areaMask = np.multiply(mask, areaMat[minLat:maxLat,minLon:maxLon])
    data_area = np.nansum(np.multiply(dataMask, areaMask))
    area = np.sum(areaMask)
    data = np.divide(data_area, area)
    ret.append(data)
    # print(row.GEOID, row.ATOTAL, area, data)
  return ret


############################################### for cell area calculation
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


############################################### for Random Forest
def ravel_by_dates(unit, arr, dates):
  ret = pd.DataFrame()
  X, Y = np.meshgrid(arr, dates)
  ret[unit] = np.ravel(X)
  ret['YYYYMM'] = np.ravel(Y)
  return ret


def get_all_X_columns():
  return ['GEOID','month','temp_F', 'burned_frac', 'popuDensity_ALAND_km2', 'precip_in', 'Rh_g_m-2', 'pm25_ug_m-3', 'NPP_g_m-2', 'C_g_m-2', 'smallf_frac', 'DM_kg_m-2', 'BB_g_m-2', 'ALAND_ATOTAL_ratio', 'median_inc']

def get_all_data(startYYYYMM, endYYYYMM):
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')
  usCensusDir = os.path.join(ppPath, 'US Census Bureau')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  gfedCountyDir = os.path.join(ppPath, 'GFED4s_county')
  countyMapFile = 'cb_2019_us_county_500k'
  ext = 'hdf5'

  fileArgs = [('deaths', os.path.join(cdcWonderDir, 'Chronic lower respiratory diseases'), 'By county - Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019, suppressed estimates'), 
  ('popu', os.path.join(usCensusDir, 'population'), 'TENA_county_pop_1999_2019'),
  ('median_inc', os.path.join(usCensusDir, 'SAIPE State and County Estimates'), 'TENA_county_median_income_2000_2019'),
  ('precip_in', nClimDivDir, 'climdiv-pcpncy-v1.0'),
  ('temp_F', nClimDivDir, 'climdiv-tmpccy-v1.0'),
  ('PDSI', nClimDivDir, 'climdiv-pdsidv-v1.0'),
  ('SP01', nClimDivDir, 'climdiv-sp01dv-v1.0'),
  ('pm25_ug_m-3', pmDir, 'TENA_county_PM25_200001_201812'),
  ('C_g_m-2', gfedCountyDir, 'TENA_C_200001_201812'),
  ('DM_kg_m-2', gfedCountyDir, 'TENA_DM_200001_201812'),
  ('BB_g_m-2', gfedCountyDir, 'TENA_BB_200001_201812'), # 2016 and before
  ('NPP_g_m-2', gfedCountyDir, 'TENA_NPP_200001_201812'), # 2016 and before
  ('Rh_g_m-2', gfedCountyDir, 'TENA_Rh_200001_201812'), # 2016 and before
  ('burned_frac', gfedCountyDir, 'TENA_burned_fraction_200001_201812'), # 2016 and before
  ('smallf_frac', gfedCountyDir, 'TENA_small_fire_fraction_200001_201812')] # 2016 and before
  

  shapeData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = clean_states_reset_index(shapeData)
  shapeData = county_changes_deaths_reset_index(shapeData)
  shapeData['ATOTAL'] = shapeData.ALAND + shapeData.AWATER
  # print(shapeData)

  dates, data = None, None

  for args in fileArgs:
    df = read_df(args[1], args[2], ext)
    if dates is None:
      dates = sorted(i for i in df if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
    df = df.loc[:,['GEOID']+dates]
    if data is None:
      data = ravel_by_dates('GEOID', [countyGEOIDstring(i) for i in df.GEOID], dates)
    data[args[0]] = np.ravel(df.loc[:,dates], order='F')

  data['deathRate'] = 100000 * data.deaths / data.popu
  data['ALAND_km2'] = ravel_by_dates('a', shapeData.ALAND, dates).a / 1000**2
  data['popuDensity_ALAND_km2'] = data.popu / data.ALAND_km2
  # data['AWATER_km2'] = ravel_by_dates('a', shapeData.AWATER, dates).a / 1000**2
  # data['AWATER_km2'] = data['AWATER_km2'].replace(NaN, 0)
  data['ATOTAL_km2'] = ravel_by_dates('a', shapeData.ATOTAL, dates).a / 1000**2
  data['popuDensity_ATOTAL_km2'] = data.popu / data.ATOTAL_km2
  data['ALAND_ATOTAL_ratio'] = data.ALAND_km2 / data.ATOTAL_km2
  data['month'] = [int(i[-2:]) for i in data.YYYYMM]
  return data


