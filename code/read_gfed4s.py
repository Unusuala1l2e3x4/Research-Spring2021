from logging import error
import h5py
import numpy as np

import json
import geopandas
from numpy.lib.function_base import extract
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os
import pathlib
import sys

import time
import datetime as dt
import copy

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
# https://shapely.readthedocs.io/en/latest/manual.html

regionNums = {None:0, 'BONA':1, 'TENA':2, 'CEAM':3, 'NHSA':4, 'SHSA':5, 'EURO':6, 'MIDE':7, 'NHAF':8, 'SHAF':9, 'BOAS':10, 'CEAS':11, 'SEAS':12, 'EQAS':13, 'AUST':14}
# GFED file groups: ['ancill', 'biosphere', 'burned_area', 'emissions', 'lat', 'lon']

def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)

def gfed_filenames(gfed_fnames):
  ret = []
  for filename in gfed_fnames:
    if '.hdf5' in filename:
      ret.append(filename)
  return ret

def flatten_list(regular_list):
  return [item for sublist in regular_list for item in sublist]

def add_matrices(list):
  return np.sum(list, axis=0)

def get_bounds_regionsMasked(regions, regionVal, entireGlobe=False):
  """
  for all regions: regionVal = 0
  for 1 region: 1 <= regionVal <= 14
  for region being the entire globe: set entireGlobe = True
  """
  rowcol = regions.shape
  regionBoundedMasked = copy.deepcopy(regions)
  bounds = {'minLonIndex':regions.shape[1]-1, 'minLatIndex':regions.shape[0]-1, 'maxLonIndex':0, 'maxLatIndex':0}

  for lat in range(rowcol[0]):
    for lon in range(rowcol[1]):
      if (regionVal and regions[lat,lon] == regionVal) or (not regionVal and regions[lat,lon] != 0) or entireGlobe:
        regionBoundedMasked[lat,lon] = 1
        if lon > bounds['maxLonIndex']:
          bounds['maxLonIndex'] = lon
        if lat > bounds['maxLatIndex']:
          bounds['maxLatIndex'] = lat
      else:
        regionBoundedMasked[lat,lon] = 0

  if entireGlobe:
    bounds = {'minLonIndex':0, 'minLatIndex':0, 'maxLonIndex':regions.shape[1]-1, 'maxLatIndex':regions.shape[0]-1}
    return bounds, regionBoundedMasked

  for lat in range(bounds['maxLatIndex'],-1,-1):
    for lon in range(bounds['maxLonIndex'],-1,-1):
      if (regionVal and regions[lat,lon] == regionVal) or (not regionVal and regions[lat,lon] != 0):
        if lon < bounds['minLonIndex']:
          bounds['minLonIndex'] = lon
        if lat < bounds['minLatIndex']:
          bounds['minLatIndex'] = lat
  return bounds, regionBoundedMasked[bounds['minLatIndex']:bounds['maxLatIndex']+1, bounds['minLonIndex']:bounds['maxLonIndex']+1]

def extract_matrix_for_region(mat, bounds, regionBoundedMasked):
  return np.multiply( mat[bounds['minLatIndex']:bounds['maxLatIndex']+1, bounds['minLonIndex']:bounds['maxLonIndex']+1], regionBoundedMasked)


def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()

def lim_length(lim):
  return lim[1] - lim[0]

def marker_size(plt, xlim1, ylim1):
  a = 0.4865063
  # 0.4378557
  x = lim_length(plt.xlim())
  y = lim_length(plt.ylim())
  x1 = lim_length(xlim1)
  y1 = lim_length(ylim1)
  return (x1*y1*a) / (x*y)


def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')
  

def fslash(a, b):
  return a + '/' + b

def save_plt(plt, folderPath, name):
  plt.savefig(os.path.join(folderPath, name + '.png'))

def save_df(df_nonzero, hist_month, hist_year, cldf, stats, folderPath, name):
  fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  del df_nonzero['color']
  print(df_nonzero)
  print(cldf)
  print(stats)
  print(hist_month)
  print(hist_year)
  fd.put('data', df_nonzero, format='table', data_columns=True)
  fd.put('colormap', cldf, format='table', data_columns=True)
  fd.put('stats', stats, format='table', data_columns=True)
  fd.put('hist_month', hist_month, format='table', data_columns=True)
  fd.put('hist_year', hist_year, format='table', data_columns=True)
  fd.close()

def save_df_plt(plt, df_nonzero, hist_month, hist_year, cldf, stats, dest, name):
  folderPath = os.path.join(dest, name)
  os.makedirs(folderPath)
  save_plt(plt, folderPath, name)
  save_df(df_nonzero, hist_month, hist_year, cldf, stats, folderPath, name)

  

def add_month(start):
  if start.month == 12: # next year, month = 1
    return dt.date(start.year + 1, 1, start.day)
  return dt.date(start.year, start.month + 1, start.day) # next month, same year

def add_year(start):
  return dt.date(start.year + 1, start.month, start.day)

def next_jan1(start):
  return dt.date(start.year + 1, 1, start.day) # next jan

def read_json(path):
  return json.load(open(path, 'r'))


# 'DM_AGRI', 'DM_BORF', 'DM_DEFO', 'DM_PEAT', 'DM_SAVA', 'DM_TEMF'
# 'C_AGRI', 'C_BORF', 'C_DEFO', 'C_PEAT', 'C_SAVA', 'C_TEMF'

def get_unit_timesArea(dataset):
  if dataset == 'DM':
    return 'kg_DM'
  elif dataset in ['C', 'BB', 'NPP', 'Rh']:
    return 'g_C'
  elif dataset == 'burned_fraction':
    return 'mE2'
  else:
    return ''

def get_unit(dataset):
  unit = get_unit_timesArea(dataset)
  if unit == 'mE2':
    return 'fraction'
  elif unit in ['g_C', 'kg_DM']:
    return unit + '_mE-2'
  else:
    return 'mE-2'




world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

if __name__ == "__main__":
  numArgs = len(sys.argv)
  startYear, startMonth = int(sys.argv[1]), int(sys.argv[2])
  endYear, endMonth = int(sys.argv[3]), int(sys.argv[4])
  group = sys.argv[5]
  dataset = sys.argv[6]
  cmap = sys.argv[7]

  # params (test)
  # group = 'burned_area'
  # dataset = 'burned_fraction'
  # startYear, startMonth = 2016, 12
  # endYear, endMonth = 2016, 12
  # cmap = 'cool'
  # end params

  regionName, df, bounds, lats, lons = None, None, None, None, None
  
  unit_timesArea = get_unit_timesArea(dataset)
  unit = get_unit(dataset)
  
  title = month_str(startMonth) + '-' + str(startYear) + '_' + month_str(endMonth) + '-' + str(endYear) + '_' + dataset
  if numArgs == 9:
    regionName = sys.argv[8]
    title = regionName + '_' + title

  
  print(title)

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  gfedDir = os.path.join(ppPath, 'GFED4s')
  gfedDir_timesArea = os.path.join(ppPath, 'GFED4s_timesArea')
  outputDir = os.path.join(pPath, 'read_gfed4s-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  
  
  gfed_fnames = os.listdir(gfedDir)
  gfed_fnames = sorted(gfed_filenames(gfed_fnames)) # same as in gfedDir_timesArea

  startDate = dt.date(startYear, startMonth, 1)
  endDate = dt.date(endYear, endMonth, 1)

  print_assert_equal('startDate <= endDate', startDate <= endDate, True)

  currentDate = startDate

  numMonths = (endDate.year - startDate.year)*12 + (endDate.month - startDate.month) + 1
  numYears = float(numMonths) / 12


  hist_month = pd.DataFrame(None, columns=['YYYYMM', unit, unit_timesArea])
  hist_year = pd.DataFrame(None, columns=['YYYY', unit, unit_timesArea])

  stats = pd.DataFrame(None, columns=['month_count', 'year_count', 'region_area', 'region_area_nonzero_cell'])
  stats.month_count = [numMonths]
  stats.year_count = [numYears]

  t0 = timer_start()
  t1 = t0


  for filename in gfed_fnames:
    if currentDate > endDate:
      break
    elif str(currentDate.year) not in filename: # skip years before startDate
      continue
    elif df is None or currentDate.month == 1: # if first/next file
      # open next file
      # by year

      # fd = h5py.File(os.path.join(gfedDir, filename), 'r')
      fd_timesArea = h5py.File(os.path.join(gfedDir_timesArea, filename), 'r')

      next_year = next_jan1(currentDate)
      curr_startMonth = currentDate.month
      curr_endMonth = 12 # default
      if endDate < next_year or currentDate == endDate:
        curr_endMonth = endDate.month

      if df is None:
        df = dict()
        bounds, df['is_in_region'] = get_bounds_regionsMasked(np.matrix(fd_timesArea['ancill/basis_regions']), regionNums[regionName], entireGlobe=False)

        lats = np.matrix(fd_timesArea['lat'])
        lons = np.matrix(fd_timesArea['lon'])

        df['grid_cell_area'] = extract_matrix_for_region(np.matrix(fd_timesArea['ancill/grid_cell_area']), bounds, df['is_in_region'])
        df['lat'] = extract_matrix_for_region(lats, bounds, df['is_in_region'])
        df['lon'] = extract_matrix_for_region(lons, bounds, df['is_in_region'])
        df[unit_timesArea] = df['is_in_region'] * 0.0 # initialize for adding unit_timesArea, eventually will divide by area and numMonths

        stats.region_area = [np.sum(df['grid_cell_area'])]
        t0 = timer_restart(t0, 'df is None')

      temp_months = [ extract_matrix_for_region(np.matrix(fd_timesArea[group][month_str(month_it)][dataset]), bounds, df['is_in_region']) for month_it in range(curr_startMonth, curr_endMonth + 1) ]
      
      df[unit_timesArea] += add_matrices(temp_months)

      monthly_totals_timesArea = [np.sum(mat) for mat in temp_months]
      monthly_totals = np.divide(monthly_totals_timesArea, stats.region_area[0])
      yearly_totals_timesArea = np.sum(monthly_totals_timesArea)
      yearly_totals = np.divide(yearly_totals_timesArea, stats.region_area[0])

      hist_month = hist_month.append(pd.DataFrame({
        # 'year': [str(currentDate.year)]*(curr_endMonth + 1 - curr_startMonth),
        'YYYYMM': [str(currentDate.year)+month_str(m) for m in range(curr_startMonth, curr_endMonth + 1)],
        unit: monthly_totals,
        unit_timesArea: monthly_totals_timesArea
      }), ignore_index=True)
      hist_year = hist_year.append(pd.DataFrame({
        'YYYY': [str(currentDate.year)],
        unit: [yearly_totals],
        unit_timesArea: [yearly_totals_timesArea]
      }), ignore_index=True)

      currentDate = next_year
      fd_timesArea.close()

    
  t0 = timer_restart(t0, '2nd-last year')

  # pd.set_option('display.max_rows', None)
  # print(hist_month)
  # print(hist_year)

  # df_nonzero = copy.deepcopy(df)
  df_nonzero = pd.DataFrame()

  for key in df.keys():
    df_nonzero[key] = np.ravel(df[key])

  # # after reading all months/years
  df_nonzero = df_nonzero[df_nonzero[unit_timesArea] != 0.0]
  stats.region_area_nonzero_cell = [np.sum(df_nonzero['grid_cell_area'])]
  del df_nonzero['is_in_region']

  df_nonzero[unit_timesArea + '_monthE-1'] = df_nonzero[unit_timesArea] / numMonths
  df_nonzero[unit] = np.divide(df_nonzero[unit_timesArea], df_nonzero['grid_cell_area'])
  df_nonzero[unit + '_monthE-1'] = df_nonzero[unit] / numMonths

  hist_month[unit_timesArea + '_monthE-1'] = hist_month[unit_timesArea] / numMonths
  hist_month[unit + '_monthE-1'] = hist_month[unit] / numMonths

  hist_year[unit_timesArea + '_monthE-1'] = hist_year[unit_timesArea] / numMonths
  hist_year[unit + '_monthE-1'] = hist_year[unit] / numMonths

  if unit == 'fraction':
    del df_nonzero[unit]
    del df_nonzero[unit_timesArea]
    del hist_month[unit]
    del hist_month[unit_timesArea]
    del hist_year[unit]
    del hist_year[unit_timesArea]

  # print(df_nonzero.shape)
  # print(df_nonzero)
  # print(stats)

  t0 = timer_restart(t0, 'flatten df_nonzero')
  # t0 = timer_restart(t0, 'stats')

  plotted_unit = unit + '_monthE-1'

  color_buf = 0 # 1e8
  color_min = min(df_nonzero[plotted_unit]) - color_buf
  color_max = max(df_nonzero[plotted_unit]) + color_buf
  # color_min = 0
  # color_max = 1

  # print(color_min)
  # print(color_max)
  norm = cl.Normalize(vmin=color_min, vmax=color_max, clip=False) # clip=False is default
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable
  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

  df_nonzero['color'] = [mapper.to_rgba(v) for v in df_nonzero[plotted_unit]]

  cldf = pd.DataFrame()
  cldf['min'] = [color_min]
  cldf['max'] = [color_max]
  cldf['cmap'] = [cmap]
  cldf['plotted_unit'] = [plotted_unit]

  # print(cldf)
  t0 = timer_restart(t0, 'colormap')

  # plot result
  with plt.style.context(("seaborn", "ggplot")):
    world.plot(figsize=(18,10),
                color="white",
                edgecolor = "grey")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title + ' (' + plotted_unit + ')')

    # plt.xlim((-180,180)) # default
    # plt.ylim((-90,90)) # default

    xlim1 = plt.xlim()
    ylim1 = plt.ylim()

    lonBounds = [lons[0,bounds['minLonIndex']], lons[0,bounds['maxLonIndex']]]
    latBounds = [lats[bounds['minLatIndex'],0], lats[bounds['maxLatIndex'],0]]
    plt.xlim((min(lonBounds), max(lonBounds)))
    plt.ylim((min(latBounds), max(latBounds)))

    ms = marker_size(plt, xlim1, ylim1)
    # print(ms)
    plt.scatter(df_nonzero['lon'], df_nonzero['lat'], s=ms, c=df_nonzero.color, alpha=1, linewidths=0, marker='s')
    plt.colorbar(mapper)

    now = utc_time_filename()
    t0 = timer_restart(t0, 'create plot')
    save_df_plt(plt, df_nonzero, hist_month, hist_year, cldf, stats, outputDir, title + '-' + utc_time_filename())

    t0 = timer_restart(t0, 'save outfiles')
    t1 = timer_restart(t1, 'total time')

    # plt.show()

