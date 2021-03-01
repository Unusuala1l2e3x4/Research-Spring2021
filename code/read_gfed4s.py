import h5py
import numpy as np

import json
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm

import os
import pathlib
import sys

import time
import datetime as dt

from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union


regionNums = {'BONA':1, 'TENA':2, 'CEAM':3, 'NHSA':4, 'SHSA':5, 'EURO':6, 'MIDE':7, 'NHAF':8, 'SHAF':9, 'BOAS':10, 'CEAS':11, 'SEAS':12, 'EQAS':13, 'AUST':14}
# GFED file groups: ['ancill', 'biosphere', 'burned_area', 'emissions', 'lat', 'lon']

def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)

def gfed_filenames(filenames):
  ret = []
  for filename in filenames:
    if 'GFED' in filename:
      ret.append(filename)
  return ret

def flatten_list(regular_list):
  return [item for sublist in regular_list for item in sublist]


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
  
def save_plt(plt, dest, name):
  plt.savefig(os.path.join(dest, name + '.png'))

def save_df(df, cldf, stats, dest, name):
  fd = pd.HDFStore(os.path.join(dest, name + '.hdf5'))
  del df['color']
  del df['region']
  fd.put('data', df, format='table', data_columns=True)
  fd.put('colormap', cldf, format='table', data_columns=True)
  fd.put('stats', stats, format='table', data_columns=True)
  fd.close()

def save_df_plt(plt, df, cldf, stats, dest, name):
  folderPath = os.path.join(dest, name)
  os.makedirs(folderPath)
  save_plt(plt, folderPath, name)
  save_df(df, cldf, stats, folderPath, name)

  

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

def are_points_inside(basisregion, df):
  return [ basisregion.contains(Point(row.lon, row.lat)) for row in df.itertuples() ]
  


world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

if __name__ == "__main__":
  startYear, startMonth = int(sys.argv[1]), int(sys.argv[2])
  endYear, endMonth = int(sys.argv[3]), int(sys.argv[4])
  group = sys.argv[5]
  dataset = sys.argv[6]
  cmap = sys.argv[7]

  regionName = None
  title = None

  if len(sys.argv) == 9:
    regionName = sys.argv[8]
    title = regionName + '_' + month_str(startMonth) + '-' + str(startYear) + '_' + month_str(endMonth) + '-' + str(endYear) + '_' + dataset
  else:
    title = month_str(startMonth) + '-' + str(startYear) + '_' + month_str(endMonth) + '-' + str(endYear) + '_' + dataset
  
  print(title)

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  gfedDir = os.path.join(ppPath, 'GFED4s')
  outputDir = os.path.join(pPath, 'read_gfed4s-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  
  # basisregion = shape( read_json(shapefilesDir + 'basisregions/' + regionName + '.geo.json')['features'][0]['geometry'] )
  
  filenames = os.listdir(gfedDir)
  
  filenames = sorted(gfed_filenames(filenames))

  # params (test)
  # group = 'burned_area'
  # dataset = 'burned_fraction'
  # startYear, startMonth = 2016, 12
  # endYear, endMonth = 2016, 12
  # cmap = 'cool'
  # end params

  startDate = dt.date(startYear, startMonth, 1)
  endDate = dt.date(endYear, endMonth, 1)

  currentDate = startDate

  t0 = timer_start()
  t1 = t0

  df = None
  regionIndices = None

  for filename in filenames:
    if currentDate > endDate:
      break
    elif str(currentDate.year) not in filename: # skip years before startDate
      continue
    elif df is None or currentDate.month == 1: # if first/next file
      # open next file
      # by year

      fd = h5py.File(os.path.join(gfedDir, filename), 'r')

      next_year = next_jan1(currentDate)

      curr_startMonth = currentDate.month
      curr_endMonth = 12 # default
      if endDate < next_year or currentDate == endDate:
        curr_endMonth = endDate.month

      temp_val = pd.DataFrame([ sum(l) for l in zip(*[flatten_list(fd[group][month_str(month_it)][dataset]) for month_it in range(curr_startMonth, curr_endMonth + 1) ]) ], columns = ['val'])

      if df is None:
        df = pd.DataFrame(None, columns = ['lat', 'lon', 'area', 'val', 'region'])
        df.lat = flatten_list(fd['lat'])
        df.lon = flatten_list(fd['lon'])
        df.region = flatten_list(fd['ancill/basis_regions'])
        df.area = flatten_list(fd['ancill/grid_cell_area'])
        df.val = temp_val.val

        # find indices in region, remove the rest

        # df = df[are_points_inside(basisregion, df)] # specific region, geojson
        if len(sys.argv) == 9:
          df = df[df.region == regionNums[regionName]]  # specific region
        else:
          df = df[df.region != 0] # all regions
        # regionIndices = df.index

        # print(len(df.val))
        t0 = timer_restart(t0, '1st year')

      else: # add to df.val

        # find matching indices
        df.val = [x + y for x, y in zip(df.val, temp_val[temp_val.index.isin(df.index)].val ) ]

      currentDate = next_year
      
      fd.close()
      
  t0 = timer_restart(t0, '2nd to last year')


  # after reading all months/years
  # df = df[df.val != 0.0] # remove zero vals
  print(len(df.val))

  # df = df[are_points_inside(basisregion, df)]
  # t0 = timer_restart(t0, 'are_points_inside')


  stats = pd.DataFrame()
  region_area = np.sum(df.area)
  stats['region_area'] = [region_area]
  stats['region_area_nonzero_cell'] = [np.sum(df[df.val != 0.0].area)]

  numMonths = (endDate.year - startDate.year)*12 + (endDate.month - startDate.month) + 1
  numYears = float(numMonths) / 12
  stats['month_count'] = [numMonths]

  # (g C / m^2 period) per cell
  
  df.val = [v / numMonths for v in df.val] # take average
  # (g C / m^2 month) per cell
  
  if dataset != 'burned_fraction':
    df['cell_total_val'] = [x*y for x,y in zip(df.val, df.area)] 
    # (g C / month) per cell
    sum_cell_total_val = np.sum(df['cell_total_val'])
    stats['region_monthly_avg_val'] = [sum_cell_total_val / numMonths]
    stats['region_yearly_avg_val'] = [sum_cell_total_val / numYears]
    # (g C / year)
  else:
    df['burned_area'] = [x*y for x,y in zip(df.val, df.area)]
    # (m^2 / month) per cell
    stats['region_monthly_avg_burned_fraction'] = [np.sum(df['burned_area']) / region_area ]
    # (m^2 / month)

  t0 = timer_restart(t0, 'stats')


  color_buf = 0 # 1e8
  color_min = min(df.val) - color_buf
  color_max = max(df.val) + color_buf
  # color_min = 0
  # color_max = 1

  # print(color_min)
  # print(color_max)
  norm = cl.Normalize(vmin=color_min, vmax=color_max, clip=False) # clip=False is default
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
  mapper = cm.ScalarMappable(norm=norm, cmap=cmap)  # cmap color range
  # https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable
  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

  df['color'] = [mapper.to_rgba(v) for v in df.val]

  cldf = pd.DataFrame()
  cldf['min'] = [color_min]
  cldf['max'] = [color_max]
  cldf['cmap'] = [cmap]

  # print(cldf)
  t0 = timer_restart(t0, 'colormap')


  # plot result
  with plt.style.context(("seaborn", "ggplot")):
    world.plot(figsize=(18,10),
                color="white",
                edgecolor = "grey")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    plt.xlim((-180,180)) # default
    plt.ylim((-90,90)) # default

    xlim1 = plt.xlim()
    ylim1 = plt.ylim()

    # bounds = basisregion.bounds # (minx, miny, maxx, maxy) 
    bounds = (min(df.lon), min(df.lat), max(df.lon), max(df.lat))

    plt.xlim((bounds[0],bounds[2]))
    plt.ylim((bounds[1],bounds[3]))

    ms = marker_size(plt, xlim1, ylim1)
    # print(ms)
    plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')
    plt.colorbar(mapper)

    # now = utc_time_filename()
    t0 = timer_restart(t0, 'create plot')
    save_df_plt(plt, df, cldf, stats, outputDir, title + '-' + utc_time_filename())

    t0 = timer_restart(t0, 'save outfiles')
    t1 = timer_restart(t1, 'total time')

    # plt.show()

    

  # after plot



