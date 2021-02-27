# https://gis.stackexchange.com/questions/253499/how-to-plot-gridded-map-from-lat-lon-and-fill-values-in-r
# https://stackoverflow.com/questions/61002973/plot-gridded-map-with-lat-lon-and-fill-values-in-csv-file-in-python
# https://towardsdatascience.com/using-gbif-data-and-geopandas-to-plot-biodiversity-trends-6325ab6632ee


# geopandas, geoplot
  # https://james-brennan.github.io/posts/fast_gridding_geopandas/
  # https://geopandas.readthedocs.io/en/latest/gallery/plotting_with_geoplot.html
  # https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html
  # https://james-brennan.github.io/posts/fast_gridding_geopandas/
  

# zip files
# https://stackoverflow.com/questions/56786321/read-multiple-csv-files-zipped-in-one-file


# hdf5 t; numpy conversion
# https://stackoverflow.com/questions/49906351/python-reading-hdf5-dataset-into-a-list-vs-numpy-array

# geojson
# https://eric.clst.org/tech/usgeojson/
 

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


# ['ancill', 'biosphere', 'burned_area', 'emissions', 'lat', 'lon']


def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)


def burned_area(fd, month, dataset_name):
  """[1997 - 2016]

  /<month> (01, 02, .. , 12)
      /burned_fraction
      /source

  burned_fraction: fraction of each grid cell that burned in that month according to the GFED4s burned area data
  
  source (of burned area): what data was used to construct the burned area maps excluding small fires. In general, ATSR and VIRS data was used before 2001, MODIS after 2001. This solely concerns the GFED4 burned area dataset.
  """


  # print(fd)
  # print(fd.keys())
  # ['ancill', 'biosphere', 'burned_area', 'emissions', 'lat', 'lon']

  # print(fd['biosphere/01'])
  # print(fd['biosphere']['01'])

  # print(fd['emissions']['08/partitioning'])

  return fd['burned_area'][month_str(month)][dataset_name]

def emissions(fd, month, dataset_name):
  """/<month> (01, 02, .. , 12)
      /DM

      /C

      /small_fire_fraction

      /daily_fraction [2003 onwards]
          /day_1
          /day_2
          /etc. (total of n days in month)

      /diurnal_cycle [2003 onwards]
          /UTC_0-3h
          /UTC_3-6h
          /etc. (total of 8)

      /partitioning ( source: SAVA/BORF/TEMF/DEFO/PEAT/AGRI (total of 6) )
          /C_<source> [1997 - 2016]
          /DM_<source>
          
  DM: dry matter (kg DM m^-2 month^-1)

  C: carbon (g C m^-2 month^-1)
  
  small_fire_fraction (unitless): indicates what fraction of total emissions stemmed from the small fire burned area.
    GFED4 emissions can be calculated by subtracting this fraction from total emissions, but we recommend using GFED4s emissions.
      Note that GFED4 burned area cannot be calculated this was for various reasons, please use the original GFED4 burned area datasets for this.

  daily_fraction (unitless): indicates what fraction of total emissions was emitted in the different days of that month

  diurnal_cycle (unitless): gives the partitioning of the daily emissions over 8 threehour windows (UTC), this is uniform over the month.

  partitioning: for both C and DM (all unitless):
    SAVA (Savanna, grassland, and shrubland fires); BORF (Boreal forest fires); TEMF (Temperature forest fires);
      DEFO (Tropical forest fires [deforestation and degradation]); PEAT (Peat fires); AGRI (Agricultural waste burning)"""
  return fd['emissions'][month_str(month)][dataset_name]

def biosphere(fd, month, dataset_name):
  """biosphere fluxes (g C m^-2 month^-1) [1997 - 2016]:

  /<month> (01, 02, .. , 12)
      /NPP: net primary production

      /Rh: heterotrophic respiration

      /BB: fire emissions"""
  return fd['biosphere'][month_str(month)][dataset_name]

def ancill(fd, dataset_name):
  """grid_cell_area: how many m^2 each grid cell contains

  basis_regions: the 14 basis regions"""
  return fd['ancill'][dataset_name]



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
  print(time.time() - t0)

def timer_restart(t0):
  timer_elapsed(t0)
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
  

def save_plt(plt, dest, name, now):
  plt.savefig(dest + name + '-' + now + '.png')


def save_df(df, cldf, dest, name, now):
  fd = pd.HDFStore(dest + name + '-' + now + '.hdf5')
  del df['color']
  fd.put('data', df, format='table', data_columns=True)
  fd.put('colormap', cldf, format='table', data_columns=True)
  fd.close()

  
  

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



world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

if __name__ == "__main__":
  startYear, startMonth = int(sys.argv[1]), int(sys.argv[2])
  endYear, endMonth = int(sys.argv[3]), int(sys.argv[4])
  group = sys.argv[5]
  dataset = sys.argv[6]
  cmap = sys.argv[7]

  gfedDir = str(pathlib.Path(__file__).parent.parent.absolute()) + '/GFED4s/'
  imagesDir = str(pathlib.Path(__file__).parent.absolute()) + '/read_gfed4s-images/'
  dataframesDir = str(pathlib.Path(__file__).parent.absolute()) + '/read_gfed4s-dataframes/'
  shapefilesDir = str(pathlib.Path(__file__).parent.absolute()) + '/shapefiles/'

  basisregionsDir = shapefilesDir + 'basisregions/'
  GFED_basisregions = read_json( basisregionsDir + 'GFED_basis_regions.geo.json')

  # C:\Users\Alex\Documents\GitHub\Research-Spring2021\code\shapefiles\basisregions\GFED_basis_regions.geo.json
  
  filenames = os.listdir(gfedDir)
  
  filenames = sorted(gfed_filenames(filenames))

  # params
  # group = 'burned_area'
  # dataset = 'burned_fraction'
  # startYear, startMonth = 2016, 12
  # endYear, endMonth = 2016, 12
  # cmap = 'cool'
  # end params

  # title = month_str(startMonth) + '-' + str(startYear) + '_' + month_str(endMonth) + '-' + str(endYear) + '_' + group + '_' + dataset
  title = month_str(startMonth) + '-' + str(startYear) + '_' + month_str(endMonth) + '-' + str(endYear) + '_' + dataset

  startDate = dt.date(startYear, startMonth, 1)
  endDate = dt.date(endYear, endMonth, 1)

  currentDate = startDate

  t0 = timer_start()

  df = None

  for filename in filenames:
    if currentDate > endDate:
      break
    elif str(currentDate.year) not in filename: # skip years before startDate
      continue
    elif df is None or currentDate.month == 1: # if first/next file
      # open next file
      # by year

      fd = h5py.File(gfedDir + filename, 'r')

      next_year = next_jan1(currentDate)

      curr_startMonth = currentDate.month
      curr_endMonth = 12 # default
      if endDate < next_year or currentDate == endDate:
        curr_endMonth = endDate.month

      temp_val = [ sum(l) for l in zip(*[flatten_list(fd[group][month_str(month_it)][dataset]) for month_it in range(curr_startMonth, curr_endMonth + 1) ]) ]

      if df is None:
        df = pd.DataFrame(None, columns = ['lat', 'lon', 'val'])
        df.lat = flatten_list(fd['lat'])
        df.lon = flatten_list(fd['lon'])
        df.val = temp_val
      else: # add to df.val
        df.val = [x + y for x, y in zip(df.val, temp_val)]

      currentDate = next_year
      
      fd.close()
      
    
    


  # after reading all months/years
  df = df[df.val != 0.0] # remove zero val
  numMonths = (endDate.year - startDate.year)*12 + (endDate.month - startDate.month) + 1
  df.val = [v / numMonths for v in df.val] # take average

  print(len(df.val))

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
  # https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

  t0 = timer_restart(t0)

  df['color'] = [mapper.to_rgba(v) for v in df.val]

  cldf = pd.DataFrame()
  cldf['min'] = [color_min]
  cldf['max'] = [color_max]
  cldf['cmap'] = [cmap]

  print(cldf)


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

    # plt.xlim((-126,-66)) # custom
    # plt.ylim((23,50)) # custom

    ms = marker_size(plt, xlim1, ylim1)
    # print(ms)
    plt.scatter(df.lon, df.lat, s=ms, c=df.color, alpha=1, linewidths=0, marker='s')
    plt.colorbar(mapper)

    now = utc_time_filename()

    save_plt(plt, imagesDir, title, now)

    t0 = timer_restart(t0)

    save_df(df, cldf, dataframesDir, title, now)

    t0 = timer_restart(t0)

    plt.show()

    

  # after plot



