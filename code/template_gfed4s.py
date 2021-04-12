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

# https://shapely.readthedocs.io/en/latest/manual.html
from shapely.geometry import shape, GeometryCollection, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

import importlib
fc = importlib.import_module('functions')

# print(fd)
# print(fd.keys())
# ['ancill', 'biosphere', 'burned_area', 'emissions', 'lat', 'lon']

# print(fd['biosphere/01'])
# print(fd['biosphere']['01'])

# print(fd['emissions']['08/partitioning'])


regionNums = {'BONA':1, 'TENA':2, 'CEAM':3, 'NHSA':4, 'SHSA':5, 'EURO':6, 'MIDE':7, 'NHAF':8, 'SHAF':9, 'BOAS':10, 'CEAS':11, 'SEAS':12, 'EQAS':13, 'AUST':14}


def burned_area(fd, month, dataset_name):
  """[1997 - 2016]

  /<month> (01, 02, .. , 12)
      /burned_fraction
      /source

  burned_fraction: fraction of each grid cell that burned in that month according to the GFED4s burned area data
  
  source (of burned area): what data was used to construct the burned area maps excluding small fires. In general, ATSR and VIRS data was used before 2001, MODIS after 2001. This solely concerns the GFED4 burned area dataset.
  """
  return fd['burned_area'][fc.month_str(month)][dataset_name]


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
      DEFO (Tropical forest fires [deforestation and degradation]); PEAT (Peat fires); AGRI (Agricultural waste burning)
      """
  return fd['emissions'][fc.month_str(month)][dataset_name]


def biosphere(fd, month, dataset_name):
  """biosphere fluxes (g C m^-2 month^-1) [1997 - 2016]:

  /<month> (01, 02, .. , 12)
      /NPP: net primary production

      /Rh: heterotrophic respiration

      /BB: fire emissions
      """
  return fd['biosphere'][fc.month_str(month)][dataset_name]


def ancill(fd, dataset_name):
  """grid_cell_area: how many m^2 each grid cell contains

  basis_regions: the 14 basis regions
  """
  return fd['ancill'][dataset_name]