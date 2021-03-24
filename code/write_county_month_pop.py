
import numpy as np

import geopandas as gpd
import pandas as pd

import os, pathlib, re

import time
import datetime as dt

import io

from pandas.core.algorithms import isin


def save_df(df, folderPath, name):
  df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  # df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format=None)
  # fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  # fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
  # fd.close()
  # fd = h5py.File(os.path.join(folderPath, name + '.hdf5'),'w')
  # fd.create_dataset('data', data=df)
  # fd.close()


def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()


def GEOID_string_state_county(state, county):
  state = str(state)
  county = str(county)
  while len(state) != 2:
    state = '0' + state
  while len(county) != 3:
    county = '0' + county
  return state + county
  
def GEOID_string_geoid(geo):
  geo = str(geo)
  while len(geo) != 5:
    geo = '0' + geo
  return geo
  

def GEOID_to_STATEFP(geoid):
  return geoid[0:2]



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

def county_changes_deaths_reset_index(df):
  # df = df[df.GEOID <= '56']
  # df = df[df.GEOID != '02']
  # df = df[df.GEOID != '15']

  # 51515 put into   51019 (2013)
  #   51019 in all
  #   51515 in 1998-2009 !

  temp1 = df[df.GEOID == '51515']
  if len(temp1.GEOID) != 0:
    temp2 = df[df.GEOID == '51019']
    print(temp2)

  # 51560 put into   51005 (2013)
  #   51005 in all
  #   51560 in 1998-1999 !

  temp1 = df[df.GEOID == '51560']
  if len(temp1.GEOID) != 0:
    temp2 = df[df.GEOID == '51005']
    print(temp2)

  # 46113 renamed to 46102 (2014)
  #   46113 in 1998-2009
  #   46102 in 2010-2019

  temp1 = df[df.GEOID == '46113']
  if len(temp1.GEOID) != 0:
    temp2 = df[df.GEOID == '46102']
    print(temp2)


  # 08014 created from parts of 08001, 08013, 08059, 08123 (2001)
  #   08014 in 2000-2019
  # ignore - no unsuppressed data for it

  return df.reset_index(drop=True)



def parse_lines_deaths(path):
  lines = open(path).readlines()
  lines = [list(filter(None, re.split('\t|\n|"|/',l))) for l in lines]
  for l in lines:
    if 'Suppressed'in l:
      l[l.index('Suppressed')] = '-1'
  lines = [[item for item in line if (item == '-1' or str.isnumeric(item))] for line in lines[1:lines.index(['---'])]]
  return [[l[0], l[1] + l[2], l[3]] for l in lines]

def deaths_by_date_geoid(title):
  linesAll = []
  if os.path.isdir(title):
    filenames = os.listdir(title)
    for filename in filenames:
      linesAll += parse_lines_deaths(os.path.join(title, filename))
  else:
    title += '.txt'
    linesAll = parse_lines_deaths(title)
  ret = pd.DataFrame(linesAll, columns=['GEOID', 'YYYYMM', 'deaths']).sort_values(by=['YYYYMM','GEOID']).reset_index(drop=True)
  ret['deaths'] = pd.to_numeric(ret['deaths'])
  return ret




if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')

  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  
  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')

  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'population')

  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title


  testing = False

  

  t0 = timer_start()
  t1 = t0

  # pd.set_option('display.max_columns', None)
  

  # for testing
  data_GEOIDs = []
  all_filenames = []
  
  # 1998 - 1999 data files - (write 1999 monthly pop)
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('stch-icen') and name.endswith('.txt')]
  for filename in filenames:
    outFileName = filename.split('.')[0] + '_totals'
    if outFileName + '.csv' in os.listdir(usCensusDir):
      fd = pd.read_csv(os.path.join(usCensusDir, outFileName + '.csv'))
      fd['GEOID'] = [GEOID_string_geoid(item) for item in fd['GEOID']]
    else:
      yyyymm = filename[-8:-4]+'07'
      fd = open(os.path.join(usCensusDir, filename), 'r').read()
      fd = pd.read_csv(io.StringIO(fd), names=['GEOID','AgeGroup','RaceSex','Ethnicity', yyyymm], sep='\s+')
      fd['GEOID'] = [GEOID_string_geoid(item) for item in fd['GEOID']]
      fd = clean_states_reset_index(fd)
      geoids = sorted(list(set(fd['GEOID'])))
      sums = []
      for geoid in geoids:
        temp = fd[fd['GEOID'] == geoid]
        sums.append(np.sum(temp[yyyymm]))
      fd = pd.DataFrame()
      fd['GEOID'] = geoids
      fd[yyyymm] = sums
      save_df(fd, usCensusDir, outFileName)
    print(fd)
    if testing:
      data_GEOIDs.append(set(fd['GEOID'])) #
      all_filenames.append(filename) #
      

  t0 = timer_restart(t0, '1998 - 1999 data files')

  # 2000 - 2019 data files
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('co-est') and name.endswith('.csv')]
  for filename in filenames:
    fd = pd.read_csv(os.path.join(usCensusDir, filename))
    yearColumns = [item for item in list(fd.keys()) if item.startswith('POPESTIMATE')][:10]
    yearColumns2 = []
    # print(yearColumns)
    fd['GEOID'] = [GEOID_string_state_county(state,county) for state,county in zip(fd['STATE'], fd['COUNTY'])]
    fd = fd[~fd['GEOID'].str.endswith('000') ]
    fd = clean_states_reset_index(fd)
    for c in yearColumns:
      fd[c[-4:]+'07'] = fd[c]
      yearColumns2.append(c[-4:]+'07')
    fd = fd[['GEOID']+yearColumns2]
    print(fd)
    if testing:
      data_GEOIDs.append(set(fd['GEOID'])) #
      all_filenames.append(filename) #

  t0 = timer_restart(t0, '2000 - 2019 data files')


  # orig pop numbers are for July 1; write other months based on avg monthly net change
  

  if testing:
    countyMapFiles = ['cb_2019_us_county_500k', 'co99_d00']
    countyMapData = dict()

    for filename in countyMapFiles:
      fd = gpd.read_file(os.path.join(usaDir, filename, filename + '.shp'))
      if 'GEOID' not in fd.keys():
        fd['GEOID'] = [GEOID_string_state_county(state,county) for state,county in zip(fd['STATE'], fd['COUNTY'])]
      fd = fd.sort_values(by=['GEOID'])
      countyMapData[filename] = clean_states_reset_index(fd)
      data_GEOIDs.append(set(countyMapData[filename].GEOID)) #
      all_filenames.append(filename) #

    t0 = timer_restart(t0, 'map files')

    deaths_GEOIDs = dict()
    countyData = deaths_by_date_geoid(os.path.join(cdcWonderDir, countyTitle)) # has missing rows
    countyData = clean_states_reset_index(countyData)
    # print(countyData)
    dates = set(countyData['YYYYMM'])
    # print(dates, len(dates))
    for date in sorted(list(dates)):
      deaths_GEOIDs[date] = set(countyData[countyData['YYYYMM'] == date]['GEOID'])

    t0 = timer_restart(t0, 'disease data files')

    data_GEOIDs_union = set().union(*[i for i in data_GEOIDs])
    print(len(data_GEOIDs_union), '\nin data_GEOIDs_union')
    for i in range(len(data_GEOIDs)):
      print('not in', all_filenames[i], '\t', len(data_GEOIDs[i]), sorted(list(data_GEOIDs_union - data_GEOIDs[i])))

    t0 = timer_restart(t0, 'show differences')

    deaths_GEOIDs_union = set().union(*[deaths_GEOIDs[i] for i in deaths_GEOIDs.keys()])
    print(len(deaths_GEOIDs_union), '\nin deaths_GEOIDs_union')
    for i in range(len(data_GEOIDs)):
      print('not in', all_filenames[i], '\t', len(data_GEOIDs[i]), sorted(list(deaths_GEOIDs_union - data_GEOIDs[i])))

    startMonth = '01'
    deaths_GEOIDs_union_subsets = dict()
    subset_date_range = ['1998', '1999', ('2000','2010'), ('2010','2020'), ('1998','2020'), ('1998','2001'),]
    print('\nin deaths_GEOIDs_union_subsets')
    for i in range(len(data_GEOIDs)):
      print('----', subset_date_range[i])
      if isinstance(subset_date_range[i], str):
        temp = [deaths_GEOIDs[j] for j in deaths_GEOIDs.keys() if j == str(subset_date_range[i] + startMonth) ]
      else:
        temp = [deaths_GEOIDs[j] for j in deaths_GEOIDs.keys() if j >= str(subset_date_range[i][0] + startMonth) and j < str(subset_date_range[i][1] + startMonth) ]

      deaths_GEOIDs_union_subsets[all_filenames[i]] = set().union(*temp)
      print(len(deaths_GEOIDs_union_subsets[all_filenames[i]]))
      print('not in', all_filenames[i], '\t', len(data_GEOIDs[i]), sorted(list(deaths_GEOIDs_union_subsets[all_filenames[i]] - data_GEOIDs[i])), '\tissubset =', deaths_GEOIDs_union_subsets[all_filenames[i]] <= data_GEOIDs[i])

    t0 = timer_restart(t0, 'show issubsets')



  


  t1 = timer_restart(t1, 'total time')


