
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

def county_changes_deaths_reset_index(df): # see code/write_county_month_pop_testing.txt
  dates = [i for i in df.keys() if i != 'GEOID']

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
  temp1 = df[df.GEOID == '08014']
  if len(temp1.GEOID) != 0:
    df = df.drop([temp1.index[0]])

  return df.sort_values(by='GEOID').reset_index(drop=True)

def month_str(month):
  if month < 10:
    return '0' + str(month)
  else:
    return str(month)


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
  doChanges = True
  origDataMonth = '07'

  all_dfs = []

  t0 = timer_start()
  t1 = t0

  
  
  # for testing
  data_GEOIDs = []
  all_filenames = []
  
  # 1998 - 1999 data files - (write 1999 monthly pop)
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('stch-icen') and name.endswith('.txt')]
  for filename in filenames:
    print(filename)
    outFileName = filename.split('.')[0] + '_totals'
    if outFileName + '.csv' in os.listdir(usCensusDir):
      df = pd.read_csv(os.path.join(usCensusDir, outFileName + '.csv'))
      df['GEOID'] = [GEOID_string_geoid(item) for item in df['GEOID']]
    else:
      yyyymm = filename[-8:-4]+origDataMonth
      df = open(os.path.join(usCensusDir, filename), 'r').read()
      df = pd.read_csv(io.StringIO(df), names=['GEOID','AgeGroup','RaceSex','Ethnicity', yyyymm], sep='\s+')
      df['GEOID'] = [GEOID_string_geoid(item) for item in df['GEOID']]
      df = clean_states_reset_index(df)
      geoids = sorted(list(set(df['GEOID'])))
      sums = []
      for geoid in geoids:
        temp = df[df['GEOID'] == geoid]
        sums.append(np.sum(temp[yyyymm]))
      df = pd.DataFrame()
      df['GEOID'] = geoids
      df[yyyymm] = sums
      save_df(df, usCensusDir, outFileName)
    # print(df)

    if doChanges:
      df = county_changes_deaths_reset_index(df)

    all_dfs.append(df)
    
    if testing:
      data_GEOIDs.append(set(df['GEOID'])) #
      all_filenames.append(filename) #

  t0 = timer_restart(t0, '1998 - 1999 data files')

  # 2000 - 2019 data files
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('co-est') and name.endswith('.csv')]
  for filename in filenames:
    print(filename)
    df = pd.read_csv(os.path.join(usCensusDir, filename))
    yearColumns = [item for item in list(df.keys()) if item.startswith('POPESTIMATE')][:10]
    yearColumns2 = []
    # print(yearColumns)
    df['GEOID'] = [GEOID_string_state_county(state,county) for state,county in zip(df['STATE'], df['COUNTY'])]
    df = df[~df['GEOID'].str.endswith('000') ]
    df = clean_states_reset_index(df)
    for c in yearColumns:
      df[c[-4:]+origDataMonth] = df[c]
      yearColumns2.append(c[-4:]+origDataMonth)
    df = df[['GEOID']+yearColumns2]
    # print(df)

    if doChanges:
      df = county_changes_deaths_reset_index(df)

    all_dfs.append(df)

    if testing:
      data_GEOIDs.append(set(df['GEOID'])) #
      all_filenames.append(filename) #

  t0 = timer_restart(t0, '2000 - 2019 data files')


  if testing:
    countyMapFiles = ['cb_2019_us_county_500k', 'co99_d00']
    countyMapData = dict()

    for filename in countyMapFiles:
      df = gpd.read_file(os.path.join(usaDir, filename, filename + '.shp'))
      if 'GEOID' not in df.keys():
        df['GEOID'] = [GEOID_string_state_county(state,county) for state,county in zip(df['STATE'], df['COUNTY'])]
      df = df.sort_values(by=['GEOID'])
      countyMapData[filename] = clean_states_reset_index(df)
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


  if doChanges: # and testing
    all_GEOIDs = [list(df['GEOID']) for df in all_dfs]
    all_GEOIDs_union = sorted(set().union(*[set(i) for i in all_GEOIDs]))
    for i in range(len(all_GEOIDs)): # see if all geoids equal + sorted across all files (by comparing to union)
      print(all_GEOIDs[i] == all_GEOIDs_union, len(all_GEOIDs[i]), len(all_GEOIDs_union))

    t0 = timer_restart(t0, 'show if GEOIDs equal across all files')


  
  # orig pop numbers are for July 1; write other months based on avg monthly net change
  # V4NA03 pm2.5 data range: 2000-01 to 2018-12

  data = pd.DataFrame(all_dfs[0])
  for i in range(1, len(all_dfs)):
    dates = [i for i in all_dfs[i] if i != 'GEOID']
    data = pd.concat([data, all_dfs[i][dates]], axis=1)
  # print(data)


  dates = sorted(i for i in data if i != 'GEOID')
  # print(dates)

  monthSeq = np.roll([month_str(month) for month in range(1,13)], -int(origDataMonth))
  # print(monthSeq)

  for i in range(len(dates[:-1])):
    year = dates[i][:4]
    prev = year + monthSeq[-1]

    monthlyChange = pd.DataFrame(data[dates[i + 1]] - data[dates[i]]) / 12
    # print(monthlyChange)

    startyyyymm = monthSeq[-1]
    for month in monthSeq[:-1]:
      if month == '01':
        year = dates[i + 1][:4]
      yyyymm = year + month
      data[yyyymm] = data[prev] + monthlyChange[0]
      prev = yyyymm

  data = data.sort_index(axis=1)
  data = data[ ['GEOID'] + [i for i in data.columns if i != 'GEOID'] ]

  dates = [i for i in data if i != 'GEOID']
  # print(dates)

  lastindex = len(data)

  data.loc[lastindex] = data.sum()
  data.loc[lastindex,'GEOID'] = ''
  # print(data)
  # exit()

  save_df(data, usCensusDir, 'county_pop_1999_2019')

  t1 = timer_restart(t1, 'total time')


