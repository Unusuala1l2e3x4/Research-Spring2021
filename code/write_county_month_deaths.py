
import numpy as np

import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, re, copy

import time
import datetime as dt

import io

from pandas.core.algorithms import isin
from pandas.io.pytables import read_hdf


def save_df(df, folderPath, name, ext):
  print('save', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  elif ext == 'hdf5':
    df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format='fixed')

def read_df(folderPath, name, ext):
  print('read', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    return pd.read_csv(os.path.join(folderPath, name + '.csv'))
  elif ext == 'hdf5':
    return pd.read_hdf(os.path.join(folderPath, name + '.hdf5'))

def is_in_dir(folderPath, name, ext):
  return name + '.' + ext in os.listdir(folderPath)

def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()

def countyGEOIDstring(geo):
  geo = str(geo)
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
  temp1 = df[df.GEOID == '08014']
  if len(temp1.GEOID) != 0:
    df = df.drop([temp1.index[0]])

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


def make_geoid_dates_df(data):
  GEOIDs = sorted(set(data.GEOID))
  dates = sorted(set(data.YYYYMM))
  ret = pd.DataFrame(None, columns=['GEOID'] + dates)
  ret.GEOID = GEOIDs

  for date in dates:
    temp = data[data.YYYYMM == date].reset_index(drop=True)
    j = 0
    for i in range(len(ret)):
      if j == len(temp):
        break
      if ret.GEOID[i] == temp.GEOID[j]:
        ret[date][i] = temp.deaths[j]
        j += 1

  ret = clean_states_reset_index(ret)
  ret = county_changes_deaths_reset_index(ret)

  return ret


def limit_dates(df, dates):
  if 'STATEFP' in df.keys():
    return df[['GEOID', 'STATEFP']+dates]
  else:
    return df[['GEOID']+dates]



if __name__ == "__main__":
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  # pmDir = os.path.join(ppPath, 'Global Annual PM2.5 Grids')
  outputDir = os.path.join(pPath, 'plot_usa-outfiles')

  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')

  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')
  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'population')

  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title

  suppValString = '-1'
  ext = 'csv' # csv/hdf5
  
  testing = False
  # END PARAMS


  all_dfs = []

  t0 = timer_start()
  t1 = t0

  countyPop = pd.read_hdf(os.path.join(usCensusDir, 'TENA_county_pop_1999_2019.hdf5'))



  statePop = pd.read_hdf(os.path.join(usCensusDir, 'TENA_state_pop_1999_2019.hdf5'))

  
  if is_in_dir(cdcWonderDir, countyTitle, ext):
    countyData = read_df(cdcWonderDir, countyTitle, ext)
  else:
    countyData = make_geoid_dates_df(deaths_by_date_geoid(os.path.join(cdcWonderDir, countyTitle), suppValString)) # has missing rows
    save_df(countyData, cdcWonderDir, countyTitle, ext)
  countyData = makeCountyFileGEOIDs_STATEFPs(countyData)

  if is_in_dir(cdcWonderDir, stateTitle, ext):
    stateData = read_df(cdcWonderDir, stateTitle, ext)
  else:
    stateData = make_geoid_dates_df(deaths_by_date_geoid(os.path.join(cdcWonderDir, stateTitle), suppValString)) # no missing rows  
    save_df(stateData, cdcWonderDir, stateTitle, ext)
  stateData = makeStateFileGEOIDs(stateData)

  if testing:
    countyGEOID = set(countyData.GEOID)
    stateGEOID = set(stateData.GEOID)
    popGEOID = set(countyPop.GEOID)
    print('countyGEOID <= popGEOID\t', countyGEOID <= popGEOID)
    print('len(stateGEOID), len(countyGEOID), len(popGEOID) =', len(stateGEOID), len(countyGEOID), len(popGEOID))
    countySTATEFP = set([i[0:2] for i in countyData.GEOID])
    stateSTATEFP = set([i for i in stateData.GEOID])
    popSTATEFP = set([i[0:2] for i in countyPop.GEOID])
    print('stateSTATEFP <= popSTATEFP\t', stateSTATEFP <= popSTATEFP, len(stateSTATEFP), len(popSTATEFP))
    print('countySTATEFP <= popSTATEFP\t', countySTATEFP <= popSTATEFP, len(countySTATEFP), len(popSTATEFP))
    print('countySTATEFP == stateSTATEFP\t', countySTATEFP == stateSTATEFP, len(countySTATEFP), len(stateSTATEFP))

  # print(countyPop)
  # print(countyData)
  # print(stateData)

  deathsDates = sorted(i for i in countyData if i != 'GEOID' and i != 'STATEFP')
  popDates = sorted(i for i in countyPop if i != 'GEOID' and i != 'STATEFP')
  dates = sorted(set(deathsDates) & set(popDates))
  

  if is_in_dir(cdcWonderDir, 'stateDataUnsup', ext):
    stateDataUnsup = read_df(cdcWonderDir, 'stateDataUnsup', ext)
  else:
    stateDataUnsup = copy.deepcopy(stateData)
    for date in deathsDates:
      stateDataUnsup.loc[:, date] = np.ravel([np.nansum(countyData.loc[countyData.STATEFP == statefp, date]) for statefp in stateData.GEOID])
    save_df(stateDataUnsup, cdcWonderDir, 'stateDataUnsup', ext)
  stateDataUnsup = makeStateFileGEOIDs(stateDataUnsup)

  # print(stateData)
  # print(stateDataUnsup)
  # exit()

  # for all dfs, constrain range of dates to intersection
  statePop, countyPop, stateData, countyData, stateDataUnsup = [limit_dates(df, dates) for df in [statePop, countyPop, stateData, countyData, stateDataUnsup]]
  
  
  # print(countyData)
  # print(countyPop)

  countyPopUnsup = countyData.reindex_like(countyPop)
  countyPopUnsup.loc[range(len(countyData.GEOID), len(countyPop.GEOID)), 'GEOID'] = list(set(countyPop.GEOID) - set(countyData.GEOID))
  countyPopUnsup = makeCountyFileGEOIDs_STATEFPs(countyPopUnsup)
  countyPopUnsup = countyPopUnsup.sort_values(by='GEOID').reset_index(drop=True)
  countyPopUnsup.loc[:, dates] = countyPop.loc[:, dates] * (countyPopUnsup.loc[:, dates] / countyPopUnsup.loc[:, dates])
  
  # print(countyPopUnsup)
  # print(countyPop)

  if is_in_dir(cdcWonderDir, 'statePopUnsup', ext):
    statePopUnsup = read_df(cdcWonderDir, 'statePopUnsup', ext)
  else:
    statePopUnsup = copy.deepcopy(statePop)
    for date in dates:
      statePopUnsup.loc[:, date] = np.ravel([np.nansum(countyPopUnsup.loc[countyPopUnsup.STATEFP == statefp, date]) for statefp in statePop.GEOID])
    save_df(statePopUnsup, cdcWonderDir, 'statePopUnsup', ext)
  statePopUnsup = makeStateFileGEOIDs(statePopUnsup)


  
  # print(statePopUnsup)
  # print(statePop)

  d = copy.deepcopy(stateData)
  d.loc[:, dates] = d.loc[:, dates] - stateDataUnsup.loc[:, dates]
  print(d)

  p = copy.deepcopy(statePop)
  p.loc[:, dates] = p.loc[:, dates] - statePopUnsup.loc[:, dates]
  print(p)

  # d


  r = copy.deepcopy(d)
  r.loc[:, dates] = r.loc[:, dates] / p.loc[:, dates]
  print(r)



  # countyData



  t1 = timer_restart(t1, 'total time')


