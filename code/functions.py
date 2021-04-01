import numpy as np

import geopandas as gpd
import pandas as pd

import os, pathlib, re, netCDF4

import time
import datetime as dt



def save_df(df, folderPath, name, ext):
  print('save', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  elif ext == 'hdf5':
    df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format='fixed')
  # df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  # df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format=None)
  # fd = pd.HDFStore(os.path.join(folderPath, name + '.hdf5'))
  # fd.put('data', df, format='table', data_columns=True, complib='blosc', complevel=5)
  # fd.close()
  # fd = h5py.File(os.path.join(folderPath, name + '.hdf5'),'w')
  # fd.create_dataset('data', data=df)
  # fd.close()

def read_df(folderPath, name, ext):
  print('read', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    return pd.read_csv(os.path.join(folderPath, name + '.csv'))
  elif ext == 'hdf5':
    return pd.read_hdf(os.path.join(folderPath, name + '.hdf5'))
  elif ext == 'nc':
    return netCDF4.Dataset(os.path.join(folderPath, name + '.nc'))

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

def GEOID_string_state_county(state, county):
  state = str(state)
  county = str(county)
  while len(state) != 2:
    state = '0' + state
  while len(county) != 3:
    county = '0' + county
  return state + county
  
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
    # Edit: if not in data, add and set pop to 0 #####################
  temp1 = df[df.GEOID == '08014']
  # if len(temp1.GEOID) != 0:
  #   df = df.drop([temp1.index[0]])
  if len(temp1.GEOID) == 0:
    df = df.append(pd.DataFrame([['08014'] + [0]*len(dates)], columns=['GEOID']+dates))

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