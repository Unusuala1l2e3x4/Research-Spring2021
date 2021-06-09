
import numpy as np

import geopandas as gpd
from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, io, sys, re, copy


import importlib
fc = importlib.import_module('functions')



if __name__ == "__main__":

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  outputDir = os.path.join(ppPath, 'US Census Bureau', 'SAIPE State and County Estimates')


  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')

  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets', 'Chronic lower respiratory diseases')
  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'SAIPE State and County Estimates', 'downloads')

  # PARAMS
  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019, suppressed estimates'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title

  testing = False
  origDataMonth = '07'
  suppValString = '-1'
  ext = 'hdf5'
  # END PARAMS
  

  countyData = fc.read_df(cdcWonderDir, countyTitle, ext)
  countyData = fc.makeCountyFileGEOIDs(countyData)
  countyData = fc.clean_states_reset_index(countyData)
  countyData = fc.county_changes_deaths_reset_index(countyData)


  all_dfs = []

  t0 = fc.timer_start()
  t1 = t0


  startDate = '2000'
  endDate = '2019'

  outFileName = 'TENA_county_median_income_'+startDate+'_'+endDate

  outData = pd.DataFrame(countyData.GEOID, columns=['GEOID'])
  # print(outData)


  # 2000-2002 data files
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('est') and name.endswith('all.dat')]
  for filename in filenames:
    yyyy = '20'+re.search(r'\d+', filename).group()
    if yyyy < startDate or yyyy > endDate:
      continue

    temp = pd.read_csv(io.StringIO(open(os.path.join(usCensusDir, filename), 'r').read()), sep="\n", skiprows=1, names=['line']) # , usecols=[0,1, 3,4,5] + list(range(133, 139)) # , names=['TIME','XGSM']
    temp['GEOID'] = [ str(i[0:2] + i[3:6]).replace(' ','0') for i in temp.line]
    temp[yyyy] = [float(i[133:139].replace('.','NaN')) for i in temp.line] # only missing values in Hawaii, which is removed
    temp = temp[['GEOID',yyyy]]
    temp = temp[~temp.GEOID.str.endswith('000')]
    temp = fc.clean_states_reset_index(temp)
    temp = fc.county_changes_deaths_reset_index(temp)

    assert list(temp.GEOID) == list(outData.GEOID)
    assert yyyy not in outData.keys()
    outData[yyyy] = temp[yyyy]
  
  # 2003-2019
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('est') and name.endswith('all.xls')]
  for filename in filenames:
    yyyy = '20'+re.search(r'\d+', filename).group()
    if yyyy < startDate or yyyy > endDate:
      continue

    temp = None
    if yyyy < '2012':
      if yyyy < '2005':
        temp = pd.read_excel(os.path.join(usCensusDir, filename), header=1, usecols = ['State FIPS', 'County FIPS', 'Median Household Income'])
      else:
        temp = pd.read_excel(os.path.join(usCensusDir, filename), header=2, usecols = ['State FIPS', 'County FIPS', 'Median Household Income'], skipfooter=3)
      temp['GEOID'] = ['{:02d}'.format(int(x))+'{:03d}'.format(int(y)) for x,y in zip(temp['State FIPS'], temp['County FIPS'])]
    else:
      if yyyy < '2013':
        temp = pd.read_excel(os.path.join(usCensusDir, filename), header=2, usecols = ['State FIPS Code', 'County FIPS Code', 'Median Household Income'])
      else:
        temp = pd.read_excel(os.path.join(usCensusDir, filename), header=3, usecols = ['State FIPS Code', 'County FIPS Code', 'Median Household Income'])
      temp['GEOID'] = ['{:02d}'.format(int(x))+'{:03d}'.format(int(y)) for x,y in zip(temp['State FIPS Code'], temp['County FIPS Code'])]

    temp[yyyy] = [float(i) for i in temp['Median Household Income'].replace([NaN, '.'], 'NaN')]  
    temp = temp[['GEOID',yyyy]]
    temp = temp[~temp.GEOID.str.endswith('000')]
    temp = fc.clean_states_reset_index(temp)
    temp = fc.county_changes_deaths_reset_index(temp)

    assert list(temp.GEOID) == list(outData.GEOID)
    assert yyyy not in outData.keys()
    outData[yyyy] = temp[yyyy]


  # only 1 missing value:
  # 08014 not in 2000 but is in everywhere else -> copy 20001 value
  # outData.loc[outData.GEOID == '08014', '2000'] = outData.loc[outData.GEOID == '08014', '2001']


  years = [i for i in outData.keys() if i != 'GEOID']
  # print(outData[outData.GEOID.str.startswith('0801')])
  for i in range(len(outData)):
    if True in np.isnan(list(outData.loc[i,years])):
      temp = pd.DataFrame()
      temp['a'] = list(outData.loc[i,years])
      temp.index = pd.to_datetime(years, format='%Y', errors='coerce')
      temp = temp.interpolate(method='linear', limit_direction='both')
      outData.loc[i,years] = list(temp.a)
      # print(outData.loc[i,years])

  outData.loc[:,years] = outData.loc[:,years].astype(int)

  # print(outData[outData.GEOID.str.startswith('0801')])
  # exit()

  

  t0 = fc.timer_restart(t0, 'write county data')

  print(outData)
  fc.save_df(outData, outputDir, outFileName+'_yearly', 'hdf5')
  fc.save_df(outData, outputDir, outFileName+'_yearly', 'csv')


  

  dates = []


  for year in years:
    for i in range(1,13):
      month = '{:02d}'.format(i)
      dates.append(year+month)
      outData[year+month] = outData[year]

  outData = outData[['GEOID']+dates]

  print(outData)

  fc.save_df(outData, outputDir, outFileName, 'hdf5')
  fc.save_df(outData, outputDir, outFileName, 'csv')

  t1 = fc.timer_restart(t1, 'total time')


