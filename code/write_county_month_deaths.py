
import numpy as np

from numpy.core.numeric import NaN
import pandas as pd

import os, pathlib, copy, sys

import importlib
fc = importlib.import_module('functions')


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
  return ret


def limit_dates(df, dates):
  if 'STATEFP' in df.keys():
    return df[['GEOID', 'STATEFP']+dates]
  else:
    return df[['GEOID']+dates]



if __name__ == "__main__":
  mode = sys.argv[1]
  ext = sys.argv[2]

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

  suppValString = '-1' #None
  # ext = 'hdf5' # csv/hdf5
  
  testing = False
  # END PARAMS


  all_dfs = []

  t0 = fc.timer_start()
  t1 = t0

  countyPop = pd.read_hdf(os.path.join(usCensusDir, 'TENA_county_pop_1999_2019.hdf5'))
  countyPop = fc.makeCountyFileGEOIDs_STATEFPs(countyPop)

  statePop = pd.read_hdf(os.path.join(usCensusDir, 'TENA_state_pop_1999_2019.hdf5'))
  statePop = fc.makeStateFileGEOIDs(statePop)
  
  if fc.is_in_dir(cdcWonderDir, countyTitle, ext) and mode != 'w':
    countyData = fc.read_df(cdcWonderDir, countyTitle, ext)
  else:
    countyData = make_geoid_dates_df(fc.deaths_by_date_geoid(os.path.join(cdcWonderDir, countyTitle), suppValString)) # has missing rows
    countyData = fc.clean_states_reset_index(countyData)
    countyData = fc.county_changes_deaths_reset_index(countyData)
    fc.save_df(countyData, cdcWonderDir, countyTitle, ext)
  countyData = fc.makeCountyFileGEOIDs_STATEFPs(countyData)

  if fc.is_in_dir(cdcWonderDir, stateTitle, ext) and mode != 'w':
    stateData = fc.read_df(cdcWonderDir, stateTitle, ext)
  else:
    stateData = make_geoid_dates_df(fc.deaths_by_date_geoid(os.path.join(cdcWonderDir, stateTitle), suppValString)) # no missing rows  
    stateData = fc.clean_states_reset_index(stateData)
    fc.save_df(stateData, cdcWonderDir, stateTitle, ext)
  stateData = fc.makeStateFileGEOIDs(stateData)

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
  

  if fc.is_in_dir(cdcWonderDir, 'stateDataUnsup', ext) and mode != 'w':
    stateDataUnsup = fc.read_df(cdcWonderDir, 'stateDataUnsup', ext)
  else:
    stateDataUnsup = copy.deepcopy(stateData)
    for date in deathsDates:
      stateDataUnsup.loc[:, date] = np.ravel([np.nansum(countyData.loc[countyData.STATEFP == statefp, date]) for statefp in stateData.GEOID])
    fc.save_df(stateDataUnsup, cdcWonderDir, 'stateDataUnsup', ext)
  stateDataUnsup = fc.makeStateFileGEOIDs(stateDataUnsup)

  # print(stateData)
  # print(stateDataUnsup)
  # exit()

  # for all dfs, constrain range of dates to intersection
  statePop, countyPop, stateData, countyData, stateDataUnsup = [limit_dates(df, dates) for df in [statePop, countyPop, stateData, countyData, stateDataUnsup]]
  
  
  # print(countyData)
  # print(countyPop)


  countyPopUnsup = countyData.reindex_like(countyPop)
  countyPopUnsup.loc[range(len(countyData.GEOID), len(countyPop.GEOID)), 'GEOID'] = list(set(countyPop.GEOID) - set(countyData.GEOID))
  countyPopUnsup = fc.makeCountyFileGEOIDs_STATEFPs(countyPopUnsup)
  countyPopUnsup = countyPopUnsup.sort_values(by='GEOID').reset_index(drop=True)
  countyData = copy.deepcopy(countyPopUnsup) # including all GEOIDS
  countyPopUnsup = countyPopUnsup.replace([0],NaN)
  countyPopUnsup.loc[:, dates] = countyPop.loc[:, dates] * (countyPopUnsup.loc[:, dates] / countyPopUnsup.loc[:, dates])

  # print(countyPop)

  if fc.is_in_dir(cdcWonderDir, 'statePopUnsup', ext) and mode != 'w':
    statePopUnsup = fc.read_df(cdcWonderDir, 'statePopUnsup', ext)
  else:
    statePopUnsup = copy.deepcopy(statePop)
    for date in dates:
      statePopUnsup.loc[:, date] = np.ravel([np.nansum(countyPopUnsup.loc[countyPopUnsup.STATEFP == statefp, date]) for statefp in statePop.GEOID])
    fc.save_df(statePopUnsup, cdcWonderDir, 'statePopUnsup', ext)
  statePopUnsup = fc.makeStateFileGEOIDs(statePopUnsup)

  t0 = fc.timer_restart(t0, 'get unsuppressed data')

  dS = copy.deepcopy(stateData)
  dS.loc[:, dates] = dS.loc[:, dates] - stateDataUnsup.loc[:, dates]
  # print(dS)

  pS = copy.deepcopy(statePop)
  pS.loc[:, dates] = pS.loc[:, dates] - statePopUnsup.loc[:, dates]
  # print(pS)

  countyData = countyData.replace([0],NaN)
  countyDataNew = copy.deepcopy(countyData)
  countyDataNew.loc[:, dates] = countyData.loc[:, dates] / countyData.loc[:, dates]
  
  countyDataNew = countyDataNew.replace([1], 0)
  countyDataNew = countyDataNew.replace([NaN], 1)
  
  rS = copy.deepcopy(dS)
  rS.loc[:, dates] = rS.loc[:, dates] / pS.loc[:, dates]
  # print(rS)

  rC = copy.deepcopy(countyPop)
  # print(rC)

  s = 0
  c = 0
  while c != len(rC) and s != len(rS):
    if rS.loc[s, 'GEOID'] == rC.loc[c, 'STATEFP']:
      rC.loc[c, dates] = rS.loc[s, dates]
      c += 1
    else:
      s += 1

  # print(rC)
  # exit()

  countyDataNew.loc[:, dates] = (countyDataNew.loc[:, dates] * countyPop.loc[:, dates] * rC.loc[:, dates]) + countyData.loc[:, dates].replace([NaN], 0)
  print(countyDataNew)

  t0 = fc.timer_restart(t0, 'make county data with suppressed estimates')

  del countyDataNew['STATEFP']
  fc.save_df(countyDataNew, cdcWonderDir, countyTitle + ', suppressed estimates', ext)

  t1 = fc.timer_restart(t1, 'total time')


