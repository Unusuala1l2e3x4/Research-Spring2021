
import numpy as np

import geopandas as gpd
import pandas as pd

import os, pathlib

import io

import importlib
fc = importlib.import_module('functions')




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

  testing = False
  doChanges = True
  origDataMonth = '07'
  suppValString = '-1'
  ext = 'hdf5'
  # END PARAMS

  all_dfs = []

  t0 = fc.timer_start()
  t1 = t0

  countyPopName = 'TENA_county_pop_1999_2019'
  statePopName = 'TENA_state_pop_1999_2019'
  

  # for testing
  data_GEOIDs = []
  all_filenames = []
  
  # 1998 - 1999 data files - (write 1999 monthly pop)
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('stch-icen') and name.endswith('.txt')]
  for filename in filenames:
    print(filename)
    outFileName = filename.split('.')[0] + '_totals'
    # if outFileName + '.csv' in os.listdir(usCensusDir): 
    if fc.is_in_dir(usCensusDir, outFileName, ext):
      # df = pd.read_csv(os.path.join(usCensusDir, outFileName + '.csv'))
      df = fc.read_df(usCensusDir, outFileName, ext)
      # df['GEOID'] = [countyGEOIDstring(item) for item in df['GEOID']]
    else:
      yyyymm = filename[-8:-4]+origDataMonth
      df = open(os.path.join(usCensusDir, filename), 'r').read()
      df = pd.read_csv(io.StringIO(df), names=['GEOID','AgeGroup','RaceSex','Ethnicity', yyyymm], sep='\s+')
      df['GEOID'] = [fc.countyGEOIDstring(item) for item in df['GEOID']]
      df = fc.clean_states_reset_index(df)
      geoids = sorted(list(set(df['GEOID'])))
      sums = []
      for geoid in geoids:
        temp = df[df['GEOID'] == geoid]
        sums.append(np.sum(temp[yyyymm]))
      df = pd.DataFrame()
      df['GEOID'] = geoids
      df[yyyymm] = sums
      fc.save_df(df, usCensusDir, outFileName, ext)
    # print(df)

    if doChanges:
      df = fc.county_changes_deaths_reset_index(df)

    all_dfs.append(df)
    
    if testing:
      data_GEOIDs.append(set(df['GEOID'])) #
      all_filenames.append(filename) #

  t0 = fc.timer_restart(t0, '1998 - 1999 data files')

  # 2000 - 2019 data files
  filenames = [name for name in os.listdir(usCensusDir) if name.startswith('co-est') and name.endswith('.csv')]
  for filename in filenames:
    print(filename)
    df = pd.read_csv(os.path.join(usCensusDir, filename))
    yearColumns = [item for item in list(df.keys()) if item.startswith('POPESTIMATE')][:10]
    yearColumns2 = []
    # print(yearColumns)
    df['GEOID'] = [fc.GEOID_string_state_county(state,county) for state,county in zip(df['STATE'], df['COUNTY'])]
    df = df[~df['GEOID'].str.endswith('000') ]
    df = fc.clean_states_reset_index(df)
    for c in yearColumns:
      df[c[-4:]+origDataMonth] = df[c]
      yearColumns2.append(c[-4:]+origDataMonth)
    df = df[['GEOID']+yearColumns2]
    # print(df)

    if doChanges:
      df = fc.county_changes_deaths_reset_index(df)

    all_dfs.append(df)

    if testing:
      data_GEOIDs.append(set(df['GEOID'])) #
      all_filenames.append(filename) #

  t0 = fc.timer_restart(t0, '2000 - 2019 data files')


  if testing:
    countyMapFiles = ['cb_2019_us_county_500k', 'co99_d00']
    countyMapData = dict()

    for filename in countyMapFiles:
      df = gpd.read_file(os.path.join(usaDir, filename, filename + '.shp'))
      if 'GEOID' not in df.keys():
        df['GEOID'] = [fc.GEOID_string_state_county(state,county) for state,county in zip(df['STATE'], df['COUNTY'])]
      df = df.sort_values(by=['GEOID'])
      countyMapData[filename] = fc.clean_states_reset_index(df)
      data_GEOIDs.append(set(countyMapData[filename].GEOID)) #
      all_filenames.append(filename) #

    t0 = fc.timer_restart(t0, 'map files')

    deaths_GEOIDs = dict()
    countyData = fc.deaths_by_date_geoid(os.path.join(cdcWonderDir, countyTitle), suppValString) # has missing rows
    countyData = fc.clean_states_reset_index(countyData)
    # print(countyData)
    dates = set(countyData['YYYYMM'])
    # print(dates, len(dates))
    for date in sorted(list(dates)):
      deaths_GEOIDs[date] = set(countyData[countyData['YYYYMM'] == date]['GEOID'])

    t0 = fc.timer_restart(t0, 'disease data files')

    data_GEOIDs_union = set().union(*[i for i in data_GEOIDs])
    print(len(data_GEOIDs_union), '\nin data_GEOIDs_union')
    for i in range(len(data_GEOIDs)):
      print('not in', all_filenames[i], '\t', len(data_GEOIDs[i]), sorted(list(data_GEOIDs_union - data_GEOIDs[i])))

    t0 = fc.timer_restart(t0, 'show differences')

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
    
    t0 = fc.timer_restart(t0, 'show issubsets')


  if doChanges and testing:
    all_GEOIDs = [list(df['GEOID']) for df in all_dfs]
    all_GEOIDs_union = sorted(set().union(*[set(i) for i in all_GEOIDs]))
    for i in range(len(all_GEOIDs)): # see if all geoids equal + sorted across all files (by comparing to union)
      print(all_GEOIDs[i] == all_GEOIDs_union, len(all_GEOIDs[i]), len(all_GEOIDs_union))

    t0 = fc.timer_restart(t0, 'show if GEOIDs equal across all files')


  # orig pop numbers are for July 1; write other months based on avg monthly net change

  countyPop = pd.DataFrame(all_dfs[0])
  for i in range(1, len(all_dfs)):
    dates = [i for i in all_dfs[i] if i != 'GEOID' and i != 'STATEFP']
    countyPop = pd.concat([countyPop, all_dfs[i][dates]], axis=1)

  dates = sorted(i for i in countyPop if i != 'GEOID' and i != 'STATEFP')
  monthSeq = np.roll([fc.month_str(month) for month in range(1,13)], -int(origDataMonth))

  for i in range(len(dates[:-1])):
    year = dates[i][:4]
    prev = year + monthSeq[-1]

    monthlyChange = pd.DataFrame(countyPop[dates[i + 1]] - countyPop[dates[i]]) / 12

    startyyyymm = monthSeq[-1]
    for month in monthSeq[:-1]:
      if month == '01':
        year = dates[i + 1][:4]
      yyyymm = year + month
      countyPop[yyyymm] = countyPop[prev] + monthlyChange[0]
      prev = yyyymm

  countyPop = countyPop.sort_index(axis=1)
  countyPop = countyPop[ ['GEOID'] + [i for i in countyPop.columns if i != 'GEOID' and i != 'STATEFP'] ]

  countyPop.loc[:, dates] = countyPop.loc[:, dates].astype(float)
  fc.save_df(countyPop, usCensusDir, countyPopName, ext)
  t0 = fc.timer_restart(t0, 'write county data')


  statefps = [item[0:2] for item in countyPop.GEOID]

  statePop = pd.DataFrame(None, columns=list(countyPop.keys()))
  statePop.GEOID = [item[0:2] for item in sorted(set(statefps))]
  countyPop['STATEFP'] = statefps

  dates = sorted(i for i in countyPop if i != 'GEOID' and i != 'STATEFP')
  # print(dates)

  for date in dates:
    statePop.loc[:, date] = np.ravel([np.sum(countyPop.loc[countyPop.STATEFP == statefp, date]) for statefp in statePop.GEOID])

  fc.save_df(statePop, usCensusDir, statePopName, ext)
  t0 = fc.timer_restart(t0, 'write state data')


  t1 = fc.timer_restart(t1, 'total time')


