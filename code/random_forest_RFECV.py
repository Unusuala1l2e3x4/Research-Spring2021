from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from sklearn.feature_selection import RFECV
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import ast

# from time import time

import os, pathlib

import importlib
fc = importlib.import_module('functions')



def ravel_by_dates(unit, arr, dates):
  ret = pd.DataFrame()
  X, Y = np.meshgrid(arr, dates)
  ret[unit] = np.ravel(X)
  ret['YYYYMM'] = np.ravel(Y)
  return ret


def n_estimators_given_max_samples(X, n_estimators_base, max_samples_base, max_samples):
  if type(max_samples_base) is float:
    max_samples_base = int(np.ceil(max_samples_base*X.shape[0]))
  
  total_samples = max_samples_base*n_estimators_base

  if type(max_samples) is float:
    max_samples = int(np.ceil(max_samples*X.shape[0]))
  
  return int(np.ceil( total_samples / max_samples ))




def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum(params['max_samples']) # all floats
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # multiplier*3600*(folds*s*n*m)/prev_time    # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(1503.79746835443)    # 11440 sesc

  print(hrs, 'hrs, or')
  print(hrs*60, 'min, or')
  print(hrs*3600, 'sec')


def plotGiniImportance(importances):
  plt.bar(importances['name'], importances['importance'])
  plt.xticks(rotation=10)
  plt.xlabel("Dataset, Unit")
  plt.ylabel("Gini Importance")
  plt.yticks(list(np.arange(0,max(importances['importance'])+0.02,0.02)))
  plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
  return plt


def main():
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())

  outputDir = os.path.join(pPath, 'random_forest-outfiles')
  shapefilesDir = os.path.join(pPath, 'shapefiles')
  usaDir = os.path.join(shapefilesDir, 'USA_states_counties')
  cdcWonderDir = os.path.join(ppPath, 'CDC data', 'CDC WONDER datasets')
  usCensusDir = os.path.join(ppPath, 'US Census Bureau', 'population')
  nClimDivDir = os.path.join(ppPath, 'nClimDiv data')
  pmDir = os.path.join(ppPath, 'Atmospheric Composition Analysis Group')
  gfedCountyDir = os.path.join(ppPath, 'GFED4s_county')

  # PARAMS
  countyMapFile = 'cb_2019_us_county_500k'

  title = 'Underlying Cause of Death - Chronic lower respiratory diseases, 1999-2019'
  countyTitle = 'By county - ' + title
  stateTitle = 'By state - ' + title
  countySupEstTitle = countyTitle + ', suppressed estimates'

  ext = 'hdf5'

  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]

  # startYYYYMM, endYYYYMM = '200001', '201812'
  startYYYYMM, endYYYYMM = '200001', '201612'

  fileArgs = [('deaths', cdcWonderDir, countySupEstTitle), 
  ('popu', usCensusDir, 'TENA_county_pop_1999_2019'),
  ('precip_in', nClimDivDir, 'climdiv-pcpncy-v1.0'),
  ('temp_F', nClimDivDir, 'climdiv-tmpccy-v1.0'),
  ('pm25_ug_m-3', pmDir, 'TENA_county_PM25_200001_201812'),
  ('C_g_m-2', gfedCountyDir, 'TENA_C_200001_201812'),
  ('DM_kg_m-2', gfedCountyDir, 'TENA_DM_200001_201812'),
  ('BB_g_m-2', gfedCountyDir, 'TENA_BB_200001_201812'), # 2016 and before
  ('NPP_g_m-2', gfedCountyDir, 'TENA_NPP_200001_201812'), # 2016 and before
  ('Rh_g_m-2', gfedCountyDir, 'TENA_Rh_200001_201812'), # 2016 and before
  ('burned_frac', gfedCountyDir, 'TENA_burned_fraction_200001_201812'), # 2016 and before
  ('smallf_frac', gfedCountyDir, 'TENA_small_fire_fraction_200001_201812')] # 2016 and before


  # END PARAMS

  t0 = fc.timer_start()
  t1 = t0

  shapeData = gpd.read_file(os.path.join(usaDir, countyMapFile, countyMapFile + '.shp')).sort_values(by=['GEOID']).reset_index(drop=True)
  shapeData = fc.clean_states_reset_index(shapeData)
  shapeData = fc.county_changes_deaths_reset_index(shapeData)
  shapeData['ATOTAL'] = shapeData.ALAND + shapeData.AWATER
  # print(shapeData)

  t0 = fc.timer_restart(t0, 'read shapeData')

  dates, data = None, None

  for args in fileArgs:
    df = fc.read_df(args[1], args[2], ext)
    if dates is None:
      dates = sorted(i for i in df if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
    df = df.loc[:,['GEOID']+dates]
    if data is None:
      data = ravel_by_dates('GEOID', [fc.countyGEOIDstring(i) for i in df.GEOID], dates)
    data[args[0]] = np.ravel(df.loc[:,dates], order='F')

  data['deathRate'] = 100000 * data.deaths / data.popu

  data['ALAND_km2'] = ravel_by_dates('a', shapeData.ALAND, dates).a / 1000**2
  data['popuDensity_ALAND_km2'] = data.popu / data.ALAND_km2

  # data['AWATER_km2'] = ravel_by_dates('a', shapeData.AWATER, dates).a / 1000**2
  # data['AWATER_km2'] = data['AWATER_km2'].replace(NaN, 0)
  
  data['ATOTAL_km2'] = ravel_by_dates('a', shapeData.ATOTAL, dates).a / 1000**2
  data['popuDensity_ATOTAL_km2'] = data.popu / data.ATOTAL_km2

  data['ALAND_ATOTAL_ratio'] = data.ALAND_km2 / data.ATOTAL_km2

  data['month'] = [int(i[-2:]) for i in data.YYYYMM]


  # print(data.keys())
  # print(data)
  # exit()





  # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/




# 'ALAND_ATOTAL_ratio',
  columns = sorted(['popuDensity_ALAND_km2', 'precip_in', 'temp_F', 'pm25_ug_m-3',  'C_g_m-2', 'DM_kg_m-2','NPP_g_m-2','BB_g_m-2','Rh_g_m-2','burned_frac','smallf_frac'])

  print(columns)
  assert set(columns).issubset(data.keys())
  # exit()
  
  X = data[columns]
  y = data.deathRate
  # print(X)
  # print(y)

  t0 = fc.timer_restart(t0, 'data')

  refit = False
  n_splits = 2
  min_features_to_select=1


  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))z


  scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2']
  scoringParam = 'r2'
  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [50] } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
  # print('params\t', params)


  estimate_time( X.shape[1]-min_features_to_select, param_grid, DEFAULT_N_JOBS, cv_indices)


  # param_grid = {'max_samples': [0.1], 'n_estimators': [150,170], 'min_samples_leaf': [2,3], 'min_samples_split': [4]   }
  param_grid_list = ParameterGrid(param_grid)
  results = pd.DataFrame()

  t2 = fc.timer_restart(t0)
  for p in param_grid_list:
    # print(p)

    clf = RandomForestRegressor(random_state=0, n_jobs=DEFAULT_N_JOBS)
    clf.set_params(**p)

    rfe = RFECV(clf, step=1, min_features_to_select=min_features_to_select, cv=cv_indices, scoring=scoringParam, verbose=0)
    t0 = fc.timer_restart(t0)
    rfe.fit(X,y)
    t0 = fc.timer_restart(t0, 'rfe.fit time')

    for item in vars(rfe):
      print('\t','num',item,'\t', vars(rfe)[item])

    columns_important = [columns[feature_list_index] for feature_list_index in rfe.get_support(indices=True)]
    # print(columns_important)

    importances = pd.DataFrame(dict(name=columns_important, importance=rfe.estimator_.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
    print(importances)

    X_important = pd.DataFrame(X, columns=columns_important)


    # exit()
    clf = RandomForestRegressor(random_state=0)
    p2 = dict()
    for i in p.keys():
      p2[i] = [p[i]]
    gs = GridSearchCV(clf, param_grid=p2, refit=False, cv=cv_indices, scoring=scoring, verbose=2, n_jobs=DEFAULT_N_JOBS) # refit=scoringParam, 
    gs.fit(X_important, y)
    t0 = fc.timer_restart(t0, 'gs.fit time')
    results_ = pd.DataFrame.from_dict(gs.cv_results_)
    results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1) #  or i == 'p'
    results_['n_features'] = len(columns_important)
    results_['features'] = "['"+ "', '".join(columns_important)+"']"

    for i in range(len(columns)):
      results_[columns[i]+'_ranking'] = rfe.ranking_[i]
    
    maxRank = np.max(rfe.ranking_)
    for i in range(maxRank):
      results_[i+1] = rfe.grid_scores_[-maxRank+i]

    print(results_)


    results = pd.concat([results, results_])


  t2 = fc.timer_restart(t2, 'loop time')


  for s in scoring:
    results['rank_test_'+s] = fc.rank_decreasing(results['mean_test_'+s])

  print('results\t', results)

  saveTime = fc.utc_time_filename()
  fc.save_df(results, outputDir, 'GridSearchCV_RF_' + saveTime, 'csv')

  # # print(results.loc[results['rank_test_'+scoringParam]==1, :])
  bestparams = next(iter(dict(results.loc[results['rank_test_'+scoringParam]==1, ['params']]['params'] ).items()))[1]     # should be same as original params
  print('best_params_\t',bestparams)

  columns_important = ast.literal_eval(list(results.loc[results['rank_test_'+scoringParam]==1, ['features']]['features'])[0])
  print('features important\t',columns_important)

  if refit:
    print('refit')


    clf = RandomForestRegressor(random_state=0, n_jobs=DEFAULT_N_JOBS)
    clf.set_params(**bestparams)     # should be same as original params
    print(clf.get_params())
    # print('base_estimator\t',clf.base_estimator_)
    t0 = fc.timer_restart(t0)

    X_important = pd.DataFrame(X, columns=columns_important)
    clf.fit(X_important,y)

    t0 = fc.timer_restart(t0, 'refit time')

    # for item in vars(clf):
    #   if item == 'estimators_':
    #     print('\t','num',item,'\t', len(vars(clf)[item]))
    #   else:
    #     print('\t',item,'\t', vars(clf)[item])

    importances = pd.DataFrame(dict(name=columns_important, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
    print(importances)
    plot = plotGiniImportance(importances)

    fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
    fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')

    # plot.show()
    # plot.close()
  
  # for item in vars(gs):
  #   print('\t',item,'\t', vars(gs)[item])

  t1 = fc.timer_restart(t1, 'total time')



if __name__ == '__main__':
  main()