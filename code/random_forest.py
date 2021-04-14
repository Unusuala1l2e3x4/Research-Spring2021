from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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




def estimate_time(params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum(params['max_samples']) # all floats
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # 3600*(folds*s*n*m)/prev_time    # 3600*(10*0.1*170*1)/prev_time
  hrs =  (5/n_jobs)*(folds*s*n*m)/(1206.5717412)    # 11440 sesc

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


  # https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees

  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
  # https://stackoverflow.com/questions/38151615/specific-cross-validation-with-random-forest

  # https://scikit-learn.org/stable/modules/tree.html#tree
  # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.get_depth

  # param tuning advice
  #   https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

  # CV
  #    https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
  #    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
  
  # regularization with cost complexity pruning
  #   https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
  #   https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

  # scoring
  #   https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
  #   https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
  #   https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination
  #   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split


# 'ALAND_ATOTAL_ratio', 
  columns = [i for i in data if i in ['popuDensity_ALAND_km2', 'precip_in', 'temp_F', 'pm25_ug_m-3','C_g_m-2', 'DM_kg_m-2','NPP_g_m-2','BB_g_m-2','Rh_g_m-2','burned_frac','smallf_frac']]
  # columns = [i for i in data if i not in ['deathRate','GEOID','YYYYMM','deaths','popu','ALAND_km2', 'ATOTAL_km2', 'popuDensity_ATOTAL_km2']] # ,'AWATER_km2','ALAND_km2', 'ATOTAL_km2', 'popuDensity_ATOTAL_km2'
  # print(columns)
  
  X = data[columns]
  y = data.deathRate
  # print(X)
  # print(y)


  # print(list(np.arange(0, 2e-6, 1e-7)))
  # exit()


  t0 = fc.timer_restart(t0, 'data')

  crossvalidate = False
  refit = True

  if crossvalidate: 
    print('crossvalidate', crossvalidate)
    
    cv_indices = KFold(n_splits=10, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
    # for train_indices, test_indices in cv_indices.split(data):
    #   print('Train: %s | test: %s' % (train_indices, test_indices))

    scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2']
    scoringParam = 'r2'

    sampleRatio = 0.1
    # print(sampleRatio*X.shape[0])


    clf = RandomForestRegressor(random_state=0)



    # max_samples = [0.1,0.2,0.3]

    # print([n_estimators_given_max_samples(X, 150, 0.1, i) for i in max_samples ]) # [150, 76, 51]

    # param_grid = [{'max_samples': [0.1], 'n_estimators': [150], 'min_samples_leaf': [2,3]},
    #               {'max_samples': [0.2], 'n_estimators': [76], 'min_samples_leaf': [2,3]},
    #               {'max_samples': [0.3], 'n_estimators': [51], 'min_samples_leaf': [2,3]}]

    # test = [2e-07, 3e-07, 5e-07, 6e-07, 7e-07, 9e-07, 1.1e-06, 1.3e-06, 1.4e-06, 1.5e-06, 1.7e-06, 1.8e-06, 1.9e-06]
    # test = set(np.arange(0,2e-06, 2e-8))

    # test = set(np.arange(0,2e-07, 4e-8))
    # print(test, len(test))


    # exit()

    # 70,90,110,130,150,170
    param_grid = {'max_samples': [0.1], 'n_estimators': [170], 'min_samples_leaf': [2], 'min_samples_split': [4]   } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))

    print(param_grid)

    estimate_time(param_grid, DEFAULT_N_JOBS, cv_indices)
    # exit()

    

    # param_grid = {'max_samples': [0.1,0.2,0.3], 'n_estimators': [60,80,100,120,140,160,180], 'min_samples_leaf': [1,2,3,4]} # , 'max_depth':[None]


    gs = GridSearchCV(clf, param_grid=param_grid, refit=False, cv=cv_indices, scoring=scoring, verbose=2, n_jobs=DEFAULT_N_JOBS) # refit=scoringParam, 
    gs.fit(X, y)
    t0 = fc.timer_restart(t0, 'fit GridSearchCV')
    results = pd.DataFrame.from_dict(gs.cv_results_)
    results = results.drop([i for i in results.keys() if i.startswith('split')], axis=1) #  or i == 'params'
    print(results)
    saveTime = fc.utc_time_filename()
    fc.save_df(results, outputDir, 'GridSearchCV_RF_' + saveTime, 'csv')

    # print(results.loc[results['rank_test_'+scoringParam]==1, :])
    bestparams = next(iter(dict(results.loc[results['rank_test_'+scoringParam]==1, ['params']]['params'] ).items()))[1]
    print('best_params_\t',bestparams)

    if refit:
      bestparams['n_jobs'] = DEFAULT_N_JOBS
      clf.set_params(**bestparams)
      # print('base_estimator\t',clf.base_estimator_)
      t0 = fc.timer_restart(t0)
      clf.fit(X,y)
      t0 = fc.timer_restart(t0, 'refit time')

      for item in vars(clf):
        if item == 'estimators_':
          print('\t','num',item,'\t', len(vars(clf)[item]))
        else:
          print('\t',item,'\t', vars(clf)[item])

      importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
      print(importances)
      plot = plotGiniImportance(importances)

      fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
      fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')

      # plot.show()
      # plot.close()
    
    # for item in vars(gs):
    #   print('\t',item,'\t', vars(gs)[item])

  else:
    print('crossvalidate', crossvalidate)
    clf = RandomForestRegressor(random_state=0, n_jobs=DEFAULT_N_JOBS)

    # bestparams =  {'max_samples': 0.1, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 50}  
    bestparams =  { 'max_samples': 0.1, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 10 }                 ############ copy of saved dict

    clf.set_params(**bestparams)
    
    # print('base_estimator\t',clf.base_estimator_)
    t0 = fc.timer_restart(t0)
    clf.fit(X,y)
    t0 = fc.timer_restart(t0, 'fit time')
    
    for item in vars(clf):
      if item == 'estimators_':
        print('\t','num',item,'\t', len(vars(clf)[item]))
      else:
        print('\t',item,'\t', vars(clf)[item])

    importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
    print(importances)
    plot = plotGiniImportance(importances)

    saveTime = fc.utc_time_filename()
    fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
    fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')

    plot.show()
    # plot.close()

  t1 = fc.timer_restart(t1, 'total time')



if __name__ == '__main__':
  main()






  # test_cores = False
  # if test_cores:
  #   # https://machinelearningmastery.com/multi-core-machine-learning-in-python/
  #   print('test_cores', test_cores)

  #   results = list()

  #   # compare timing for number of cores
  #   n_cores = [1, 2, 3, 4, 5, 6, 7, 8]
  #   for n in n_cores:
  #     # define the model
  #     clf = RandomForestRegressor(random_state=0, max_samples=100)
  #     # define the evaluation procedure
  #     cv_indices = KFold(n_splits=10)
  #     # record the current time
  #     start = time()
  #     # evaluate the model
  #     n_scores = cross_validate(clf, X, y,  cv=cv_indices, n_jobs=n) # 
  #     # print(np.mean(n_scores))
  #     # record the current time
  #     end = time()
  #     # store execution time
  #     result = end - start
  #     print('>cores=%d: %.3f seconds' % (n, result))
  #     results.append(result)
  #   plt.plot(n_cores, results)
  #   plt.show()

  # elif crossvalidate: 