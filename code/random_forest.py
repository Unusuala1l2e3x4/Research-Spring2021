from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from mlxtend.evaluate import bias_variance_decomp

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
  usCensusDir = os.path.join(ppPath, 'US Census Bureau')
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
  DEFAULT_RANDOM_STATE = 0

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]

  # startYYYYMM, endYYYYMM = '200001', '201812'
  startYYYYMM, endYYYYMM = '200001', '201612'

  fileArgs = [('deaths', os.path.join(cdcWonderDir, 'Chronic lower respiratory diseases'), countySupEstTitle), 
  ('popu', os.path.join(usCensusDir, 'population'), 'TENA_county_pop_1999_2019'),
  ('median_inc', os.path.join(usCensusDir, 'SAIPE State and County Estimates'), 'TENA_county_median_income_2000_2019'),
  ('precip_in', nClimDivDir, 'climdiv-pcpncy-v1.0'),
  ('temp_F', nClimDivDir, 'climdiv-tmpccy-v1.0'),
  ('PDSI', nClimDivDir, 'climdiv-pdsidv-v1.0'),
  ('SP01', nClimDivDir, 'climdiv-sp01dv-v1.0'),
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

  t0 = fc.timer_restart(t0, 'load data')

  # print(list(data.keys()))
  print(data)
  exit()


  # https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees

  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
  # https://stackoverflow.com/questions/38151615/specific-cross-validation-with-random-forest

  # https://scikit-learn.org/stable/modules/tree.html#tree
  # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.get_depth


  # https://stats.stackexchange.com/questions/18856/is-cross-validation-a-proper-substitute-for-validation-set


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


  # Bias-variance
  #   https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/
  #   http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/#example-2-bias-variance-decomposition-of-a-decision-tree-regressor


  # PARAMS
  crossvalidate = True
  refit = True
  do_biasvariancedecomp = False
  n_splits = 10


  columns_list = [] 
  # columns_list.append(['GEOID','month','temp_F', 'burned_frac', 'popuDensity_ALAND_km2', 'precip_in', 'Rh_g_m-2', 'pm25_ug_m-3', 'NPP_g_m-2', 'C_g_m-2', 'smallf_frac', 'DM_kg_m-2', 'BB_g_m-2'])
  # columns_list.append(['GEOID','month','temp_F', 'burned_frac', 'popuDensity_ALAND_km2', 'precip_in', 'Rh_g_m-2', 'pm25_ug_m-3', 'NPP_g_m-2', 'C_g_m-2', 'smallf_frac', 'DM_kg_m-2', 'ALAND_ATOTAL_ratio'])
  columns_list.append(['GEOID','month','temp_F', 'burned_frac', 'popuDensity_ALAND_km2', 'precip_in', 'Rh_g_m-2', 'pm25_ug_m-3', 'NPP_g_m-2', 'C_g_m-2', 'smallf_frac', 'DM_kg_m-2', 'BB_g_m-2', 'ALAND_ATOTAL_ratio'])

  scoring=['neg_mean_absolute_error','neg_mean_squared_error','explained_variance','r2']
  scoringParam = 'r2'



  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [170] } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
  # print(param_grid)
  # 70,90,110,130,150,170
  # param_grid = {'max_samples': [0.1,0.2,0.3], 'n_estimators': [60,80,100,120,140,160,180], 'min_samples_leaf': [1,2,3,4]} # , 'max_depth':[None]

  param_grid_list = ParameterGrid(param_grid)



  # END PARAMS

  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))

  estimate_time(len(columns_list), param_grid, DEFAULT_N_JOBS, cv_indices)
  # exit()

  t3 = fc.timer_start()
  for p in param_grid_list:
    print(p)
    for columns in columns_list:
      assert set(columns).issubset(data.keys())
      print('included:\t',columns)
      print('not included:\t',set(data.keys()) - set(columns_list[0]))
      t2 = fc.timer_start()
      # continue

      X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=0.7, random_state=2)
      # print(X)
      # print(y)

      if crossvalidate: 
        clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)

        p2 = dict()
        for i in p.keys():
          p2[i] = [p[i]]
        gs = GridSearchCV(clf, param_grid=p2, refit=False, cv=cv_indices, scoring=scoring, verbose=2, n_jobs=DEFAULT_N_JOBS) # refit=scoringParam, 
        t0 = fc.timer_start()
        gs.fit(X, y)
        t0 = fc.timer_restart(t0, 'fit GridSearchCV')
        results = pd.DataFrame.from_dict(gs.cv_results_)
        results = results.drop([i for i in results.keys() if i.startswith('split')], axis=1) #  or i == 'params'
        results['n_features'] = len(columns)
        results['features'] = "['"+ "', '".join(columns)+"']"

        print(results)


        saveTime = fc.utc_time_filename()
        fc.save_df(results, outputDir, 'GridSearchCV_RF_' + saveTime, 'csv')

        # print(results.loc[results['rank_test_'+scoringParam]==1, :])
        bestparams = next(iter(dict(results.loc[results['rank_test_'+scoringParam]==1, ['params']]['params'] ).items()))[1]
        print('best_params_\t',bestparams)

        if refit:
          print('refit')
          bestparams['n_jobs'] = DEFAULT_N_JOBS
          clf.set_params(**bestparams)
          # print('base_estimator\t',clf.base_estimator_)
          t0 = fc.timer_start()
          clf.fit(X,y)
          t0 = fc.timer_restart(t0, 'refit time')

          importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
          print(importances)
          plot = plotGiniImportance(importances)

          fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
          fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')
          # plot.show()
          # plot.close()
        

      else:
        # print('crossvalidate', crossvalidate)
        clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
        clf.set_params(**p)
        
        t0 = fc.timer_start()
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

        # plot.show()
        # plot.close()





      if do_biasvariancedecomp:
        t0 = fc.timer_start()
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf, X, y, X_test, y_test, loss='mse', random_seed=DEFAULT_RANDOM_STATE)
        t0 = fc.timer_restart(t0, 'bias_variance_decomp')

        print('Average expected loss: %f' % avg_expected_loss)
        print('Average bias: %f' % avg_bias)
        print('Average variance: %f' % avg_var)

      t2 = fc.timer_restart(t2, 'time for this columns permutation')
  

  t3 = fc.timer_restart(t3, 'loop time')
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
  #     clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, max_samples=100)
  #     # define the evaluation procedure
  #     cv_indices = KFold(n_splits=n_splits)
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