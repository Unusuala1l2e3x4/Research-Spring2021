from numpy.core.numeric import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from mlxtend.evaluate import bias_variance_decomp
from scipy import stats

import pandas as pd
import numpy as np

# from time import time

import os, pathlib, ast

import importlib
fc = importlib.import_module('functions')




def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum(params['max_samples']) # all floats
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # multiplier*3600*(folds*s*n*m)/prev_time    # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(2000.79746835443)    # 11440 sesc

  print(hrs, 'hrs, or')
  print(hrs*60, 'min, or')
  print(hrs*3600, 'sec')


def main():

  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8
  DEFAULT_RANDOM_STATE = 0
  DEFAULT_TRAIN_TEST_SPLIT_RANDOM_STATE = 2

  t0 = fc.timer_start()
  t1 = t0
  
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())

  outputDir = os.path.join(pPath, 'random_forest-outfiles')

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]

  # startYYYYMM, endYYYYMM = '200001', '201812'
  startYYYYMM, endYYYYMM = '200001', '201612'

  numMonthLags = 2




  data = fc.get_all_data(startYYYYMM, endYYYYMM, numMonthLags)
  data = fc.clean_data_before_train_test_split(data)

  t0 = fc.timer_restart(t0, 'load data')

  

  # print(list(data.keys()))
  # print(data)
  # exit()


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
  refit = True  # needed for test set score
  do_biasvariancedecomp = False
  save_best_refit = True
  n_splits = 10
  train_size = 0.7




  columns_list = [] 
  # columns_list.append(['popuDensity_ALAND_km2', 'month', 'temp_F', 'median_inc', 'year', 'PDSI', 'GEOID', 'Rh_g_m-2', 'SP01', 'months_from_start', 'precip_in', 'pm25_ug_m-3', 'NPP_g_m-2', 'ALAND_ATOTAL_ratio' ])
  # columns_list.append(['popuDensity_ALAND_km2', 'month', 'temp_F', 'median_inc', 'year', 'PDSI', 'GEOID', 'Rh_g_m-2', 'SP01', 'months_from_start', 'precip_in', 'pm25_ug_m-3'])
  # columns_list.append(['popuDensity_ALAND_km2', 'month', 'temp_F', 'median_inc', 'year', 'PDSI', 'GEOID', 'Rh_g_m-2', 'SP01', 'months_from_start', 'AQI'])
  
  # columns_list.append(fc.get_lags_from_columns(['GEOID',  'SP01', 'STATEFP', 'month', 'months_from_start', 'popuDensity_ALAND_km2', 'temp_F'], numMonthLags))
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'NPP_g_m-2_1m_lag', 'PDSI_2m_lag'])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'NPP_g_m-2_1m_lag', 'PDSI_2m_lag', 'pm25_ug_m-3_1m_lag', ])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'NPP_g_m-2_1m_lag', 'PDSI_2m_lag', 'temp_F' ])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'NPP_g_m-2_1m_lag', 'PDSI_2m_lag', 'temp_F', 'pm25_ug_m-3_1m_lag'])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'NPP_g_m-2_1m_lag', 'PDSI_2m_lag', 'temp_F', 'pm25_ug_m-3_1m_lag', 'median_inc'])

  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'PDSI_2m_lag'])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'PDSI_2m_lag', 'pm25_ug_m-3_1m_lag', ])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'PDSI_2m_lag', 'temp_F' ])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'PDSI_2m_lag', 'temp_F', 'pm25_ug_m-3_1m_lag'])
  columns_list.append(['GEOID', 'STATEFP', 'popuDensity_ALAND_km2', 'months_from_start', 'month', 'temp_F_1m_lag', 'temp_F_2m_lag', 'PDSI_2m_lag', 'temp_F', 'pm25_ug_m-3_1m_lag', 'median_inc'])
  






  # columns_list.append(fc.get_X_columns(numMonthLags))


  # 'precip_in', 'pm25_ug_m-3', 'NPP_g_m-2', 'ALAND_ATOTAL_ratio' ----> NEEDS TESTING WITH RFECV ----> try different column orderings


  scoring=['max_error','neg_mean_absolute_percentage_error', 'neg_mean_absolute_error','neg_mean_squared_error','explained_variance', 'r2']
  scoringParam = 'r2'
  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [140] } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
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
  results = pd.DataFrame()
  importances_list = []

  t3 = fc.timer_start()
  # for p in param_grid_list:
  #   print(p)


  if crossvalidate: 
    for columns in columns_list:
      # assert set(columns).issubset(fc.get_X_columns(numMonthLags))
      print('included:\t',columns)
      # print('not included:\t',set(fc.get_X_columns(numMonthLags)) - set(columns))
      # print('not included (from all of data):\t',set(data.keys()) - set(columns_list[0]))
    
      t2 = fc.timer_start()
      # continue

      X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=train_size, random_state=DEFAULT_TRAIN_TEST_SPLIT_RANDOM_STATE)
      print(X)
      # print(y)
      # exit()
      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
      gs = GridSearchCV(clf, param_grid=param_grid, refit=False, cv=cv_indices, scoring=scoring, verbose=1, n_jobs=DEFAULT_N_JOBS)
      t0 = fc.timer_start()
      gs.fit(X, y)
      t0 = fc.timer_restart(t0, 'fit GridSearchCV')
      results_ = pd.DataFrame.from_dict(gs.cv_results_)
      results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1) #  or i == 'params'
      if refit:
        results_['test_set_r2'] = NaN
        results_['rank_test_set_r2'] = NaN

      for i in results_.index:
        if refit:
          t0 = fc.timer_start()

          params_i = dict(results_.loc[i,'params'])
          
          clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
          clf.set_params(**params_i)
          clf.fit(X,y)

          importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
          importances_list.append(importances)
          # print(importances)
        
          results_.loc[i,'test_set_r2'] = clf.score(X_test, y_test)

          t0 = fc.timer_restart(t0, 'refit params for this column set')

        results_.loc[i,'n_features'] = len(columns)
        results_.loc[i,'features'] = "['"+ "', '".join(columns)+"']"

      # print(results_)
      results = pd.concat([results, results_], ignore_index=True)
      t2 = fc.timer_restart(t2, 'time for this set of columns')



    t3 = fc.timer_restart(t3, 'loop time')
    

    for s in scoring:
      results['rank_test_'+s] = fc.rank_decreasing(results['mean_test_'+s])
    if refit:
      results['rank_test_set_r2'] = fc.rank_decreasing(results['test_set_r2'])


    saveTime = fc.utc_time_filename()
    if save_best_refit:
      # get index
      i = int(results.loc[results['rank_test_'+scoringParam] == 1,:].index[0])

      print(i)
      # print(ast.literal_eval(results_.loc[i,'features']))

      importances = importances_list[i]
      print(importances)
      plot = fc.plotGiniImportance(importances)
      fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
      fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')
    
    

    fc.save_df(results, outputDir, 'GridSearchCV_RF_' + saveTime, 'csv')
    







  else:
    # print('crossvalidate', crossvalidate)
  
    for columns in columns_list:
      for p in param_grid_list:
        clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
        clf.set_params(**p)
        
        X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=train_size, random_state=DEFAULT_TRAIN_TEST_SPLIT_RANDOM_STATE)

        t0 = fc.timer_start()
        clf.fit(X,y)
        t0 = fc.timer_restart(t0, 'fit time')
        
        # for item in vars(clf):
        #   if item == 'estimators_':
        #     print('\t','num',item,'\t', len(vars(clf)[item]))
        #   else:
        #     print('\t',item,'\t', vars(clf)[item])

        importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
        print(importances)
        plot = fc.plotGiniImportance(importances)

        saveTime = fc.utc_time_filename()
        fc.save_plt(plot, outputDir, 'GiniImportance_RF_' + saveTime, 'png')
        fc.save_df(importances, outputDir, 'GiniImportance_RF_' + saveTime, 'csv')

        if do_biasvariancedecomp:
          assert not crossvalidate or refit
          t0 = fc.timer_start()
          avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf, X, y, X_test, y_test, loss='mse', random_seed=DEFAULT_RANDOM_STATE)
          t0 = fc.timer_restart(t0, 'bias_variance_decomp')

          print('Average expected loss: %f' % avg_expected_loss)
          print('Average bias: %f' % avg_bias)
          print('Average variance: %f' % avg_var)

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