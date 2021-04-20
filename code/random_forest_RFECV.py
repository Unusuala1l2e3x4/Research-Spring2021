from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.feature_selection import RFECV
from mlxtend.evaluate import bias_variance_decomp

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import ast, copy

# from time import time

import os, pathlib, random

import importlib
fc = importlib.import_module('functions')





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

  # PARAMS

  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8
  DEFAULT_RANDOM_STATE = 0

  startYYYYMM, endYYYYMM = '200001', '201612'

  # END PARAMS

  t0 = fc.timer_start()
  t1 = t0


  data = fc.get_all_data(startYYYYMM, endYYYYMM)
  t0 = fc.timer_restart(t0, 'load data')
  print(data.keys())
  print(data)
  # exit()

  # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/





  # PARAMS

  # 'ALAND_ATOTAL_ratio',
  columns = fc.get_all_X_columns()
  # columns = sorted(columns)
  print(columns)
  assert set(columns).issubset(data.keys())
  # exit()
  



  refit = False
  do_biasvariancedecomp = False
  n_splits = 10
  min_features_to_select=9


  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))z


  scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2']
  scoringParam = 'r2'
  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [100] } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
  # print('params\t', params)

  numberShuffles = 6

  # END PARAMS



  X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=0.7, random_state=2)
  # print(X)
  # print(y)

  estimate_time( numberShuffles * (X.shape[1]-min_features_to_select), param_grid, DEFAULT_N_JOBS, cv_indices)

  param_grid_list = ParameterGrid(param_grid)
  results = pd.DataFrame()

  # importances_list = []
  columns_list = []

  random.seed(10)
  while len(columns_list) != numberShuffles:
    while columns in columns_list:
      random.shuffle(columns)
    columns_list.append(copy.deepcopy(columns))
    # print(columns)

  # exit()
  

  t2 = fc.timer_start()
  for p in param_grid_list:
    print(p)
    for columns_i in columns_list:
      print(columns_i)

      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
      clf.set_params(**p)
      rfe = RFECV(clf, step=1, min_features_to_select=min_features_to_select, cv=cv_indices, scoring=scoringParam, verbose=0)
      t0 = fc.timer_start()
      rfe.fit(X,y)
      t0 = fc.timer_restart(t0, 'rfe.fit time')

      print(rfe.ranking_)
      print(rfe.grid_scores_)

      columns_important = [columns_i[feature_list_index] for feature_list_index in rfe.get_support(indices=True)]
      # print(columns_important)

      importances = pd.DataFrame(dict(name=columns_important, importance=rfe.estimator_.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
      print(importances)
      # importances_list.append(importances)


      # exit()
      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
      p2 = dict()
      for i in p.keys():
        p2[i] = [p[i]]
      gs = GridSearchCV(clf, param_grid=p2, refit=False, cv=cv_indices, scoring=scoring, verbose=1, n_jobs=DEFAULT_N_JOBS) # refit=scoringParam, 
      X_important = pd.DataFrame(X, columns=columns_important)
      t0 = fc.timer_start()
      gs.fit(X_important, y)
      t0 = fc.timer_restart(t0, 'gs.fit time')

      results_ = pd.DataFrame.from_dict(gs.cv_results_)
      results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1)
      results_['n_features'] = len(columns_important)
      results_['features'] = "['"+ "', '".join(columns_important)+"']"

      for i in range(len(columns_i)):
        results_[columns_i[i]+'_ranking'] = rfe.ranking_[i]
      
      maxRank = np.max(rfe.ranking_)
      for i in range(maxRank):
        results_[i+1] = rfe.grid_scores_[-maxRank+i]

      # print(results_)
      results = pd.concat([results, results_], ignore_index=True)

  t2 = fc.timer_restart(t2, 'loop time')

  # exit()
  
  # assert len(param_grid_list) == len(importances_list)
  # for i in range(len(param_grid_list)):
  #   print(param_grid_list[i])
  #   print(importances_list[i])


  # assert numberShuffles == len(columns_list)
  # for i in range(numberShuffles):
  #   print(columns_list[i])
  #   print(importances_list[i])
  # exit()

  for s in scoring:
    results['rank_test_'+s] = fc.rank_decreasing(results['mean_test_'+s])

  print('results\t', results)

  saveTime = fc.utc_time_filename()
  fc.save_df(results, outputDir, 'RFECV_RF_' + saveTime, 'csv')

  # # print(results.loc[results['rank_test_'+scoringParam]==1, :])
  bestparams = next(iter(dict(results.loc[results['rank_test_'+scoringParam]==1, ['params']]['params'] ).items()))[1]     # should be same as original params
  print('best_params_\t',bestparams)

  columns_important = ast.literal_eval(list(results.loc[results['rank_test_'+scoringParam]==1, ['features']]['features'])[0])
  print('features important\t',columns_important)

  if refit:
    print('refit')
    clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
    clf.set_params(**bestparams)     # should be same as original params
    print(clf.get_params())
    # print('base_estimator\t',clf.base_estimator_)
    

    X_important = pd.DataFrame(X, columns=columns_important)
    t0 = fc.timer_start()
    clf.fit(X_important,y)
    t0 = fc.timer_restart(t0, 'refit time')

    importances = pd.DataFrame(dict(name=columns_important, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
    print(importances)
    plot = plotGiniImportance(importances)

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






  t1 = fc.timer_restart(t1, 'total time')


if __name__ == '__main__':
  main()