from numpy.core.numeric import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
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
  # print(list(data.keys()))
  # print(data)
  # exit()

  # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/



  # PARAMS

  

  # print(data)
  columns = fc.get_all_X_columns()
  print(sorted(columns))
  print('not included:\t',set(data.keys()) - set(columns))
  assert set(columns).issubset(data.keys())



  refit = True
  do_biasvariancedecomp = False
  n_splits = 10
  min_features_to_select = 8
  train_size = 0.7


  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))


  scoring=['neg_mean_absolute_error','neg_mean_squared_error','explained_variance','r2']
  scoringParam = 'r2'
  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [120] } # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
  # print('params\t', params)
  
  numberShuffles = 13 # max: 18 choose 15 = 816

  # END PARAMS



  X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=train_size, random_state=2)
  # print(X)
  
  # print(y)
  # exit()

  estimate_time( numberShuffles * (X.shape[1]-min_features_to_select), param_grid, DEFAULT_N_JOBS, cv_indices)
  # exit()

  param_grid_list = ParameterGrid(param_grid)
  results = pd.DataFrame()

  columns_list = []
  columns_list_sets = []
  random.seed(678)
  random.shuffle(columns)
  while len(columns_list) != numberShuffles:
    i = 0
    while set(columns[:15]) in columns_list_sets or columns[:15] in columns_list:  # RFECV can only handle 15 without weird behavior
      random.shuffle(columns)
      i += 1
      if i == 10000:
        print('intervene loop')
        break
    if 'GEOID' not in copy.deepcopy(columns[:15]):
      random.shuffle(columns)
      continue
    columns_list.append(copy.deepcopy(columns[:15]))
    columns_list_sets.append(set(copy.deepcopy(columns[:15])))
    # print(columns)

  for i in columns_list:
    print('\t',i)
  print()
  # for i in columns_list_sets:
  #   print(len(i), i)
  # exit()


  t2 = fc.timer_start()
  for p in param_grid_list:
    print(p)
    for columns_i in columns_list: # test 1 
      print('\t',columns_i)

      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
      clf.set_params(**p)
      rfe = RFECV(clf, step=1, min_features_to_select=min_features_to_select, cv=cv_indices, scoring=scoringParam, verbose=0)
      t0 = fc.timer_start()
      # print(X[columns_i])
      # continue
      rfe.fit(X[columns_i],y)
      t0 = fc.timer_restart(t0, 'rfe.fit time')

      # print(rfe.ranking_)
      # print(rfe.grid_scores_)

      columns_important = [columns_i[feature_list_index] for feature_list_index in rfe.get_support(indices=True)]
      # print(columns_important)

      importances = pd.DataFrame(dict(name=columns_important, importance=rfe.estimator_.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
      print(importances)


      # exit()
      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
      p2 = dict()
      for i in p.keys():
        p2[i] = [p[i]]
      gs = GridSearchCV(clf, param_grid=p2, refit=False, cv=cv_indices, scoring=scoring, verbose=1, n_jobs=DEFAULT_N_JOBS)
      # print(X[columns_important])
      t0 = fc.timer_start()
      gs.fit(X[columns_important], y)
      t0 = fc.timer_restart(t0, 'gs.fit time')
      results_ = pd.DataFrame.from_dict(gs.cv_results_)
      results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1)

      t0 = fc.timer_start()
      results_['test_set_r2_score'] = rfe.estimator_.score(X_test[columns_important], y_test)
      t0 = fc.timer_restart(t0, 'test set scoring time')


      results_['n_features'] = len(columns_important)
      results_['features'] = "['"+ "', '".join(columns_important)+"']"
      if len(results) == 0:
        for c in sorted(columns):
          results_[c+'_ranking'] = NaN
      for i in range(len(columns_i)):
        results_[columns_i[i]+'_ranking'] = rfe.ranking_[i]
      maxRank = np.max(rfe.ranking_)
      for i in range(maxRank):
        results_[i+1] = rfe.grid_scores_[-maxRank+i]
      results = pd.concat([results, results_], ignore_index=True)


  t2 = fc.timer_restart(t2, 'loop time')




  for s in scoring:
    results['rank_test_'+s] = fc.rank_decreasing(results['mean_test_'+s])
  # print('results\t', results)

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
    # print(X[columns_important])
    t0 = fc.timer_start()
    clf.fit(X[columns_important],y)
    t0 = fc.timer_restart(t0, 'refit time')

    importances = pd.DataFrame(dict(name=columns_important, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
    print(importances)
    plot = plotGiniImportance(importances)

    fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
    fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')
    # plot.show()
    # plot.close()
  



  if do_biasvariancedecomp:
    assert refit
    t0 = fc.timer_start()
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf, X[columns_important], y, X_test[columns_important], y_test, loss='mse', random_seed=DEFAULT_RANDOM_STATE)
    t0 = fc.timer_restart(t0, 'bias_variance_decomp')

    print('Average expected loss: %f' % avg_expected_loss)
    print('Average bias: %f' % avg_bias)
    print('Average variance: %f' % avg_var)






  t1 = fc.timer_restart(t1, 'total time')


if __name__ == '__main__':
  main()