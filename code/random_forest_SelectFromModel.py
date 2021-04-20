from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import ast

# from time import time

import os, pathlib

import importlib
fc = importlib.import_module('functions')







def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum(params['max_samples']) # all floats
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # 3600*(folds*s*n*m)/prev_time    # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(1206.5717412)    # 11440 sesc

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


  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]

  # startYYYYMM, endYYYYMM = '200001', '201812'
  startYYYYMM, endYYYYMM = '200001', '201612'


  # END PARAMS

  t0 = fc.timer_start()
  t1 = t0


  data = fc.get_all_data(startYYYYMM, endYYYYMM)
  t0 = fc.timer_restart(t0, 'load data')
  # print(data.keys())
  # print(data)
  # exit()





  # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/



  columns = fc.get_all_X_columns()
  print(columns)

  
  X = data[columns]
  y = data.deathRate
  print(X)
  # print(y)


  t0 = fc.timer_restart(t0, 'data')

  refit = True
  n_splits = 10


  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))

  scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2']
  scoringParam = 'r2'
  results = pd.DataFrame()
  params = { 'max_samples': 0.1, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 170} # , 'max_depth':[None] , 'min_impurity_decrease':[0, 1.8e-7], , 'max_features':list(range(11,X.shape[1]+1))
  # print('params\t', params)

  param_grid = dict()
  for i in params.keys():
    param_grid[i] = [params[i]]

  clf = RandomForestRegressor(random_state=0, n_jobs=DEFAULT_N_JOBS)
  clf.set_params(**params)

  t0 = fc.timer_start()
  clf.fit(X,y)
  t0 = fc.timer_restart(t0, 'clf.fit time')

  importances = pd.DataFrame(dict(name=columns, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
  # print(importances)
  # print(sorted(importances.importance))

  thresholds = [i + 1e-10 for i in sorted(importances.importance) if i < 0.1]
  print(thresholds)

  estimate_time(len(thresholds), param_grid, DEFAULT_N_JOBS, cv_indices)

  # for threshold in [0.02, 0.05]:
  for threshold in thresholds:
    
    # sfm = SelectFromModel(clf, threshold=threshold)
    # t0 = fc.timer_start()
    # sfm.fit(X, y)
    # t0 = fc.timer_restart(t0, 'sfm.fit time')
    # columns_important = [columns[feature_list_index] for feature_list_index in sfm.get_support(indices=True)]
    # X_important = pd.DataFrame(sfm.transform(X), columns=columns_important) # does not perform as well; sfm.transform(X) lowers precision of X ???

    columns_important = list(importances.loc[importances.importance > threshold,'name'])
    X_important = pd.DataFrame(X, columns=columns_important)
    print(len(columns_important), columns_important)
    # print('X_important\n', X_important)
    
    clf_important = RandomForestRegressor(random_state=0) # , n_jobs=DEFAULT_N_JOBS



    # print('param_grid\t', param_grid)

    # exit()
    

    gs = GridSearchCV(clf_important, param_grid=param_grid, refit=False, cv=cv_indices, scoring=scoring, verbose=2, n_jobs=DEFAULT_N_JOBS) # refit=scoringParam, 
    t0 = fc.timer_start()
    gs.fit(X_important, y)
    t0 = fc.timer_restart(t0, 'gs.fit time')
    results_ = pd.DataFrame.from_dict(gs.cv_results_)
    results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1) #  or i == 'params'
    results_['n_features'] = len(columns_important)
    results_['features'] = "['"+ "', '".join(columns_important)+"']"
    results = pd.concat([results, results_])

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
    bestparams['n_jobs'] = DEFAULT_N_JOBS
    clf.set_params(**bestparams)     # should be same as original params
    # print('base_estimator\t',clf.base_estimator_)
    

    X_important = pd.DataFrame(X, columns=columns_important)
    t0 = fc.timer_start()
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