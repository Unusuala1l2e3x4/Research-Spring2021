from numpy.core.numeric import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from mlxtend.evaluate import bias_variance_decomp

import pandas as pd
import numpy as np

import ast, copy

# from time import time

import os, pathlib, random

import importlib
fc = importlib.import_module('functions')


def hasConsecP(l1, l2, n):
  for i1 in range(len(l1) - n + 1):
    for i2 in range(len(l2) - n + 1):
      # print(l1[i1:i1+2], l2[i2:i2+2])
      if l1[i1 : i1 + n] == l2[i2 : i2 + n]:
        return True
  return False
def hasConsecC(l1, l2, n):
  for i1 in range(len(l1) - n + 1):
    for i2 in range(len(l2) - n + 1):
      # print(l1[i1:i1+2], l2[i2:i2+2])
      if set(l1[i1 : i1 + n]) == set(l2[i2 : i2 + n]):
        return True
  return False
def hasConsecCombInExisting(ls, l1, n):
  for l in ls:
    if hasConsecC(l1, l, n):
      return True
  return False
def hasConsecPermInExisting(ls, l1, n):
  for l in ls:
    if hasConsecP(l1, l, n):
      return True
  return False


def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum([i for i in params['max_samples'] if i is not None]) # all floats
  if None in params['max_samples']:
    s += 1
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # multiplier*3600*(folds*s*n*m)/prev_time    # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(1503.79746835443)    # 11440 sesc

  print(hrs, 'hrs, or')
  print(hrs*60, 'min, or')
  print(hrs*3600, 'sec')




def main():
  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  outputDir = os.path.join(pPath, 'random_forest-outfiles')

  # PARAMS

  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8
  DEFAULT_RANDOM_STATE = 0

  startYYYYMM, endYYYYMM = '200001', '201612'

  numMonthLags = 2

  # END PARAMS

  t0 = fc.timer_start()
  t1 = t0


  data = fc.get_all_data(startYYYYMM, endYYYYMM, numMonthLags)
  data = fc.clean_data_before_train_test_split(data)

  t0 = fc.timer_restart(t0, 'load data')
  # print(list(data.keys()))
  # print(data)
  # exit()

  # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/



  # PARAMS

  

  # print(data)
  # columns = fc.get_X_columns(numMonthLags)
  columns = ['STATEFP', 'NPP_g_m-2_1m_lag', 'month', 'months_from_start', 'popuDensity_ALAND_km2', 'GEOID', 'Rh_g_m-2', 'Rh_g_m-2_1m_lag', 'Rh_g_m-2_2m_lag'] # final

  # Iter. 1
  # columns = ['GEOID','month','temp_F','burned_frac','popuDensity_ALAND_km2','precip_in','Rh_g_m-2','pm25_ug_m-3','NPP_g_m-2','C_g_m-2','smallf_frac','BB_g_m-2','ALAND_ATOTAL_ratio','median_inc','PDSI','SP01','DM_kg_m-2','months_from_start']


  

  # assert set(columns).issubset(fc.get_X_columns(numMonthLags))
  print('included:\t',len(columns),columns)
  # print('not included:\t',set(fc.get_X_columns(numMonthLags)) - set(columns))

  # exit()



  refit = True
  do_biasvariancedecomp = False
  n_splits = 10
  min_features_to_select = 9
  train_size = 0.7
  shufflecolumns = True
  numberShuffles = 13 # if shufflecolumns = True

  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2
  # for train_indices, test_indices in cv_indices.split(data):
  #   print('Train: %s | test: %s' % (train_indices, test_indices))


  scoring=['max_error','neg_mean_absolute_percentage_error', 'neg_mean_absolute_error','neg_mean_squared_error','explained_variance', 'r2']
  scoringParam = 'r2'
  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [140]} # , 'max_depth':[None] , 'min_impurity_decrease':[0],
  # print('params\t', params)
  
  

  # END PARAMS



  X, X_test, y, y_test = train_test_split(data[columns], data.mortalityRate, train_size=train_size, random_state=2)
  print(X)
  
  # print(y)
  # exit()

  estimate_time( (26088/19904)*(6890/9049) * (numberShuffles if shufflecolumns else 1) * (len(columns[:15])-min_features_to_select), param_grid, DEFAULT_N_JOBS, cv_indices)
  # exit()

  param_grid_list = ParameterGrid(param_grid)
  results = pd.DataFrame()

  t0 = fc.timer_start()

  columns_list = []

  if shufflecolumns:
    random.seed(678) # 1454
    random.shuffle(columns)
    
    n = 2
    maxitsec = 40
    print('n =',n)
    while len(columns_list) != numberShuffles:
      t = fc.timer_start()
      while hasConsecCombInExisting(columns_list, columns[:15], n):  # RFECV can only handle 15 max without weird behavior
        random.shuffle(columns)
        if fc.timer_elapsed(t) > maxitsec:
          n += 1
          print('n =',n)
          break
        if not {'GEOID','month','months_from_start'}.issubset(set(columns[:15])) or columns[:15] in columns_list: # in the unlikey chance we land on existing ordering
          continue
      columns_list.append(copy.deepcopy(columns[:15]))
      print(columns[:15], '{:.3f}'.format(fc.timer_elapsed(t)))

    t0 = fc.timer_restart(t0, 'create columns_list')

    # for i in columns_list:
    #   print(len(i),i)
    # print()
    # for i in columns_list:
    #   print(len(i), sorted(i))
    # exit()
  else:
    columns_list.append(copy.deepcopy(columns[:15]))

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

      if refit:
        results_['test_set_r2_score'] = rfe.estimator_.score(X_test[columns_important], y_test)

      results_['n_features'] = len(columns_important)
      results_['features'] = "['"+ "', '".join(columns_important)+"']"
      if len(results) == 0:
        for c in sorted(columns):
          results_[c] = NaN
      for i in range(len(columns_i)):
        results_[columns_i[i]] = rfe.ranking_[i]
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

  if refit: # test set r2 score already calculated from rfe
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
    plot = fc.plotGiniImportance(importances)

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