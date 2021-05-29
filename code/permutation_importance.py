
from numpy.core.numeric import NaN
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import pandas as pd
import numpy as np

import os, pathlib

import importlib
fc = importlib.import_module('functions')




def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum([i for i in params['max_samples'] if i is not None]) # all floats
  if None in params['max_samples']:
    s += 1
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

  outputDir = os.path.join(pPath, 'spearman-outfiles')

  # startYYYYMM, endYYYYMM = sys.argv[1], sys.argv[2]

  # startYYYYMM, endYYYYMM = '200001', '201812'
  startYYYYMM, endYYYYMM = '200001', '201612'

  numMonthLags = 2
  train_size = 0.7




  data = fc.get_all_data(startYYYYMM, endYYYYMM, numMonthLags)
  data = fc.clean_data_before_train_test_split(data)

  t0 = fc.timer_restart(t0, 'load data')

  
  # print(list(data.keys()))
  # print(data)
  # exit()

  

  columns_list = []
  # columns_list.append(fc.get_lags_from_columns(['GEOID', 'NPP_g_m-2','Rh_g_m-2', 'PDSI', 'SP01', 'STATEFP', 'median_inc', 'month', 'months_from_start', 'pm25_ug_m-3', 'popuDensity_ALAND_km2', 'temp_F'], numMonthLags))
  # columns_list.append(fc.get_lags_from_columns(['NPP_g_m-2','Rh_g_m-2', 'PDSI', 'SP01', 'median_inc', 'month', 'months_from_start', 'pm25_ug_m-3', 'popuDensity_ALAND_km2', 'temp_F'], numMonthLags))
  columns_list.append(fc.get_spearman_columns(numMonthLags))


  # print(sorted(columns_list[0]))

  temp = ['deathRate', 'popuDensity_ALAND_km2', 'popuDensity_ATOTAL_km2', 'ALAND_ATOTAL_ratio', 'month', 'months_from_start','median_inc']


  for columns in columns_list:
    # assert set(columns).issubset(fc.get_X_columns(numMonthLags))
    columns = sorted([i for i in columns if i not in temp]) + sorted([i for i in columns if i in temp])
    print('included:\t',len(columns),columns)

    
    # continue

    # print('not included:\t',set(fc.get_X_columns(numMonthLags)) - set(columns))
    # print('not included (from all of data):\t',set(data.keys()) - set(columns_list[0]))

    # X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=train_size, random_state=DEFAULT_TRAIN_TEST_SPLIT_RANDOM_STATE)
    X = data[columns]
    print(X)

    # print(X.isna().any())


    saveTime = fc.utc_time_filename()
    fc.save_df(df_rho, outputDir, 'correlation_' + saveTime, 'csv')
    fc.save_df(df_pval, outputDir, 'pvalue_' + saveTime, 'csv')








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