# based on https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html

import numpy as np
from numpy.core.numeric import NaN
from numpy.lib.npyio import save, savetxt
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
# from sklearn.datasets import fetch_california_housing
import os, pathlib, re
import importlib
fc = importlib.import_module('functions')
import pandas as pd
import geopandas as gpd

from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt


def estimate_time(multiplier, params, n_jobs, cv_indices): # assuming n_jobs = DEFAULT_N_JOBS = 5
  assert [type(i) == float for i in params['max_samples']]
  folds = cv_indices.get_n_splits()
  s = np.sum([i for i in params['max_samples'] if i is not None]) # all floats
  if None in params['max_samples']:
    s += 1
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # multiplier*3600*(folds*s*n*m)/prev_time  # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(1503.79746835443)  # 11440 sesc

  hrs += (120/3600)*len(aqiFiles)

  print(hrs, 'hrs, or')
  print(hrs*60, 'min, or')
  print(hrs*3600, 'sec')


def get_scores_for_imputer(name, imputer, X_missing, y_missing):
  columns_i = list(X_missing.columns)

  # estimator = make_pipeline(imputer, regressor)
  t = fc.timer_start()
  X_transform = imputer.fit_transform(X_missing,y_missing)
  t = fc.timer_restart(t, 'fit imputer '+name)

  gs = GridSearchCV(regressor, param_grid=param_grid, refit=False, cv=cv_indices, scoring=scoring, verbose=1, n_jobs=DEFAULT_N_JOBS)
  gs.fit(X_transform, y_missing)
  t = fc.timer_restart(t, 'fit GridSearchCV '+name)
  results_ = pd.DataFrame.from_dict(gs.cv_results_)
  results_['impute_method'] = name
  results_ = results_.drop([i for i in results_.keys() if i.startswith('split')], axis=1) #  or i == 'params'
  # if refit:
  #   results_['test_set_r2'] = NaN
  #   results_['rank_test_set_r2'] = NaN

  for i in results_.index:
    if refit:
      params_i = dict(results_.loc[i,'params'])

      clf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, n_jobs=DEFAULT_N_JOBS)
      clf.set_params(**params_i)
      clf.fit(X_transform,y_missing)

      importances = pd.DataFrame(dict(name=columns_i, importance=clf.feature_importances_)).sort_values('importance', ascending=False) # .head(10)
      importances_list.append(importances)
      # print(importances)
    
      # results_.loc[i,'test_set_r2'] = clf.score(X_test, y_test)

    results_['n_features'] = len(columns_i)
    results_['features'] = "['"+ "', '".join(columns_i)+"']"

  return results_


def get_impute_zero_score(X_missing, y_missing):
  imputer = SimpleImputer(missing_values=np.nan, add_indicator=False,strategy='constant', fill_value=0)
  zero_impute_scores = get_scores_for_imputer('zero', imputer, X_missing, y_missing)
  # return zero_impute_scores.mean(), zero_impute_scores.std()
  return zero_impute_scores

def get_impute_knn_score(X_missing, y_missing):
  imputer = KNNImputer(missing_values=np.nan, add_indicator=False)
  knn_impute_scores = get_scores_for_imputer('knn', imputer, X_missing, y_missing)
  # return knn_impute_scores.mean(), knn_impute_scores.std()
  return knn_impute_scores

def get_impute_mean(X_missing, y_missing):
  imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=False)
  mean_impute_scores = get_scores_for_imputer('mean', imputer, X_missing, y_missing)
  # return mean_impute_scores.mean(), mean_impute_scores.std()
  return mean_impute_scores

def get_impute_iterative(X_missing, y_missing):
  imputer = IterativeImputer(missing_values=np.nan, add_indicator=False,random_state=3,sample_posterior=True, skip_complete=True) #  n_nearest_features=5,
  iterative_impute_scores = get_scores_for_imputer('iterative', imputer, X_missing, y_missing)
  # return iterative_impute_scores.mean(), iterative_impute_scores.std()
  return iterative_impute_scores



if __name__ == "__main__":
  rng = np.random.RandomState(42)


  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  outputDir = os.path.join(pPath, 'impute_county_month_AQI-outfiles')
  epaAqDir = os.path.join(ppPath, 'EPA AQ data')

  # PARAMS
  ext = 'hdf5'

  n_splits = 10 # TEST
  train_size = 0.7
  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8
  DEFAULT_RANDOM_STATE = 0

  startYYYYMM, endYYYYMM = '200001', '201612'

  numMonthLags = 1


  crossvalidate = True
  refit = True #  X_test isnt transformed by imputer
  save_best_refit = True
  do_biasvariancedecomp = False

  scoring=['neg_mean_absolute_error','neg_mean_squared_error','explained_variance','r2']
  scoringParam = 'r2'

  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [140] }  # TEST
  param_grid_list = ParameterGrid(param_grid)
  # print(list(param_grid_list))
  # END PARAMS
  params = list(param_grid_list)[0]
  print(params)

  t0 = fc.timer_start()
  t1 = t0

  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2

  data_all = fc.get_all_data(startYYYYMM, endYYYYMM, numMonthLags)
  # data = fc.clean_data_before_train_test_split(data_all)      # call after adding AQI data

  t0 = fc.timer_restart(t0, 'load data')
  # print(sorted(data_all.keys()))
  # print(data)
  # exit()

  # columns = fc.get_X_columns(numMonthLags)
  columns = ['popuDensity_ALAND_km2', 'month', 'temp_F', 'median_inc', 'year', 'PDSI', 'GEOID', 'Rh_g_m-2', 'SP01', 'months_from_start']


  print(sorted(columns))
  print('not included:\t',set(data_all.keys()) - set(columns))
  assert set(columns).issubset(data_all.keys())


  regressor = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE) # , n_jobs=DEFAULT_N_JOBS
  regressor.set_params(**params)

  # exit()
  aqiFiles = sorted([i for i in os.listdir(epaAqDir) if i.startswith('TENA_county_AQI') and i.endswith(ext)]) # [4:6] # TEST

  # for i in aqiFiles:
  #   print(i)
  # exit()
                          # 
  x_labels = ['Zero imputation','KNN Imputation','Mean Imputation','Iterative Imputation']
  estimate_time(len(x_labels)* len(aqiFiles)* 0.85, param_grid, DEFAULT_N_JOBS, cv_indices)
  # exit()

  dates, prev = None, None
  importances_list=[]
  gridcv_results = pd.DataFrame()


  t0 = fc.timer_start()
  saveTime = fc.utc_time_filename()
  for file in aqiFiles:                                    # best: 'limit-0_linear'
    keyword = 'limit'+re.split('.'+ext+'|limit',file)[1]
    print(keyword)
    # print(file)
    # continue

    fileArgs = ('AQI_'+keyword, epaAqDir, file.split('.')[0])
    df = fc.read_df(fileArgs[1], fileArgs[2], ext)

    if dates is None:
      dates = sorted(i for i in df if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
    df = df.loc[:,['GEOID']+dates]

    if prev is not None:
      del data_all[prev]
      prev = fileArgs[0]
    
    data_all[fileArgs[0]] = np.ravel(df.loc[:,dates], order='F') 
    data = fc.clean_data_before_train_test_split(data_all) # only removes where y is NaN

    X, X_test, y, y_test = train_test_split(data[columns+[fileArgs[0]]], data.mortalityRate, train_size=train_size, random_state=2)

    
    results = pd.DataFrame()
    results = pd.concat([results, get_impute_zero_score (X, y)],  ignore_index=True)
    results = pd.concat([results, get_impute_knn_score  (X, y)],  ignore_index=True)
    results = pd.concat([results, get_impute_mean       (X, y)],  ignore_index=True)
    results = pd.concat([results, get_impute_iterative  (X, y)],  ignore_index=True)

    t0 = fc.timer_restart(t0, 'scoring - '+file)
  
    for sp in scoring:
      mses = np.zeros(len(x_labels))
      stds = np.zeros(len(x_labels))

      for i in range(len(results)):
        mses[i] = float(results.loc[i, 'mean_test_'+sp])
        stds[i] = float(results.loc[i, 'std_test_'+sp])

      n_bars = len(mses)
      xval = np.arange(n_bars)
      colors = ['r', 'g', 'b', 'orange', 'black']
      title = sp + '_'+file.split('.')[0]

      plt.figure(figsize=(12, 6))
      ax = plt.subplot(122)
      for j in xval:
        ax.barh(j, mses[j], xerr=stds[j], color=colors[j], alpha=0.6, align='center')
      
      title = 'Imputation Techniques - '+ sp + '_'+ fileArgs[0]
      ax.set_title(title)
      ax.set_yticks(xval)
      ax.set_xlabel(sp)
      ax.invert_yaxis()
      ax.set_yticklabels(x_labels)
      ax.set_aspect('auto')

      fc.save_plt(plt,outputDir, title+'_'+saveTime, 'png')

      # plt.show()
      plt.close()
    
    t0 = fc.timer_restart(t0, 'saving results - '+file)

    gridcv_results = pd.concat([gridcv_results, results],  ignore_index=True)
        

  # You can also try different techniques. For instance, the median is a more
  # robust estimator for data with high magnitude variables which could dominate
  # results (otherwise known as a 'long tail').

  for s in scoring:
    gridcv_results['rank_test_'+s] = fc.rank_decreasing(gridcv_results['mean_test_'+s])
  # if refit:
  #   gridcv_results['rank_test_set_r2'] = fc.rank_decreasing(gridcv_results['test_set_r2'])
  print(gridcv_results)

  fc.save_df(gridcv_results, outputDir, 'imputer_results_' + saveTime, 'csv')
  

  if save_best_refit:
    # get index
    i = int(gridcv_results.loc[gridcv_results['rank_test_'+scoringParam] == 1,:].index[0])

    print(i)

    print(len(importances_list))

    importances = importances_list[i]
    print(importances)
    plot = fc.plotGiniImportance(importances)
    fc.save_plt(plot, outputDir, 'best_GiniImportance_RF_' + saveTime, 'png')
    fc.save_df(importances, outputDir, 'best_GiniImportance_RF_' + saveTime, 'csv')


    
  



  
  t1 = fc.timer_restart(t1, 'total time')
