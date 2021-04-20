# based on https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html

import numpy as np
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
  s = np.sum(params['max_samples']) # all floats
  n = np.sum(params['n_estimators']) # all ints
  m = np.prod([len(params[i]) for i in params.keys() if i not in ['max_samples', 'n_estimators']]) # all lengths (int)

  # hrs =  (5/n_jobs)*(folds*s*n*m)/(2010.952902) # multiplier*3600*(folds*s*n*m)/prev_time  # 3600*(10*0.1*170*1)/prev_time
  hrs = multiplier * (5/n_jobs)*(folds*s*n*m)/(1503.79746835443)  # 11440 sesc

  print(hrs, 'hrs, or')
  print(hrs*60, 'min, or')
  print(hrs*3600, 'sec')


def get_scores_for_imputer(name, imputer, X_missing, y_missing):
  estimator = make_pipeline(imputer, regressor)
  gs = GridSearchCV(estimator, param_grid=param_grid, refit=False, cv=cv_indices, scoring=scoring, verbose=2, n_jobs=DEFAULT_N_JOBS)
  t = fc.timer_start()
  gs.fit(X_missing, y_missing)
  t = fc.timer_restart(t, 'fit GridSearchCV '+name)
  results = pd.DataFrame.from_dict(gs.cv_results_)
  results['impute_method'] = name
  results = results.drop([i for i in results.keys() if i.startswith('split')], axis=1) #  or i == 'params'
  results['n_features'] = len(columns)
  results['features'] = "['"+ "', '".join(columns)+"']"
  

  print(results)


  # impute_scores = cross_val_score(estimator, X_missing, y_missing,scoring=scoringParam,cv=cv_indices, n_jobs=DEFAULT_N_JOBS)
  return results
def get_impute_zero_score(X_missing, y_missing):
  imputer = SimpleImputer(missing_values=np.nan, add_indicator=True,strategy='constant', fill_value=0)
  zero_impute_scores = get_scores_for_imputer('zero', imputer, X_missing, y_missing)
  # return zero_impute_scores.mean(), zero_impute_scores.std()
  return zero_impute_scores

def get_impute_knn_score(X_missing, y_missing):
  imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
  knn_impute_scores = get_scores_for_imputer('knn', imputer, X_missing, y_missing)
  # return knn_impute_scores.mean(), knn_impute_scores.std()
  return knn_impute_scores

def get_impute_mean(X_missing, y_missing):
  imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
  mean_impute_scores = get_scores_for_imputer('mean', imputer, X_missing, y_missing)
  # return mean_impute_scores.mean(), mean_impute_scores.std()
  return mean_impute_scores

def get_impute_iterative(X_missing, y_missing):
  imputer = IterativeImputer(missing_values=np.nan, add_indicator=True,random_state=3, n_nearest_features=5,sample_posterior=True)
  iterative_impute_scores = get_scores_for_imputer('iterative', imputer, X_missing, y_missing)
  # return iterative_impute_scores.mean(), iterative_impute_scores.std()
  return iterative_impute_scores



if __name__ == "__main__":
  rng = np.random.RandomState(42)

  # X, y = fetch_california_housing(return_X_y=True)
  # X = X[:400]
  # y = y[:400]

  pPath = str(pathlib.Path(__file__).parent.absolute())
  ppPath = str(pathlib.Path(__file__).parent.parent.absolute())
  outputDir = os.path.join(pPath, 'plot_missing_values-outfiles')
  epaAqDir = os.path.join(ppPath, 'EPA AQ data')

  # PARAMS
  ext = 'hdf5'

  n_splits = 10
  crossvalidate = True
  DEFAULT_N_JOBS = 5 # 4-6; avoid 7, 8
  DEFAULT_RANDOM_STATE = 0


  

  startYYYYMM, endYYYYMM = '200001', '201612'


  t0 = fc.timer_start()
  t1 = t0
  # END PARAMS

  cv_indices = KFold(n_splits=n_splits, shuffle=True, random_state=1) # DEFAULT_N_JOBS*2

  data = fc.get_all_data(startYYYYMM, endYYYYMM)
  t0 = fc.timer_restart(t0, 'load data')
  print(sorted(data.keys()))
  print(data)


  columns = fc.get_all_X_columns()

  print(sorted(columns))

  print('not included:\t',set(data.keys()) - set(columns))


  X, X_test, y, y_test = train_test_split(data[columns], data.deathRate, train_size=0.7, random_state=2)

  # PARAMS
  crossvalidate = True
  refit = True
  do_biasvariancedecomp = False

  scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2'] # 'explained_variance',
  scoringParam = 'r2'

  param_grid = { 'max_samples': [0.1], 'min_samples_leaf': [2], 'min_samples_split': [4], 'n_estimators': [170] } 
  param_grid_list = ParameterGrid(param_grid)
  # print(list(param_grid_list))
  # END PARAMS

  X_miss, y_miss = X, y
  params = list(param_grid_list)[0]
  print(params)

  regressor = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE) # , n_jobs=DEFAULT_N_JOBS
  regressor.set_params(**params)

  # exit()
  aqiFiles = sorted([i for i in os.listdir(epaAqDir) if i.startswith('TENA_county_AQI') and i.endswith(ext)])

  # print(aqiFiles)

  x_labels = ['Zero imputation','KNN Imputation','Mean Imputation','Iterative Imputation'] # 'Full data',
  estimate_time(len(x_labels) * len(aqiFiles), param_grid, DEFAULT_N_JOBS, cv_indices)

  dates = None


  t0 = fc.timer_start()

  # exit()
  for file in aqiFiles:
    fileArgs = ('AQI', epaAqDir, file.split('.')[0])
    df = fc.read_df(fileArgs[1], fileArgs[2], ext)
    if dates is None:
      dates = sorted(i for i in df if i != 'GEOID' and i >= startYYYYMM and i <= endYYYYMM)
    df = df.loc[:,['GEOID']+dates]
    data[fileArgs[0]] = np.ravel(df.loc[:,dates], order='F')

    # print()

    # continue


    gridcv_results = pd.DataFrame()
    gridcv_results = pd.concat([gridcv_results, get_impute_zero_score (X_miss, y_miss)],  ignore_index=True)
    gridcv_results = pd.concat([gridcv_results, get_impute_knn_score  (X_miss, y_miss)],  ignore_index=True)
    gridcv_results = pd.concat([gridcv_results, get_impute_mean       (X_miss, y_miss)],  ignore_index=True)
    gridcv_results = pd.concat([gridcv_results, get_impute_iterative  (X_miss, y_miss)],  ignore_index=True)

    t0 = fc.timer_restart(t0, 'scoring - '+file)
    
    for s in scoring:
      gridcv_results['rank_test_'+s] = fc.rank_decreasing(gridcv_results['mean_test_'+s])
    print(gridcv_results)

    saveTime = fc.utc_time_filename()
    fc.save_df(gridcv_results, outputDir, 'imputer_results_' +  + saveTime, 'csv')

    for sp in scoring:
      mses = np.zeros(len(x_labels))
      stds = np.zeros(len(x_labels))

      for i in range(len(gridcv_results)):
        mses[i] = float(gridcv_results.loc[i, 'mean_test_'+sp])
        stds[i] = float(gridcv_results.loc[i, 'std_test_'+sp])
      mses = mses * -1

      n_bars = len(mses)
      xval = np.arange(n_bars)
      colors = ['r', 'g', 'b', 'orange', 'black']
      title = 'Imputation Techniques - '+ sp + '_limit'+re.split('.'+ext+'|limit',file)[1]

      plt.figure(figsize=(12, 6))
      ax = plt.subplot(122)
      for j in xval:
        ax.barh(j, mses[j], xerr=stds[j], color=colors[j], alpha=0.6, align='center')
      
      ax.set_title(title)
      ax.set_yticks(xval)
      ax.set_xlabel(sp)
      ax.invert_yaxis()
      ax.set_yticklabels(x_labels)
      ax.set_aspect('auto')

      fc.save_plt(plt,outputDir, title+'_'+saveTime, 'png')

      plt.show()
      plt.close()
    
    t0 = fc.timer_restart(t0, 'saving results - '+file)
      

    # You can also try different techniques. For instance, the median is a more
    # robust estimator for data with high magnitude variables which could dominate
    # results (otherwise known as a 'long tail').

  t1 = fc.timer_restart(t1, 'total time')
